import os
import torch
import logging
import argparse
import json
import gc
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
from einops import rearrange
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDIMScheduler
from torch.utils.data import Dataset

from genphoto.pipelines.pipeline_animation import GenPhotoPipeline
from genphoto.pipelines.pipeline_inversion import GenPhotoInversionPipeline
from genphoto.models.unet import UNet3DConditionModelCameraCond
from genphoto.models.camera_adaptor import CameraCameraEncoder, CameraAdaptor
from genphoto.utils.util import save_videos_grid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. Embedding Helper Functions
# ==========================================

def create_bokehK_embedding(bokehK_values, target_height, target_width):
    bokehK_values = bokehK_values.cpu().float()
    f = bokehK_values.shape[0]
    bokehK_embedding = torch.zeros((f, 3, target_height, target_width), dtype=torch.float32)
    
    for i in range(f):
        K_value = bokehK_values[i].item()
        kernel_size = max(K_value, 1.0)
        sigma = K_value / 3.0

        ax = np.linspace(-(kernel_size / 2), kernel_size / 2, int(np.ceil(kernel_size)))
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= np.sum(kernel)
        scale = kernel[int(np.ceil(kernel_size) / 2), int(np.ceil(kernel_size) / 2)]
        
        bokehK_embedding[i] = scale
    
    return bokehK_embedding

def create_focal_length_embedding(focal_length_values, target_height, target_width, base_focal_length=24.0, sensor_height=24.0, sensor_width=36.0):
    device = 'cpu'
    focal_length_values = focal_length_values.to(device).float()
    f = focal_length_values.shape[0]

    sensor_width_t = torch.tensor(sensor_width, device=device)
    sensor_height_t = torch.tensor(sensor_height, device=device)
    base_focal_length_t = torch.tensor(base_focal_length, device=device)

    base_fov_x = 2.0 * torch.atan(sensor_width_t * 0.5 / base_focal_length_t)
    base_fov_y = 2.0 * torch.atan(sensor_height_t * 0.5 / base_focal_length_t)

    target_fov_x = 2.0 * torch.atan(sensor_width_t * 0.5 / focal_length_values)
    target_fov_y = 2.0 * torch.atan(sensor_height_t * 0.5 / focal_length_values)

    crop_ratio_xs = target_fov_x / base_fov_x
    crop_ratio_ys = target_fov_y / base_fov_y

    center_h, center_w = target_height // 2, target_width // 2
    focal_length_embedding = torch.zeros((f, 3, target_height, target_width), dtype=torch.float32, device=device)

    for i in range(f):
        idx_val_y = crop_ratio_ys[i] if crop_ratio_ys.ndim == 1 else crop_ratio_ys[i].item()
        idx_val_x = crop_ratio_xs[i] if crop_ratio_xs.ndim == 1 else crop_ratio_xs[i].item()

        crop_h = torch.round(torch.tensor(idx_val_y) * target_height).int().item()
        crop_w = torch.round(torch.tensor(idx_val_x) * target_width).int().item()

        crop_h = max(1, min(target_height, crop_h))
        crop_w = max(1, min(target_width, crop_w))

        focal_length_embedding[i, :,
            center_h - crop_h // 2: center_h + crop_h // 2,
            center_w - crop_w // 2: center_w + crop_w // 2] = 1.0

    return focal_length_embedding

def kelvin_to_rgb(kelvin):
    if torch.is_tensor(kelvin):
        kelvin = kelvin.cpu().item()  
    temp = kelvin / 100.0
    if temp <= 66:
        red = 255
        green = 99.4708025861 * np.log(temp) - 161.1195681661 if temp > 0 else 0
        if temp <= 19:
            blue = 0
        else:
            blue = 138.5177312231 * np.log(temp - 10) - 305.0447927307
    elif 66 < temp <= 88:
        red = 0.5 * (255 + 329.698727446 * ((temp - 60) ** -0.19332047592))
        green = 0.5 * (288.1221695283 * ((temp - 60) ** -0.1155148492) + 
                       (99.4708025861 * np.log(temp) - 161.1195681661 if temp > 0 else 0))
        blue = 0.5 * (138.5177312231 * np.log(temp - 10) - 305.0447927307 + 255)
    else:
        red = 329.698727446 * ((temp - 60) ** -0.19332047592)
        green = 288.1221695283 * ((temp - 60) ** -0.1155148492)
        blue = 255
    return np.array([red, green, blue], dtype=np.float32) / 255.0

def create_color_temperature_embedding(color_temperature_values, target_height, target_width, min_color_temperature=2000, max_color_temperature=10000):
    values = color_temperature_values.cpu()
    f = values.shape[0]
    rgb_factors = []
    iter_values = values.view(-1)
    
    for val_tensor in iter_values:
        val = val_tensor.item()
        # Replicating the logic from training/original inference which treats input as normalized
        # even if it is raw Kelvin. This results in a specific embedding (likely solid blue for raw Kelvin)
        # that the model learned to expect.
        kelvin = min_color_temperature + (val * (max_color_temperature - min_color_temperature))
        
        rgb = kelvin_to_rgb(kelvin)
        rgb_factors.append(rgb)
    
    rgb_factors = torch.tensor(np.array(rgb_factors)).float()
    rgb_factors = rgb_factors.unsqueeze(2).unsqueeze(3)
    color_temperature_embedding = rgb_factors.expand(f, 3, target_height, target_width)
    return color_temperature_embedding

def create_shutter_speed_embedding(shutter_speed_values, target_height, target_width, base_exposure=0.5):
    shutter_speed_values = shutter_speed_values.cpu().float()
    f = shutter_speed_values.shape[0]
    fwc = 32000.0
    scales = (shutter_speed_values / base_exposure) * (fwc / (fwc + 0.0001))
    scales = scales.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(f, 3, target_height, target_width)
    return scales

# ==========================================
# 2. Universal Camera Embedding Class
# ==========================================

class Universal_Camera_Embedding(Dataset):
    def __init__(self, setting_type, values, tokenizer, text_encoder, device, sample_size=[256, 384]):
        self.setting_type = setting_type
        self.values = values.to(device).float()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device  
        self.sample_size = sample_size

    def load(self):
        prompts = []
        vals_flat = self.values.view(-1)
        
        for v in vals_flat:
            val_item = v.item()
            if self.setting_type == 'bokeh':
                prompt = f"<bokeh kernel size: {val_item}>"
            elif self.setting_type == 'focal':
                prompt = f"<focal length: {val_item}>"
            elif self.setting_type == 'shutter':
                prompt = f"<exposure: {val_item}>"
            elif self.setting_type == 'color':
                prompt = f"<color temperature: {val_item}>"
            else:
                raise ValueError(f"Unknown setting type: {self.setting_type}")
            prompts.append(prompt)

        with torch.no_grad():
            prompt_ids = self.tokenizer(
                prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            encoder_hidden_states = self.text_encoder(input_ids=prompt_ids).last_hidden_state

        differences = []
        for i in range(1, encoder_hidden_states.size(0)):
            diff = encoder_hidden_states[i] - encoder_hidden_states[i - 1]
            differences.append(diff.unsqueeze(0))
        
        if encoder_hidden_states.size(0) > 0:
            diff = encoder_hidden_states[-1] - encoder_hidden_states[0]
            differences.append(diff.unsqueeze(0))
            concatenated_differences = torch.cat(differences, dim=0)
        else:
            concatenated_differences = torch.zeros_like(encoder_hidden_states)

        pad_len = 128 - concatenated_differences.size(1)
        if pad_len > 0:
            concatenated_differences = F.pad(concatenated_differences, (0, 0, 0, pad_len))

        frame = concatenated_differences.size(0)
        ccl_embedding = concatenated_differences.reshape(frame, self.sample_size[0], self.sample_size[1])
        ccl_embedding = ccl_embedding.unsqueeze(1).expand(-1, 3, -1, -1).to(self.device)

        values_cpu = self.values.cpu()
        if self.setting_type == 'bokeh':
            vis_emb = create_bokehK_embedding(values_cpu, self.sample_size[0], self.sample_size[1])
        elif self.setting_type == 'focal':
            vis_emb = create_focal_length_embedding(values_cpu, self.sample_size[0], self.sample_size[1])
        elif self.setting_type == 'shutter':
            vis_emb = create_shutter_speed_embedding(values_cpu, self.sample_size[0], self.sample_size[1])
        elif self.setting_type == 'color':
            vis_emb = create_color_temperature_embedding(values_cpu, self.sample_size[0], self.sample_size[1])
        
        vis_emb = vis_emb.to(self.device)
        camera_embedding = torch.cat((vis_emb, ccl_embedding), dim=1)
        return camera_embedding

# ==========================================
# 3. Model Loading (Single Model)
# ==========================================

def load_model_for_setting(setting_type, method):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config_map = {
        'bokeh': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_bokehK.yaml',
        'focal': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_focal_length.yaml',
        'shutter': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_shutter_speed.yaml',
        'color': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_color_temperature.yaml'
    }
    
    cfg = OmegaConf.load(config_map[setting_type])
    
    # 1. Load VAE, Tokenizer, Text Encoder
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_path, subfolder="text_encoder").to(device)
    text_encoder.requires_grad_(False)
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(cfg.noise_scheduler_kwargs))

    # 2. Load UNet
    unet = UNet3DConditionModelCameraCond.from_pretrained_2d(
        cfg.pretrained_model_path,
        subfolder=cfg.unet_subfolder,
        unet_additional_kwargs=cfg.unet_additional_kwargs
    ).to(device)
    unet.requires_grad_(False)

    # 3. Load Camera Adaptor
    camera_encoder = CameraCameraEncoder(**cfg.camera_encoder_kwargs).to(device)
    camera_encoder.requires_grad_(False)
    camera_adaptor = CameraAdaptor(unet, camera_encoder)
    camera_adaptor.requires_grad_(False)
    camera_adaptor.to(device)

    logger.info("Setting the attention processors")
    unet.set_all_attn_processor(
        add_spatial_lora=cfg.lora_ckpt is not None,
        add_motion_lora=cfg.motion_lora_rank > 0,
        lora_kwargs={"lora_rank": cfg.lora_rank, "lora_scale": cfg.lora_scale},
        motion_lora_kwargs={"lora_rank": cfg.motion_lora_rank, "lora_scale": cfg.motion_lora_scale},
        **cfg.attention_processor_kwargs
    )

    # 4. Load Weights
    if cfg.lora_ckpt is not None:
        logger.info(f"Loading LoRA from {cfg.lora_ckpt}")
        lora_checkpoints = torch.load(cfg.lora_ckpt, map_location=unet.device)
        if 'lora_state_dict' in lora_checkpoints.keys():
            lora_checkpoints = lora_checkpoints['lora_state_dict']
        unet.load_state_dict(lora_checkpoints, strict=False)

    if cfg.motion_module_ckpt is not None:
        logger.info(f"Loading Motion Module from {cfg.motion_module_ckpt}")
        mm_checkpoints = torch.load(cfg.motion_module_ckpt, map_location=unet.device)
        if 'state_dict' in mm_checkpoints: mm_checkpoints = mm_checkpoints['state_dict']
        unet.load_state_dict(mm_checkpoints, strict=False)

    if cfg.camera_adaptor_ckpt is not None:
        logger.info(f"Loading Camera Adaptor from {cfg.camera_adaptor_ckpt}")
        ckpt = torch.load(cfg.camera_adaptor_ckpt, map_location=device)
        camera_adaptor.camera_encoder.load_state_dict(ckpt['camera_encoder_state_dict'], strict=False)
        camera_adaptor.unet.load_state_dict(ckpt['attention_processor_state_dict'], strict=False)

    if method == 'DDIM_Inversion':
        logger.info("Using GenPhotoInversionPipeline for DDIM Inversion")
        pipeline = GenPhotoInversionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=noise_scheduler,
            camera_encoder=camera_encoder
        ).to(device)
    else:
        logger.info("Using GenPhotoPipeline for SDEdit")
        pipeline = GenPhotoPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=noise_scheduler,
            camera_encoder=camera_encoder
        ).to(device)
        
    pipeline.enable_vae_slicing()
    
    return pipeline, device

def preprocess_image_pil(pil_image, height, width):
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    ])
    return transform(pil_image).unsqueeze(0)

# ==========================================
# 4. Inference Logic (Tree Expansion)
# ==========================================

def run_inference(multi_params, input_image_path, strength, output_dir, base_scene, method):
    # multi_params: {'bokeh': [1, 5], 'color': [3000, 5000]}
    
    height = 256
    width = 384
    # Separate output directory by method
    output_dir = os.path.join(output_dir, method)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Input Pool
    # Each item: {'image': PIL.Image, 'id': str}
    init_pil = Image.open(input_image_path).convert("RGB")
    current_inputs = [{'image': init_pil, 'id': 'init'}]
    
    # Iterate through each setting sequentially
    for stage_idx, (setting_type, values) in enumerate(multi_params.items()):
        logger.info(f"==========================================")
        logger.info(f"Processing Stage {stage_idx+1}: {setting_type} (Values: {values})")
        logger.info(f"Method: {method}")
        logger.info(f"Input Pool Size: {len(current_inputs)}")
        logger.info(f"==========================================")
        
        # 1. Load Model for this specific setting
        pipeline, device = load_model_for_setting(setting_type, method)
        
        # 2. Prepare Values
        val_tensor = torch.tensor(values).unsqueeze(1)
        
        # 3. Prepare Embedding (Shared for all inputs in this stage)
        embed_obj = Universal_Camera_Embedding(
            setting_type, val_tensor, pipeline.tokenizer, pipeline.text_encoder, device, sample_size=[height, width]
        )
        camera_embedding = embed_obj.load()
        camera_embedding = rearrange(camera_embedding.unsqueeze(0), "b f c h w -> b c f h w")
        
        next_stage_inputs = []
        
        # 4. Process each input image
        for input_idx, item in enumerate(current_inputs):
            input_pil = item['image']
            input_id = item['id']
            
            logger.info(f"  > Processing Input {input_idx+1}/{len(current_inputs)} (ID: {input_id})...")
            
            # Preprocess Image
            init_tensor = preprocess_image_pil(input_pil, height, width).to(device)
            
            if method == 'SDEdit':
                # Run Pipeline (I2V / SDEdit)
                with torch.no_grad():
                    output = pipeline(
                        prompt=base_scene,
                        camera_embedding=camera_embedding,
                        video_length=len(values),
                        height=height,
                        width=width,
                        num_inference_steps=25,
                        guidance_scale=8.0,
                        image=init_tensor,
                        strength=strength
                    )
            elif method == 'DDIM_Inversion':
                # Determine Source Value for Inversion
                if setting_type == 'focal':
                    source_val_item = 24.0 
                elif setting_type == 'color':
                    source_val_item = 5500.0
                elif setting_type == 'shutter':
                    source_val_item = 0.5 
                elif setting_type == 'bokeh':
                    source_val_item = values[0] # Use first target value as source approximation
                else:
                    source_val_item = values[0]
                
                source_vals = torch.tensor([source_val_item] * len(values), dtype=torch.float32)
                source_embed_obj = Universal_Camera_Embedding(
                    setting_type, source_vals, pipeline.tokenizer, pipeline.text_encoder, device, sample_size=[height, width]
                )
                source_embed = source_embed_obj.load()
                source_embed = rearrange(source_embed.unsqueeze(0), "b f c h w -> b c f h w")
                
                # Run Inversion
                logger.info(f"    Running Inversion (Source Val: {source_val_item})...")
                inverted_latents = pipeline.invert(
                    image=init_tensor,
                    prompt=base_scene,
                    camera_embedding=source_embed,
                    num_inference_steps=25,
                    video_length=len(values)
                )
                
                # Run Generation
                guidance_scale = 2.0 if setting_type == 'color' else 1.5
                logger.info(f"    Running Generation (Guidance: {guidance_scale})...")
                
                with torch.no_grad():
                    output = pipeline(
                        prompt=base_scene,
                        camera_embedding=camera_embedding,
                        video_length=len(values),
                        height=height,
                        width=width,
                        num_inference_steps=25,
                        guidance_scale=guidance_scale,
                        latents=inverted_latents
                    )

            # Extract Video
            video_tensor = output.videos.squeeze(0) # [3, F, H, W]
            video_tensor = video_tensor.permute(1, 0, 2, 3) # [F, 3, H, W]
            
            # Convert to PIL frames
            frames = []
            for i in range(video_tensor.shape[0]):
                frame_tensor = video_tensor[i]
                ndarr = frame_tensor.permute(1, 2, 0).cpu().numpy()
                ndarr = (ndarr * 255).astype(np.uint8)
                frames.append(Image.fromarray(ndarr))
            
            # Save GIF for this input
            # Naming: stage{N}_{setting}_{prev_id}.gif
            # If prev_id is 'init', name is stage1_bokeh_init.gif
            # If prev_id is 'init_0', name is stage2_color_init_0.gif
            save_name = f"stage{stage_idx+1}_{setting_type}_{input_id}.gif"
            save_path = os.path.join(output_dir, save_name)
            
            frames[0].save(
                save_path,
                save_all=True,
                append_images=frames[1:],
                duration=200,
                loop=0
            )
            logger.info(f"    Saved result to {save_path}")
            
            # Collect frames for next stage
            for f_idx, frame in enumerate(frames):
                next_id = f"{input_id}_{f_idx}" if input_id != 'init' else f"{f_idx}"
                next_stage_inputs.append({'image': frame, 'id': next_id})
        
        # Update input pool for next stage
        current_inputs = next_stage_inputs
        
        # Cleanup Model
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Model unloaded.")

    logger.info("All stages completed.")

# ==========================================
# 5. Main Function
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_params", type=str, required=True, help="JSON dict of settings")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--strength", type=float, default=0.75, help="Denoising strength (SDEdit only)")
    parser.add_argument("--output_dir", type=str, default="outputs/unified_inference", help="Output directory")
    parser.add_argument("--base_scene", type=str, required=True, help="Prompt text")
    parser.add_argument("--method", type=str, default="SDEdit", choices=["SDEdit", "DDIM_Inversion"], help="Inference method")

    args = parser.parse_args()
    
    try:
        params_dict = json.loads(args.multi_params)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for --multi_params")

    run_inference(
        params_dict,
        args.input_image,
        args.strength,
        args.output_dir,
        args.base_scene,
        args.method
    )

if __name__ == "__main__":
    main()
