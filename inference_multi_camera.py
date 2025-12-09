import os
import torch
import logging
import argparse
import json
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
from einops import rearrange
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDIMScheduler
from torch.utils.data import Dataset
from datetime import datetime

# 引入你的 Pipeline 與 Model
from genphoto.pipelines.pipeline_inversion import GenPhotoInversionPipeline
from genphoto.models.unet import UNet3DConditionModelCameraCond
from genphoto.models.camera_adaptor import CameraCameraEncoder
# [修正] 補上 save_videos_grid
from genphoto.utils.util import save_videos_grid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. Embedding 生成邏輯 (嚴格參照官方實作)
# ==========================================

def create_bokehK_embedding(bokehK_values, target_height, target_width):
    # bokehK_values: tensor [N] or [N, 1]
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
        if val > 100:
            kelvin = val
        else:
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
# 2. Universal Dataset Class
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
# 3. Load Models
# ==========================================
def load_models_inversion(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(cfg.noise_scheduler_kwargs))
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    
    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_path, subfolder="text_encoder").to(device)
    text_encoder.requires_grad_(False)
    
    unet = UNet3DConditionModelCameraCond.from_pretrained_2d(
        cfg.pretrained_model_path,
        subfolder=cfg.unet_subfolder,
        unet_additional_kwargs=cfg.unet_additional_kwargs
    ).to(device)
    unet.requires_grad_(False)

    camera_encoder = CameraCameraEncoder(**cfg.camera_encoder_kwargs).to(device)
    camera_encoder.requires_grad_(False)
    
    pipeline = GenPhotoInversionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        camera_encoder=camera_encoder
    ).to(device)
    pipeline.enable_vae_slicing()

    logger.info("Setting the attention processors...")
    unet.set_all_attn_processor(
        add_spatial_lora=cfg.lora_ckpt is not None,
        add_motion_lora=cfg.motion_lora_rank > 0, 
        lora_kwargs={"lora_rank": cfg.lora_rank, "lora_scale": cfg.lora_scale},
        motion_lora_kwargs={"lora_rank": cfg.motion_lora_rank, "lora_scale": cfg.motion_lora_scale},
        **cfg.attention_processor_kwargs
    )

    if cfg.lora_ckpt is not None:
        logger.info(f"Loading Spatial LoRA from {cfg.lora_ckpt}")
        lora_state = torch.load(cfg.lora_ckpt, map_location=device)
        if 'lora_state_dict' in lora_state: lora_state = lora_state['lora_state_dict']
        unet.load_state_dict(lora_state, strict=False)

    if cfg.motion_module_ckpt is not None:
        logger.info(f"Loading Motion Module from {cfg.motion_module_ckpt}")
        mm_state = torch.load(cfg.motion_module_ckpt, map_location=device)
        unet.load_state_dict(mm_state, strict=False)

    if cfg.camera_adaptor_ckpt is not None:
        logger.info(f"Loading Camera Adaptor from {cfg.camera_adaptor_ckpt}")
        ca_state = torch.load(cfg.camera_adaptor_ckpt, map_location=device)
        camera_encoder.load_state_dict(ca_state['camera_encoder_state_dict'], strict=False)
        unet.load_state_dict(ca_state['attention_processor_state_dict'], strict=False)
    else:
        logger.warning("!!! No Camera Adaptor Checkpoint defined in YAML. Inference will likely fail or do nothing !!!")

    pipeline.to(device)
    return pipeline, device

# ==========================================
# 4. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config. If not provided, it will be auto-selected based on setting_type.")
    parser.add_argument("--setting_type", type=str, required=True, choices=['bokeh', 'focal', 'shutter', 'color'], help="Type of camera parameter")
    parser.add_argument("--base_scene", type=str, required=True, help="Prompt text")
    parser.add_argument("--param_list", type=str, required=True, help="JSON list of values, e.g., '[1.0, 5.0]'")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="outputs/ddim_multi_camera", help="Output directory")
    args = parser.parse_args()

    # 自動選擇 Config 邏輯
    default_config_map = {
        'bokeh': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_bokehK.yaml',
        'focal': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_focal_length.yaml',
        'shutter': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_shutter_speed.yaml',
        'color': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_color_temperature.yaml'
    }

    if args.config is None:
        if args.setting_type in default_config_map:
            args.config = default_config_map[args.setting_type]
            logger.info(f"✨ 偵測到未輸入 Config，已自動選用: {args.config}")
        else:
            raise ValueError(f"無法自動匹配 {args.setting_type} 的 Config，請手動輸入 --config")
    else:
        config_name = args.config.lower()
        type_map_check = {
            'bokeh': 'bokeh',
            'focal': 'focal_length',
            'shutter': 'shutter_speed',
            'color': 'color_temperature'
        }
        expected_keyword = type_map_check[args.setting_type]
        if expected_keyword not in config_name:
            logger.warning("================================================================")
            logger.warning(f"⚠️  警告: 設定檔名稱 '{args.config}' 似乎與參數類型 '{args.setting_type}' 不符！")
            logger.warning(f"預期設定檔應包含關鍵字: '{expected_keyword}'")
            logger.warning("================================================================")

    # 1. 初始化
    cfg = OmegaConf.load(args.config)
    pipeline, device = load_models_inversion(cfg)
    
    # 影像前處理
    raw_image = Image.open(args.input_image).convert("RGB").resize((384, 256))
    image_tensor = transforms.ToTensor()(raw_image).unsqueeze(0).to(device)
    image_tensor = image_tensor * 2.0 - 1.0

    # 2. 準備參數
    target_vals_list = json.loads(args.param_list)
    video_len = len(target_vals_list)
    
    initial_val = target_vals_list[0]
    source_vals = torch.tensor([initial_val] * video_len, dtype=torch.float32)
    target_vals = torch.tensor(target_vals_list, dtype=torch.float32)

    # 建立 Embedding
    source_embed = Universal_Camera_Embedding(args.setting_type, source_vals, pipeline.tokenizer, pipeline.text_encoder, device).load()
    target_embed = Universal_Camera_Embedding(args.setting_type, target_vals, pipeline.tokenizer, pipeline.text_encoder, device).load()
    
    source_embed = rearrange(source_embed.unsqueeze(0), "b f c h w -> b c f h w")
    target_embed = rearrange(target_embed.unsqueeze(0), "b f c h w -> b c f h w")

    # 3. Inversion
    logger.info(f"Running Inversion with STATIC {args.setting_type} (Val: {initial_val})...")
    inverted_latents = pipeline.invert(
        image=image_tensor,
        prompt=args.base_scene,
        camera_embedding=source_embed,
        num_inference_steps=25,
        video_length=video_len
    )

    # 4. Generation
    logger.info(f"Running Generation with DYNAMIC {args.setting_type}...")
    with torch.no_grad():
        output = pipeline(
            prompt=args.base_scene,
            camera_embedding=target_embed,
            video_length=video_len,
            height=256,
            width=384,
            num_inference_steps=25,
            guidance_scale=1.5,
            latents=inverted_latents 
        ).videos[0]

    # 5. 存檔
    timestamp = datetime.now().strftime("%H%M%S")
    save_dir = os.path.join(args.output_dir, args.setting_type)
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"{timestamp}_val{initial_val}_to_{target_vals_list[-1]}.gif")
    save_videos_grid(output[None, ...], save_path)
    logger.info(f"Saved result to {save_path}")

if __name__ == "__main__":
    main()

'''
python inference_multi_camera.py \
  --setting_type bokeh \
  --base_scene "A photo of a park with green grass and trees" \
  --param_list "[2.44, 8.3, 10.1, 17.2, 24.0]" \
  --input_image ./input_image/my_park_photo.jpg

python inference_multi_camera.py \
  --setting_type focal \
  --base_scene "A photo of a park with green grass and trees" \
  --param_list "[25.0, 35.0, 45.0, 55.0, 65.0]" \
  --input_image ./input_image/my_park_photo.jpg

python inference_multi_camera.py \
  --setting_type shutter \
  --base_scene "A photo of a park with green grass and trees" \
  --param_list "[0.1, 0.3, 0.52, 0.7, 0.8]" \
  --input_image ./input_image/my_park_photo.jpg

python inference_multi_camera.py \
  --setting_type color \
  --base_scene "A photo of a park with green grass and trees" \
  --param_list "[5455.0, 5155.0, 5555.0, 6555.0, 7555.0]" \
  --input_image ./input_image/my_park_photo.jpg
'''