import os
import torch
import logging
import argparse
import json
import numpy as np
import pandas as pd
import lpips
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
from einops import rearrange
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDIMScheduler
from datetime import datetime
from torch.utils.data import Dataset

# 引入你的 Pipeline
from genphoto.pipelines.pipeline_inversion import GenPhotoInversionPipeline
from genphoto.models.unet import UNet3DConditionModelCameraCond
from genphoto.models.camera_adaptor import CameraCameraEncoder
from genphoto.utils.util import save_videos_grid

# 嘗試引入 torchmetrics 用於 PSNR/SSIM，若無則使用簡易版計算
try:
    from torchmetrics.functional import peak_signal_noise_ratio as psnr_func
    from torchmetrics.functional import structural_similarity_index_measure as ssim_func
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False
    print("Warning: torchmetrics not found. Using simple PSNR/SSIM calculation.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. Embedding Logic (複製自各個 inference 腳本)
# ==========================================
def kelvin_to_rgb(kelvin):
    if torch.is_tensor(kelvin): kelvin = kelvin.cpu().item()
    temp = kelvin / 100.0
    if temp <= 66:
        red = 255
        green = 99.4708025861 * np.log(temp) - 161.1195681661 if temp > 0 else 0
        blue = 0 if temp <= 19 else 138.5177312231 * np.log(temp - 10) - 305.0447927307
    elif temp <= 88:
        red = 0.5 * (255 + 329.698727446 * ((temp - 60) ** -0.19332047592))
        green = 0.5 * (288.1221695283 * ((temp - 60) ** -0.1155148492) + (99.4708025861 * np.log(temp) - 161.1195681661 if temp > 0 else 0))
        blue = 0.5 * (138.5177312231 * np.log(temp - 10) - 305.0447927307 + 255)
    else:
        red = 329.698727446 * ((temp - 60) ** -0.19332047592)
        green = 288.1221695283 * ((temp - 60) ** -0.1155148492)
        blue = 255
    return np.array([red, green, blue], dtype=np.float32) / 255.0

class Validation_Embedding(Dataset):
    def __init__(self, setting_type, val_list, tokenizer, text_encoder, device, sample_size=[256, 384]):
        self.setting_type = setting_type
        self.val_list = val_list
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.sample_size = sample_size

    def create_visual_embedding(self):
        f = len(self.val_list)
        h, w = self.sample_size
        embedding = torch.zeros((f, 3, h, w), device='cpu') # Build on CPU first

        if self.setting_type == 'none':
            pass # Return zeros

        elif self.setting_type == 'bokeh':
            for i, val in enumerate(self.val_list):
                K = val
                kernel_size = max(K, 1)
                sigma = K / 3.0
                ax = np.linspace(-(kernel_size/2), kernel_size/2, int(np.ceil(kernel_size)))
                xx, yy = np.meshgrid(ax, ax)
                kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
                kernel /= np.sum(kernel)
                scale = kernel[int(np.ceil(kernel_size)/2), int(np.ceil(kernel_size)/2)]
                embedding[i] = scale

        elif self.setting_type == 'focal':
            sensor_w, sensor_h, base_fl = 36.0, 24.0, 24.0
            base_fov_x = 2 * np.arctan(sensor_w * 0.5 / base_fl)
            base_fov_y = 2 * np.arctan(sensor_h * 0.5 / base_fl)
            center_h, center_w = h // 2, w // 2
            
            for i, fl in enumerate(self.val_list):
                target_fov_x = 2 * np.arctan(sensor_w * 0.5 / fl)
                target_fov_y = 2 * np.arctan(sensor_h * 0.5 / fl)
                crop_w = int(max(1, min(w, np.round((target_fov_x / base_fov_x) * w))))
                crop_h = int(max(1, min(h, np.round((target_fov_y / base_fov_y) * h))))
                embedding[i, :, center_h - crop_h//2 : center_h + crop_h//2, 
                                center_w - crop_w//2 : center_w + crop_w//2] = 1.0

        elif self.setting_type == 'shutter':
            base_exp, fwc = 0.5, 32000
            for i, ss in enumerate(self.val_list):
                scale = (ss / base_exp) * (fwc / (fwc + 0.0001))
                embedding[i] = scale

        elif self.setting_type == 'color':
            for i, ct in enumerate(self.val_list):
                # Check if input is raw Kelvin (>100) or normalized
                kelvin = ct if ct > 100 else 2000 + (ct * (10000 - 2000))
                rgb = kelvin_to_rgb(kelvin)
                embedding[i] = torch.tensor(rgb).view(3, 1, 1)

        return embedding.to(self.device)

    def load(self):
        # 1. Text Embedding
        prompts = []
        for v in self.val_list:
            if self.setting_type == 'none': prompt = ""
            elif self.setting_type == 'bokeh': prompt = f"<bokeh kernel size: {v}>"
            elif self.setting_type == 'focal': prompt = f"<focal length: {v}>"
            elif self.setting_type == 'shutter': prompt = f"<exposure: {v}>"
            elif self.setting_type == 'color': prompt = f"<color temperature: {v}>"
            else: prompt = ""
            prompts.append(prompt)

        with torch.no_grad():
            prompt_ids = self.tokenizer(prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(self.device)
            encoder_hidden_states = self.text_encoder(input_ids=prompt_ids).last_hidden_state

        differences = []
        for i in range(1, encoder_hidden_states.size(0)):
            differences.append((encoder_hidden_states[i] - encoder_hidden_states[i-1]).unsqueeze(0))
        # Handle single frame case or end of sequence
        if encoder_hidden_states.size(0) > 1:
            differences.append((encoder_hidden_states[-1] - encoder_hidden_states[0]).unsqueeze(0))
            text_diff = torch.cat(differences, dim=0)
        else:
            text_diff = torch.zeros_like(encoder_hidden_states)

        pad_len = 128 - text_diff.size(1)
        if pad_len > 0: text_diff = F.pad(text_diff, (0, 0, 0, pad_len))
        
        ccl_embedding = text_diff.reshape(text_diff.size(0), self.sample_size[0], self.sample_size[1])
        ccl_embedding = ccl_embedding.unsqueeze(1).expand(-1, 3, -1, -1).to(self.device)

        # 2. Visual Embedding
        vis_emb = self.create_visual_embedding()
        
        return torch.cat((vis_emb, ccl_embedding), dim=1)

# ==========================================
# 2. Main Validation Logic
# ==========================================
def calculate_metrics(pred, target, lpips_fn):
    # Inputs: [1, 3, H, W], Range [0, 1]
    # LPIPS expects [-1, 1]
    pred_norm = pred * 2.0 - 1.0
    target_norm = target * 2.0 - 1.0
    
    score_lpips = lpips_fn(pred_norm, target_norm).item()
    
    if HAS_TORCHMETRICS:
        score_psnr = psnr_func(pred, target, data_range=1.0).item()
        score_ssim = ssim_func(pred, target, data_range=1.0).item()
    else:
        # Simple Fallback
        mse = torch.mean((pred - target) ** 2)
        score_psnr = -10.0 * torch.log10(mse).item() if mse > 0 else 100.0
        score_ssim = 0.0 # Skipping complex manual implementation
        
    return score_lpips, score_psnr, score_ssim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_scene", type=str, required=True)
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--setting_type", type=str, default="none", choices=['none', 'bokeh', 'focal', 'shutter', 'color'])
    parser.add_argument("--param_val", type=float, default=0.0, help="Static parameter value for reconstruction")
    parser.add_argument("--steps_list", type=int, nargs='+', default=[25, 50], help="List of steps to test")
    parser.add_argument("--output_dir", type=str, default="validation_results")
    args = parser.parse_args()

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = OmegaConf.load(args.config)
    
    # Initialize LPIPS (Author's method)
    lpips_loss = lpips.LPIPS(net='vgg').to(device)

    # Load Model (Copy from your working inference script)
    from inference_multi_camera import load_models_inversion
    pipeline, _ = load_models_inversion(cfg)
    
    # Prepare Input
    raw_image = Image.open(args.input_image).convert("RGB").resize((384, 256))
    gt_tensor = transforms.ToTensor()(raw_image).unsqueeze(0).to(device) # [1, 3, H, W]
    image_input = gt_tensor * 2.0 - 1.0

    # Prepare Embedding (Static)
    video_len = 5
    val_list = [args.param_val] * video_len
    embed_obj = Validation_Embedding(args.setting_type, val_list, pipeline.tokenizer, pipeline.text_encoder, device)
    camera_embedding = embed_obj.load()
    camera_embedding = rearrange(camera_embedding.unsqueeze(0), "b f c h w -> b c f h w")

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, f"{timestamp}_{args.setting_type}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"--- Starting Validation: {args.setting_type} (Val: {args.param_val}) ---")

    for step in args.steps_list:
        print(f"Testing Steps: {step}...")
        
        # 1. Inversion
        inverted_latents = pipeline.invert(
            image=image_input,
            prompt=args.base_scene,
            camera_embedding=camera_embedding,
            num_inference_steps=step,
            video_length=video_len
        )

        # 2. Reconstruction
        with torch.no_grad():
            output = pipeline(
                prompt=args.base_scene,
                camera_embedding=camera_embedding, # Same static embedding
                video_length=video_len,
                height=256,
                width=384,
                num_inference_steps=step,
                guidance_scale=1.0, 
                latents=inverted_latents
            ).videos # [1, 3, F, H, W]

        # 3. Metrics (Compare 1st frame with GT)
        rec_frame = output[0, :, 0, :, :].unsqueeze(0).to(device)
        l, p, s = calculate_metrics(rec_frame, gt_tensor, lpips_loss)
        
        print(f"Step {step} | LPIPS: {l:.4f} | PSNR: {p:.2f} | SSIM: {s:.4f}")
        results.append({"steps": step, "LPIPS": l, "PSNR": p, "SSIM": s})
        
        save_videos_grid(output, os.path.join(save_dir, f"rec_step_{step}.gif"))

    # Save Report
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)
    print(f"Saved report to {save_dir}/metrics.csv")

if __name__ == "__main__":
    main()
'''
python validate_reconstruction.py \
  --config configs/inference_genphoto/adv3_256_384_genphoto_relora_bokehK.yaml \
  --base_scene "A photo of a park with green grass and trees" \
  --input_image ./input_image/test.jpg \
  --setting_type none \
  --steps_list 25 50 100

python validate_reconstruction.py \
  --config configs/inference_genphoto/adv3_256_384_genphoto_relora_bokehK.yaml \
  --base_scene "A photo of a park with green grass and trees" \
  --input_image ./input_image/test.jpg \
  --setting_type bokeh \
  --param_val 5.0 \
  --steps_list 50
'''

