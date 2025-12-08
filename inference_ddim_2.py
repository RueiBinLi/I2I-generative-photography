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

# 引入你的模組
from genphoto.pipelines.pipeline_inversion import GenPhotoInversionPipeline
from genphoto.models.unet import UNet3DConditionModelCameraCond
from genphoto.models.camera_adaptor import CameraCameraEncoder, CameraAdaptor
from genphoto.utils.util import save_videos_grid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 複製自 inference_bokehK.py 的輔助類別與函式
# 這樣就不用依賴互相 import 了
# ==========================================

def create_bokehK_embedding(bokehK_values, target_height, target_width):
    f = bokehK_values.shape[0]
    bokehK_embedding = torch.zeros((f, 3, target_height, target_width), dtype=bokehK_values.dtype)
    
    for i in range(f):
        K_value = bokehK_values[i].item()
        kernel_size = max(K_value, 1)
        sigma = K_value / 3.0

        ax = np.linspace(-(kernel_size / 2), kernel_size / 2, int(np.ceil(kernel_size)))
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= np.sum(kernel)
        scale = kernel[int(np.ceil(kernel_size) / 2), int(np.ceil(kernel_size) / 2)]
        
        bokehK_embedding[i] = scale
    
    return bokehK_embedding

class Camera_Embedding(Dataset):
    def __init__(self, bokehK_values, tokenizer, text_encoder, device, sample_size=[256, 384]):
        self.bokehK_values = bokehK_values.to(device)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device  
        self.sample_size = sample_size

    def load(self):
        # 這裡不硬性限制長度為 5，讓外部控制
        
        prompts = []
        for bb in self.bokehK_values:
            prompt = f"<bokeh kernel size: {bb.item()}>"
            prompts.append(prompt)
        
        with torch.no_grad():
            prompt_ids = self.tokenizer(
                prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)

            encoder_hidden_states = self.text_encoder(input_ids=prompt_ids).last_hidden_state

        differences = []
        for i in range(1, encoder_hidden_states.size(0)):
            diff = encoder_hidden_states[i] - encoder_hidden_states[i - 1]
            diff = diff.unsqueeze(0)
            differences.append(diff)  

        final_diff = encoder_hidden_states[-1] - encoder_hidden_states[0]
        final_diff = final_diff.unsqueeze(0)
        differences.append(final_diff)

        concatenated_differences = torch.cat(differences, dim=0)
        
        # Padding logic
        pad_length = 128 - concatenated_differences.size(1)
        if pad_length > 0:
            concatenated_differences = F.pad(concatenated_differences, (0, 0, 0, pad_length))

        frame = concatenated_differences.size(0)
        ccl_embedding = concatenated_differences.reshape(frame, self.sample_size[0], self.sample_size[1])
        ccl_embedding = ccl_embedding.unsqueeze(1)  
        ccl_embedding = ccl_embedding.expand(-1, 3, -1, -1)
        ccl_embedding = ccl_embedding.to(self.device)
        
        bokehK_embedding_tensor = create_bokehK_embedding(self.bokehK_values, self.sample_size[0], self.sample_size[1]).to(self.device)
        camera_embedding = torch.cat((bokehK_embedding_tensor, ccl_embedding), dim=1)
        return camera_embedding

# ==========================================
# 主要邏輯
# ==========================================

def load_models_inversion(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(cfg.noise_scheduler_kwargs))
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    
    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_path, subfolder="text_encoder").to(device)
    text_encoder.requires_grad_(False)
    
    # 載入支援 Camera Condition 的 UNet
    unet = UNet3DConditionModelCameraCond.from_pretrained_2d(
        cfg.pretrained_model_path,
        subfolder=cfg.unet_subfolder,
        unet_additional_kwargs=cfg.unet_additional_kwargs
    ).to(device)
    unet.requires_grad_(False)

    camera_encoder = CameraCameraEncoder(**cfg.camera_encoder_kwargs).to(device)
    camera_encoder.requires_grad_(False)
    
    # 建立 Pipeline (使用我們修改過的 Inversion Pipeline)
    pipeline = GenPhotoInversionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        camera_encoder=camera_encoder
    ).to(device)
    
    pipeline.enable_vae_slicing()

    # 設定 Attention Processors (包含 Motion LoRA 設定)
    logger.info("Setting the attention processors...")

    unet.set_all_attn_processor(
        add_spatial_lora=cfg.lora_ckpt is not None,
        add_motion_lora=cfg.motion_lora_rank > 0, # [重要] 強制開啟 Motion LoRA
        lora_kwargs={"lora_rank": cfg.lora_rank, "lora_scale": cfg.lora_scale},
        motion_lora_kwargs={"lora_rank": cfg.motion_lora_rank, "lora_scale": cfg.motion_lora_scale},
        **cfg.attention_processor_kwargs
    )

    # 載入權重 (Spatial LoRA, Motion LoRA, Camera Adaptor)
    if cfg.lora_ckpt is not None:
        logger.info(f"Loading LoRA from {cfg.lora_ckpt}")
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

    pipeline.to(device)
    return pipeline, device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--base_scene", type=str, required=True, help="Prompt text")
    parser.add_argument("--bokehK_list", type=str, required=True, help="Target bokeh values list (JSON)")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="outputs/ddim_inversion", help="Output directory")
    args = parser.parse_args()

    # 1. 初始化
    cfg = OmegaConf.load(args.config)
    pipeline, device = load_models_inversion(cfg)
    
    # 讀取與處理輸入圖片
    raw_image = Image.open(args.input_image).convert("RGB").resize((384, 256)) # 注意這裡的 resize 要配合 config
    # 轉換圖片為 Tensor [-1, 1]
    image_tensor = transforms.ToTensor()(raw_image).unsqueeze(0).to(device)
    image_tensor = image_tensor * 2.0 - 1.0

    # 2. 準備兩組 Embedding (Source vs Target)
    target_vals = json.loads(args.bokehK_list)
    video_len = len(target_vals)
    
    # A. Source Embedding (靜態/初始狀態)
    # 使用列表的第一個值，複製 N 次，代表「這張圖是相機參數為 X 的靜止狀態」
    initial_val = target_vals[0]
    source_vals = [initial_val] * video_len
    
    source_bokeh_tensor = torch.tensor(source_vals).unsqueeze(1)
    source_cam_embed_obj = Camera_Embedding(source_bokeh_tensor, pipeline.tokenizer, pipeline.text_encoder, device)
    source_camera_embedding = source_cam_embed_obj.load()
    # 調整維度 [b, c, f, h, w]
    source_camera_embedding = rearrange(source_camera_embedding.unsqueeze(0), "b f c h w -> b c f h w")

    # B. Target Embedding (動態/目標狀態)
    # 使用使用者輸入的漸變列表
    target_bokeh_tensor = torch.tensor(target_vals).unsqueeze(1)
    target_cam_embed_obj = Camera_Embedding(target_bokeh_tensor, pipeline.tokenizer, pipeline.text_encoder, device)
    target_camera_embedding = target_cam_embed_obj.load()
    target_camera_embedding = rearrange(target_camera_embedding.unsqueeze(0), "b f c h w -> b c f h w")

    # 3. 執行 Inversion (使用 Source Embedding)
    logger.info("Running DDIM Inversion with STATIC camera parameters...")
    inverted_latents = pipeline.invert(
        image=image_tensor,
        prompt=args.base_scene,
        camera_embedding=source_camera_embedding, # 關鍵：告訴模型這是靜態的
        num_inference_steps=25,
        video_length=video_len
    )

    # 4. 執行 Generation (使用 Target Embedding)
    logger.info("Running Generation with DYNAMIC camera parameters...")
    with torch.no_grad():
        output = pipeline(
            prompt=args.base_scene,
            camera_embedding=target_camera_embedding, # 關鍵：現在我要動起來
            video_length=video_len,
            height=256,
            width=384,
            num_inference_steps=25,
            guidance_scale=2.0,
            latents=inverted_latents # 傳入反推的 latents
        ).videos[0]

    # 5. 存檔
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "ddim_result.gif")
    save_videos_grid(output[None, ...], save_path)
    logger.info(f"Saved result to {save_path}")

if __name__ == "__main__":
    main()
'''
python inference_ddim_2.py \
  --config configs/inference_genphoto/adv3_256_384_genphoto_relora_bokehK.yaml \
  --base_scene "A photo of a park with green grass and trees" \
  --bokehK_list "[2.44, 8.3, 10.1, 17.2, 24.0]" \
  --input_image ./input_image/my_park_photo.jpg \
  --output_dir outputs/ddim_test_2
'''