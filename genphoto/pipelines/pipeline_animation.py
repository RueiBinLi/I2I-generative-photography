# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
import torch

import numpy as np

from typing import Callable, List, Optional, Union
from dataclasses import dataclass
from diffusers.utils import is_accelerate_available
from packaging import version
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import deprecate, logging, BaseOutput

from genphoto.models.camera_adaptor import CameraCameraEncoder
from genphoto.models.unet import UNet3DConditionModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class AnimationPipeline(DiffusionPipeline, LoraLoaderMixin):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in range(latents.shape[0]):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    '''
    SDEdit 新增的 Function
    1. get_timesteps: 根據 strength 計算初始 timestep
    2. prepare_latents_from_image: 從圖片產生初始 latents
    3. 在 __call__ 裡面加入 SDEdit 的邏輯
    '''
    def get_timesteps(self, num_inference_steps, strength, device):
        # 根據 strength 計算初始 timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def prepare_latents_from_image(self, image, batch_size, num_videos_per_prompt, video_length, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, list)):
            raise ValueError(f"`image` has to be of type `torch.Tensor` or list but is {type(image)}")

        # 1. 如果輸入是 Tensor (B, C, H, W)，移動到 device 並轉換 dtype
        image = image.to(device=device, dtype=dtype)

        # Support 5D Tensor (Video Input)
        if image.ndim == 5: # (B, C, F, H, W)
            b, c, f, h, w = image.shape
            # Flatten to (B*F, C, H, W) for VAE
            image_flat = rearrange(image, "b c f h w -> (b f) c h w")
            
            # VAE Encode
            if isinstance(generator, list):
                # Assuming generator list matches batch_size (b)
                # We need to repeat generator for each frame if we want to be precise, 
                # but usually generator is for the noise, here we are encoding.
                # VAE encode is deterministic usually unless we sample. 
                # latent_dist.sample uses generator.
                # Let's just use the first generator or handle it simply.
                # For now, let's assume generator is a single object or handle b*f expansion if needed.
                # To be safe, let's just loop if it's a list.
                init_latents_list = []
                for i in range(b):
                    # Get generator for this batch item
                    gen = generator[i] if i < len(generator) else None
                    # Encode frames for this batch item
                    # image_flat[i*f : (i+1)*f]
                    sub_batch = image_flat[i*f : (i+1)*f]
                    encoded = self.vae.encode(sub_batch).latent_dist.sample(gen)
                    init_latents_list.append(encoded)
                init_latents = torch.cat(init_latents_list, dim=0)
            else:
                init_latents = self.vae.encode(image_flat).latent_dist.sample(generator)

            init_latents = self.vae.config.scaling_factor * init_latents
            
            # Reshape back to (B, C, F, H, W)
            # init_latents is now (B*F, C, H', W')
            init_latents = rearrange(init_latents, "(b f) c h w -> b c f h w", b=b, f=f)
            
            # If video_length > f, we might need to repeat or pad?
            # For now, assume f == video_length or we just use what we have.
            # If f < video_length, we repeat the last frame?
            if f < video_length:
                diff = video_length - f
                last_frame = init_latents[:, :, -1:, :, :]
                padding = last_frame.repeat(1, 1, diff, 1, 1)
                init_latents = torch.cat([init_latents, padding], dim=2)
            elif f > video_length:
                init_latents = init_latents[:, :, :video_length, :, :]
                
            return init_latents

        # 2. VAE Encode (Original 4D Logic)
        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)

        init_latents = self.vae.config.scaling_factor * init_latents

        # 3. 處理維度: (B, C, H, W) -> (B, C, F, H, W)
        # 我們要把單張圖片複製，填滿整個 video_length，讓 Temporal Attention 有東西可以參考
        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        # 這裡將 (Batch, Channel, Height, Width) -> (Batch, Channel, Video_Length, Height, Width)
        init_latents = init_latents.unsqueeze(2).repeat(1, 1, video_length, 1, 1)

        return init_latents
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,

        multidiff_total_steps: int = 1,
        multidiff_overlaps: int = 12,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        single_model_length = video_length
        video_length = multidiff_total_steps * (video_length - multidiff_overlaps) + multidiff_overlaps
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_pred_full = torch.zeros_like(latents).to(latents.device)
                mask_full = torch.zeros_like(latents).to(latents.device)
                noise_preds = []

                for multidiff_step in range(multidiff_total_steps):
                    start_idx = multidiff_step * (single_model_length - multidiff_overlaps)
                    latent_partial = latents[:, :, start_idx: start_idx + single_model_length].contiguous()
                    mask_full[:, :, start_idx: start_idx + single_model_length] += 1

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latent_partial] * 2) if do_classifier_free_guidance else latent_partial
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample.to(dtype=latents_dtype)

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_preds.append(noise_pred)

                for pred_idx, noise_pred in enumerate(noise_preds):
                    start_idx = pred_idx * (single_model_length - multidiff_overlaps)
                    noise_pred_full[:, :, start_idx: start_idx + single_model_length] += noise_pred / mask_full[:, :, start_idx: start_idx + single_model_length]

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred_full, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Post-processing
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)
    


class GenPhotoPipeline(AnimationPipeline):
    _optional_components = []

    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet3DConditionModel,
                 scheduler: Union[
                     DDIMScheduler,
                     PNDMScheduler,
                     LMSDiscreteScheduler,
                     EulerDiscreteScheduler,
                     EulerAncestralDiscreteScheduler,
                     DPMSolverMultistepScheduler],
                 camera_encoder: CameraCameraEncoder):

        super().__init__(vae, text_encoder, tokenizer, unet, scheduler)

        self.register_modules(
            camera_encoder=camera_encoder
        )

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in range(latents.shape[0]):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    # 在 GenPhotoPipeline 類別中
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        camera_embedding: torch.FloatTensor,
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        multidiff_total_steps: int = 1,
        multidiff_overlaps: int = 12,
        # --- 新增參數 ---
        image: Optional[torch.FloatTensor] = None,
        strength: float = 0.8,
        **kwargs,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(prompt, height, width, callback_steps)

        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = camera_embedding[0].device if isinstance(camera_embedding, list) else camera_embedding.device
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # --- I2V (SDEdit) Logic ---
        if image is not None:
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
            # [DEBUG]
            print(f"\n[DEBUG] Strength: {strength}")
            print(f"[DEBUG] New num_inference_steps: {num_inference_steps}")
            if len(timesteps) > 0:
                print(f"[DEBUG] Start Timestep (target_t): {timesteps[0]}")
            else:
                print(f"[DEBUG] WARNING: Timesteps is empty! Strength might be too low.")

        single_model_length = video_length
        video_length = multidiff_total_steps * (video_length - multidiff_overlaps) + multidiff_overlaps
        num_channels_latents = self.unet.in_channels
        
        # Prepare Latents
        if image is not None:
            pixel_latents = self.prepare_latents_from_image(
                image, batch_size * num_videos_per_prompt, num_videos_per_prompt, video_length, text_embeddings.dtype, device, generator
            )
            # [DEBUG] Check pixel_latents statistics
            print(f"[DEBUG] Pixel Latents (Image) -> Mean: {pixel_latents.mean():.4f}, Std: {pixel_latents.std():.4f}")
            
            noise = torch.randn(pixel_latents.shape, generator=generator, device=device, dtype=text_embeddings.dtype)
            
            # [DEBUG] Check dimensions
            bs, c, f, h, w = pixel_latents.shape
            print(f"[DEBUG] Latents Shape: {pixel_latents.shape}")

            target_t = timesteps[0]
            
            # Expand target_t to match batch size for safety (Diffusers bug workaround)
            # Make sure target_t is a tensor of shape (B*F,)
            if isinstance(target_t, torch.Tensor) and target_t.ndim == 0:
                    target_t = target_t.repeat(bs * f)
            elif isinstance(target_t, int) or isinstance(target_t, float):
                    target_t = torch.tensor([target_t] * (bs * f), device=device)
            
            # Flatten for add_noise
            pixel_latents_flat = rearrange(pixel_latents, "b c f h w -> (b f) c h w")
            noise_flat = rearrange(noise, "b c f h w -> (b f) c h w")
            
            # Add Noise
            latents_flat = self.scheduler.add_noise(pixel_latents_flat, noise_flat, target_t)
            
            # [DEBUG] Check Noisy Latents statistics
            print(f"[DEBUG] Noisy Latents -> Mean: {latents_flat.mean():.4f}, Std: {latents_flat.std():.4f}")
            
            # Reshape back
            latents = rearrange(latents_flat, "(b f) c h w -> b c f h w", b=bs, f=f)
            
        else:
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                video_length,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents,
            )
        
        latents_dtype = latents.dtype
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # Prepare Camera Embeddings
        if isinstance(camera_embedding, list):
            assert all([x.ndim == 5 for x in camera_embedding])
            bs = camera_embedding[0].shape[0]
            camera_embedding_features = []
            for pe in camera_embedding:
                camera_embedding_feature = self.camera_encoder(pe)
                camera_embedding_feature = [rearrange(x, '(b f) c h w -> b c f h w', b=bs) for x in camera_embedding_feature]
                camera_embedding_features.append(camera_embedding_feature)
        else:
            bs = camera_embedding.shape[0]
            assert camera_embedding.ndim == 5
            camera_embedding_features = self.camera_encoder(camera_embedding)
            camera_embedding_features = [rearrange(x, '(b f) c h w -> b c f h w', b=bs)
                                        for x in camera_embedding_features]

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        if isinstance(camera_embedding_features[0], list):
            camera_embedding_features = [[torch.cat([x, x], dim=0) for x in camera_embedding_feature]
                                        for camera_embedding_feature in camera_embedding_features] \
                if do_classifier_free_guidance else camera_embedding_features
        else:
            camera_embedding_features = [torch.cat([x, x], dim=0) for x in camera_embedding_features] \
                if do_classifier_free_guidance else camera_embedding_features

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_pred_full = torch.zeros_like(latents).to(latents.device)
                mask_full = torch.zeros_like(latents).to(latents.device)
                noise_preds = []
                for multidiff_step in range(multidiff_total_steps):
                    start_idx = multidiff_step * (single_model_length - multidiff_overlaps)
                    latent_partial = latents[:, :, start_idx: start_idx + single_model_length].contiguous()
                    mask_full[:, :, start_idx: start_idx + single_model_length] += 1

                    if isinstance(camera_embedding, list):
                        camera_embedding_features_input = camera_embedding_features[multidiff_step]
                    else:
                        camera_embedding_features_input = [x[:, :, start_idx: start_idx + single_model_length]
                                                            for x in camera_embedding_features]

                    latent_model_input = torch.cat([latent_partial] * 2) if do_classifier_free_guidance else latent_partial
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings,
                                            camera_embedding_features=camera_embedding_features_input).sample.to(dtype=latents_dtype)
                    
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_preds.append(noise_pred)
                for pred_idx, noise_pred in enumerate(noise_preds):
                    start_idx = pred_idx * (single_model_length - multidiff_overlaps)
                    noise_pred_full[:, :, start_idx: start_idx + single_model_length] += noise_pred / mask_full[:, :, start_idx: start_idx + single_model_length]

                latents = self.scheduler.step(noise_pred_full, t, latents, **extra_step_kwargs).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        video = self.decode_latents(latents)

        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)
