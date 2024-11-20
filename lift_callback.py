import pprint
from typing import List, Optional, Dict, Any, Union

import pyrallis
import torch
from PIL import Image
from tqdm import trange
import os
from pathlib import Path

from config import LiftConfig
from pipeline_attend_and_excite import AttendAndExcitePipeline
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class LiftCallback:

    def __init__(self, config: LiftConfig):
        self.config = config

    def get_timesteps(self, num_train_timesteps: int):
        n_trials = self.config.n_samples
        if self.config.t_schedule == "interleave":
            ts = torch.linspace(0, num_train_timesteps-1, n_trials).round().long().clamp(0, num_train_timesteps-1)
        elif self.config.t_schedule == "random":
            ts = torch.randint(0, num_train_timesteps, (n_trials,))
        return ts

    def get_noise(self, latent_shape):
        # check if noise is already generated
        noise_path = Path(str(self.config.noise_path).replace(".pt", f"_{str(latent_shape).replace(',', '_').replace(' ', '')}.pt"))
        if os.path.exists(noise_path):
            noise = torch.load(noise_path, weights_only=True)
        else:
            noise = torch.randn((2048, *latent_shape))
            torch.save(noise, noise_path)
        if self.config.same_noise:
            return noise[0, ...].expand(self.config.n_samples, (-1,) * len(latent_shape))
        else:
            return noise[:self.config.n_samples]

    @torch.inference_mode()
    def calculate_lift(self,
                       pipeline: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
                       prompts: List[str],
                       algebras: List[str],
                       cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Check if latents are available
        if self.latest_latents is None:
            raise ValueError("No latents available. Make sure the callback is called before calculate_lift.")

        latents = self.latest_latents.clone()
        self.latest_latents = None

        # Compile the unet, might take a while at the first run
        unet = pipeline.unet
        # unet = torch.compile(pipeline.unet, mode="max-autotune")

        device = latents.device
        scheduler = pipeline.scheduler
        ts = self.get_timesteps(len(scheduler)).to(device)
        n_prompts = len(prompts)
        # -> (n_trials,)
        B, *latent_shape = latents.shape
        noise = self.get_noise(latent_shape).to(device)
        # -> (n_trials, *Latent_shape)
        num_latent_dims = len(latents.shape) - 1

        log_lift_results = torch.zeros((B, n_prompts, len(ts)), device=device)

        image_idxs = torch.arange(B, device=device)[:, None, None].expand(-1, n_prompts, len(ts)).flatten().to(device)
        algebra_idxs = torch.arange(n_prompts, device=device)[None, :, None].expand(B, -1, len(ts)).flatten().to(device)
        trial_idxs = torch.arange(len(ts), device=device)[None, None, :].expand(B, n_prompts, -1).flatten().to(device)

        idx = 0
        for _ in trange(len(trial_idxs) // self.config.batch_size + int(len(trial_idxs) % self.config.batch_size != 0), leave=False):
            current_latents = latents[image_idxs[idx:idx + self.config.batch_size]]
            # -> (batch_size, *Latent_shape)
            current_prompts = [prompts[idx] for idx in algebra_idxs[idx:idx + self.config.batch_size]]
            # -> (batch_size, embed_dim)
            current_noise = noise[trial_idxs[idx:idx + self.config.batch_size]]
            # -> (batch_size, *Latent_shape)
            current_ts = ts[trial_idxs[idx:idx + self.config.batch_size]]
            # -> (batch_size,)

            # Prepare the noisy sample
            sqrt_alpha_prod = scheduler.alphas_cumprod.to(device)[current_ts] ** 0.5
            sqrt_alpha_prod = sqrt_alpha_prod.flatten()
            # -> (batch_size,)
            sqrt_one_minus_alpha_prod = (1 - scheduler.alphas_cumprod.to(device)[current_ts]) ** 0.5
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
            # -> (batch_size,)
            noisy_sample = sqrt_alpha_prod.view(-1, *([1] * num_latent_dims)) * current_latents + \
                           sqrt_one_minus_alpha_prod.view(-1, *([1] * num_latent_dims)) * current_noise
            # -> (batch_size, *Latent_shape)

            # Double the batch size to add unconditional prediction
            noisy_sample = torch.cat([current_latents] * 2)
            t = torch.cat([current_ts] * 2)
            latent_model_input = scheduler.scale_model_input(noisy_sample, t)

            # Encode input prompt
            if isinstance(pipeline, StableDiffusionPipeline):
                prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
                    current_prompts,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                )
                current_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                # predict the noise residual
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=current_prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

            elif isinstance(pipeline, StableDiffusionXLPipeline):
                # Encode input prompt
                (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
                 negative_pooled_prompt_embeds) = pipeline.encode_prompt(
                    current_prompts,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                )
                # For classifier free guidance, concatenate the embeddings
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])

                # Get timestep conditioning
                timestep_cond = None
                if pipeline.unet.config.time_cond_proj_dim is not None:
                    guidance_scale_tensor = torch.tensor(pipeline.guidance_scale - 1).repeat(len(current_ts))
                    timestep_cond = pipeline.get_guidance_scale_embedding(
                        guidance_scale_tensor,
                        embedding_dim=pipeline.unet.config.time_cond_proj_dim
                    ).to(device=device, dtype=latent_model_input.dtype)

                # Get added time IDs conditioning
                add_text_embeds = pooled_prompt_embeds
                text_encoder_projection_dim = (
                    pipeline.text_encoder_2.config.projection_dim
                    if pipeline.text_encoder_2 is not None
                    else int(pooled_prompt_embeds.shape[-1])
                )
                add_time_ids = pipeline._get_add_time_ids(
                    original_size=(1024, 1024),
                    crops_coords_top_left=(0, 0),
                    target_size=(1024, 1024),
                    dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=text_encoder_projection_dim
                ).to(device)
                add_time_ids = add_time_ids.repeat(len(current_ts) * 2, 1)

                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                # predict the noise residual
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            current_log_lift = (noise_pred_uncond - current_noise).pow(2) - \
                               (noise_pred_text - current_noise).pow(2)
            # -> (batch_size,)
            log_lift_results[image_idxs[idx:idx + self.config.batch_size],
                             algebra_idxs[idx:idx + self.config.batch_size],
                             trial_idxs[idx:idx + self.config.batch_size]] = current_log_lift.view(len(current_ts), -1).mean(dim=-1)

            idx += len(current_ts)

        is_valid = torch.ones((B), device=device, dtype=torch.bool)
        for algebra_idx, algebra in enumerate(algebras):
            if algebra == "product":
                is_valid = is_valid & (log_lift_results[:, algebra_idx].mean(dim=1) > 0)
            elif algebra == "summation":
                is_valid = is_valid | (log_lift_results[:, algebra_idx].mean(dim=1) > 0)
            elif algebra == "negation":
                is_valid = is_valid & (log_lift_results[:, algebra_idx].mean(dim=1) <= 0)
            else:
                raise ValueError(f"Invalid algebra: {algebra}")

        return log_lift_results, latents, is_valid

    def __call__(self, i, t, latents) -> Any:
        self.latest_latents = latents
