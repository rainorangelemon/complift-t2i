import pprint
from typing import List, Optional, Dict, Any

import pyrallis
import torch
from PIL import Image
from tqdm import trange
import os

from config import LiftConfig
from pipeline_attend_and_excite import AttendAndExcitePipeline
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
        noise_path = self.config.noise_path
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
                       pipeline: AttendAndExcitePipeline,
                       prompts: List[str],
                       algebras: List[str],
                       return_intermediate_results: bool = False,
                       cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Check if latents are available
        if self.latest_latents is None:
            raise ValueError("No latents available. Make sure the callback is called before calculate_lift.")

        latents = self.latest_latents.clone()
        self.latest_latents = None

        # Compile the unet, might take a while at the first run
        unet = torch.compile(pipeline.unet, mode="max-autotune")

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

            # Encode input prompt
            _, current_prompt_embeds = pipeline._encode_prompt(
                current_prompts,
                device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )

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

            # predict the noise residual
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=current_prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            current_log_lift = (noise_pred_uncond - current_noise).pow(2).mean(dim=(1, 2, 3)) - \
                               (noise_pred_text - current_noise).pow(2).mean(dim=(1, 2, 3))
            # -> (batch_size,)
            log_lift_results[image_idxs[idx:idx + self.config.batch_size],
                             algebra_idxs[idx:idx + self.config.batch_size],
                             trial_idxs[idx:idx + self.config.batch_size]] = current_log_lift

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

        if return_intermediate_results:
            return log_lift_results, is_valid
        else:
            return is_valid

    def __call__(self, i, t, latents) -> Any:
        self.latest_latents = latents
