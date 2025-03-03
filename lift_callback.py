import pprint
from typing import List, Optional, Dict, Any, Union

import pyrallis
import torch
from PIL import Image
from tqdm import trange
import os
from pathlib import Path
from copy import deepcopy

from config import LiftConfig
from pipeline_attend_and_excite import AttendAndExcitePipeline
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDPMScheduler, PNDMScheduler, SchedulerMixin
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class LiftCallback:

    def __init__(self, config: LiftConfig):
        self.config = config
        self.latest_latents = None
        self.intermediate_latents = []
        self.intermediate_ts = []

    def get_timesteps(self, scheduler: DDPMScheduler):
        n_trials = self.config.n_samples
        num_train_timesteps = len(scheduler.timesteps)
        timesteps = scheduler.timesteps
        if self.config.t_schedule == "interleave":
            ts = torch.linspace(0, num_train_timesteps-1, n_trials).round().long().clamp(0, num_train_timesteps-1)
            ts = timesteps[ts.long()]
        elif self.config.t_schedule == "random":
            ts = torch.randint(0, num_train_timesteps, (n_trials,))
            ts = timesteps[ts.long()]
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
            return noise[[0], ...].expand(self.config.n_samples, *latent_shape)
        else:
            return noise[:self.config.n_samples]


    @torch.inference_mode()
    def calculate_score(self,
                        latents,
                        pipeline: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
                        prompts: List[str],
                        noise: Optional[torch.Tensor] = None,
                        timesteps: Optional[torch.Tensor] = None,
                        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        device = latents.device
        # print whether the scheduler uses v-prediction or epsilon or others
        assert pipeline.scheduler.config.prediction_type == "epsilon", "Only epsilon prediction is supported for unet"
        scheduler_config = pipeline.scheduler.config
        scheduler = DDPMScheduler(beta_schedule=scheduler_config["beta_schedule"],
                                 beta_start=scheduler_config["beta_start"],
                                 beta_end=scheduler_config["beta_end"],
                                 num_train_timesteps=scheduler_config["num_train_timesteps"])
        # set scheduler to training mode
        scheduler.set_timesteps(num_inference_steps=1000, device="cuda")

        # Get the UNet's dtype
        unet_dtype = pipeline.unet.dtype

        # Convert latents to UNet's dtype
        latents = latents.to(dtype=unet_dtype)

        n_prompts = len(prompts)
        # -> (n_trials,)
        B, *latent_shape = latents.shape
        if noise is None:
            noise = self.get_noise(latent_shape).to(device, dtype=unet_dtype)
        else:
            noise = noise.to(device, dtype=unet_dtype)
        # -> (n_trials, *Latent_shape)
        if timesteps is None:
            ts = self.get_timesteps(scheduler).to(device)
        else:
            ts = timesteps.to(device)
        # -> (n_trials,)

        score_results = torch.zeros((B, n_prompts, len(ts), *latent_shape), dtype=unet_dtype)
        image_idxs, algebra_idxs, trial_idxs = self._get_model_inputs_and_indices(
            device, ts, B, n_prompts
        )

        idx = 0
        for _ in trange(len(trial_idxs) // self.config.batch_size + int(len(trial_idxs) % self.config.batch_size != 0), leave=False):
            batch_size = min(self.config.batch_size, len(trial_idxs) - idx)
            current_batch = (
                scheduler.add_noise(
                    latents[image_idxs[idx:idx + batch_size]],
                    noise[trial_idxs[idx:idx + batch_size]],
                    ts[trial_idxs[idx:idx + batch_size]]
                ),
                [prompts[i] for i in algebra_idxs[idx:idx + batch_size]],
                ts[trial_idxs[idx:idx + batch_size]],
                (image_idxs[idx:idx + batch_size],
                 algebra_idxs[idx:idx + batch_size],
                 trial_idxs[idx:idx + batch_size])
            )
            self._process_batch(pipeline, score_results, current_batch, device, cross_attention_kwargs)
            idx += batch_size

        return score_results

    @torch.inference_mode()
    def calculate_score_with_latent_model_inputs(self,
                        pipeline: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
                        prompts: List[str],
                        latent_model_inputs: torch.Tensor,
                        timesteps: torch.Tensor,
                        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        device = latent_model_inputs.device
        # print whether the scheduler uses v-prediction or epsilon or others
        assert pipeline.scheduler.config.prediction_type == "epsilon", "Only epsilon prediction is supported for unet"
        # Get the UNet's dtype
        unet_dtype = pipeline.unet.dtype

        # Convert latents to UNet's dtype
        latent_model_inputs = latent_model_inputs.to(dtype=unet_dtype)

        n_prompts = len(prompts)
        B, n_trials, *latent_shape = latent_model_inputs.shape
        # -> (B, n_trials, *Latent_shape)
        ts = timesteps.to(device)
        # -> (n_trials,)
        assert len(ts) == n_trials, "timesteps must have the same length as the number of trials"

        score_results = torch.zeros((B, n_prompts, len(ts), *latent_shape), dtype=unet_dtype)
        image_idxs, algebra_idxs, trial_idxs = self._get_model_inputs_and_indices(
            device, ts, B, n_prompts
        )

        idx = 0
        for _ in trange(len(trial_idxs) // self.config.batch_size + int(len(trial_idxs) % self.config.batch_size != 0), leave=False):
            batch_size = min(self.config.batch_size, len(trial_idxs) - idx)
            current_batch = (
                latent_model_inputs[
                    image_idxs[idx:idx + batch_size],
                    trial_idxs[idx:idx + batch_size]
                ],
                [prompts[i] for i in algebra_idxs[idx:idx + batch_size]],
                ts[trial_idxs[idx:idx + batch_size]],
                (image_idxs[idx:idx + batch_size],
                 algebra_idxs[idx:idx + batch_size],
                 trial_idxs[idx:idx + batch_size])
            )
            self._process_batch(pipeline, score_results, current_batch, device, cross_attention_kwargs)
            idx += batch_size

        return score_results

    def __call__(self, pipe, i, t, callback_kwargs) -> Any:
        # detach and clone to avoid memory leak
        self.latest_latents = callback_kwargs['latents'].detach().clone()
        if self.config.save_intermediate_latent:
            # check if use cfg
            if pipe.guidance_scale > 1:
                self.intermediate_latents.append(callback_kwargs['latent_model_input'][0].detach().clone())
            else:
                self.intermediate_latents.append(callback_kwargs['latent_model_input'].detach().clone())
            self.intermediate_ts.append(t)
        return {}

    def clear(self):
        self.latest_latents = None
        self.intermediate_latents = []
        self.intermediate_ts = []

    # ============================ Helper methods ============================

    def _get_model_inputs_and_indices(self, device, ts, B, n_prompts):
        """Helper method to prepare model inputs and indices."""
        image_idxs = torch.arange(B, device=device)[:, None, None].expand(-1, n_prompts, len(ts)).flatten().to(device)
        algebra_idxs = torch.arange(n_prompts, device=device)[None, :, None].expand(B, -1, len(ts)).flatten().to(device)
        trial_idxs = torch.arange(len(ts), device=device)[None, None, :].expand(B, n_prompts, -1).flatten().to(device)
        return image_idxs, algebra_idxs, trial_idxs

    def _predict_noise(self, pipeline, latent_model_input, t, prompts, device, cross_attention_kwargs):
        """Helper method to handle noise prediction for both SD and SDXL."""
        if isinstance(pipeline, StableDiffusionPipeline):
            prompt_embeds, _ = pipeline.encode_prompt(
                prompts, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False,
            )
            return pipeline.unet(
                latent_model_input, t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

        elif isinstance(pipeline, StableDiffusionXLPipeline):
            prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
                prompts, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False,
            )

            # Get timestep conditioning
            timestep_cond = None
            if pipeline.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(pipeline.guidance_scale - 1).repeat(len(t))
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
            ).to(device).repeat(len(t), 1)

            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

            return pipeline.unet(
                latent_model_input, t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

    def _process_batch(self, pipeline, score_results, current_batch, device, cross_attention_kwargs):
        """Helper method to process a batch of inputs and update score results."""
        latent_input, prompts, ts, indices = current_batch
        noise_pred = self._predict_noise(pipeline, latent_input, ts, prompts, device, cross_attention_kwargs)
        image_idxs, algebra_idxs, trial_idxs = indices
        score_results[image_idxs, algebra_idxs, trial_idxs] = noise_pred.cpu()
        return score_results
