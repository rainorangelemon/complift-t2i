import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
    rescale_noise_cfg,
    StableDiffusionXLPipelineOutput,
    retrieve_timesteps,
)
from diffusers.utils import logging
from utils.gaussian_smoothing import GaussianSmoothing
from utils.ptp_utils import AttentionStore, aggregate_attention

logger = logging.get_logger(__name__)

class AttendAndExcitePipelineXL(StableDiffusionXLPipeline):
    def _compute_max_attention_per_index(self,
                                       attention_maps: torch.Tensor,
                                       indices_to_alter: List[int],
                                       smooth_attentions: bool = False,
                                       sigma: float = 0.5,
                                       kernel_size: int = 3,
                                       normalize_eot: bool = False) -> List[torch.Tensor]:
        """ Computes the maximum attention value for each of the tokens we wish to alter. """
        last_idx = -1
        if normalize_eot:
            prompt = self.prompt
            if isinstance(self.prompt, list):
                prompt = self.prompt[0]
            last_idx = len(self.tokenizer_2.encode(prompt)) - 1  # Use tokenizer_2 for SDXL
        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        indices_to_alter = [index - 1 for index in indices_to_alter]

        # Extract the maximum values
        max_indices_list = []
        for i in indices_to_alter:
            image = attention_for_text[:, :, i]
            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                image = smoothing(input).squeeze(0).squeeze(0)
            max_indices_list.append(image.max())
        return max_indices_list

    def _aggregate_and_get_max_attention_per_token(self, attention_store: AttentionStore,
                                                 indices_to_alter: List[int],
                                                 attention_res: int = 16,
                                                 smooth_attentions: bool = False,
                                                 sigma: float = 0.5,
                                                 kernel_size: int = 3,
                                                 normalize_eot: bool = False):
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)
        max_attention_per_index = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices_to_alter=indices_to_alter,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot)
        return max_attention_per_index

    @staticmethod
    def _compute_loss(max_attention_per_index: List[torch.Tensor], return_losses: bool = False) -> torch.Tensor:
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        losses = [max(0, 1. - curr_max) for curr_max in max_attention_per_index]
        loss = max(losses)
        if return_losses:
            return loss, losses
        else:
            return loss

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents

    def _perform_iterative_refinement_step(self,
                                         latents: torch.Tensor,
                                         indices_to_alter: List[int],
                                         loss: torch.Tensor,
                                         threshold: float,
                                         text_embeddings: torch.Tensor,
                                         text_input,
                                         attention_store: AttentionStore,
                                         step_size: float,
                                         t: int,
                                         attention_res: int = 16,
                                         smooth_attentions: bool = True,
                                         sigma: float = 0.5,
                                         kernel_size: int = 3,
                                         max_refinement_steps: int = 20,
                                         normalize_eot: bool = False):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens.
        """

        iteration = 0
        target_loss = max(0, 1. - threshold)
        while loss > target_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)

            # Adapt for SDXL's added condition inputs
            added_cond_kwargs = {"text_embeds": text_embeddings[2], "time_ids": text_embeddings[3]}
            noise_pred = self.unet(
                latents,
                t,
                encoder_hidden_states=text_embeddings[1].unsqueeze(0),
                added_cond_kwargs=added_cond_kwargs,
            ).sample
            self.unet.zero_grad()

            max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                attention_store=attention_store,
                indices_to_alter=indices_to_alter,
                attention_res=attention_res,
                smooth_attentions=smooth_attentions,
                sigma=sigma,
                kernel_size=kernel_size,
                normalize_eot=normalize_eot
            )

            loss, losses = self._compute_loss(max_attention_per_index, return_losses=True)

            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)

            with torch.no_grad():
                # Get predictions with updated latents
                noise_pred_uncond = self.unet(
                    latents,
                    t,
                    encoder_hidden_states=text_embeddings[0].unsqueeze(0),
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
                noise_pred_text = self.unet(
                    latents,
                    t,
                    encoder_hidden_states=text_embeddings[1].unsqueeze(0),
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

            try:
                low_token = np.argmax([l.item() if type(l) != int else l for l in losses])
            except Exception as e:
                print(e)
                low_token = np.argmax(losses)

            # Use tokenizer_2 for SDXL
            if isinstance(text_input, dict):
                input_ids = text_input["input_ids"][0]  # Access input_ids from dict
            else:
                input_ids = text_input.input_ids[0]  # Original behavior

            low_word = self.tokenizer_2.decode(input_ids[indices_to_alter[low_token]])
            print(f'\t Try {iteration}. {low_word} has a max attention of {max_attention_per_index[low_token]}')

            if iteration >= max_refinement_steps:
                print(f'\t Exceeded max number of iterations ({max_refinement_steps})! '
                      f'Finished with a max attention of {max_attention_per_index[low_token]}')
                break

        # Run one more time but don't compute gradients and update the latents
        latents = latents.clone().detach().requires_grad_(True)
        noise_pred = self.unet(
            latents,
            t,
            encoder_hidden_states=text_embeddings[1].unsqueeze(0),
            added_cond_kwargs=added_cond_kwargs,
        ).sample
        self.unet.zero_grad()

        max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
            attention_store=attention_store,
            indices_to_alter=indices_to_alter,
            attention_res=attention_res,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot)
        loss, losses = self._compute_loss(max_attention_per_index, return_losses=True)
        print(f"\t Finished with loss of: {loss}")
        return loss, latents, max_attention_per_index

    def get_text_inputs(self, prompt):
        """
        Tokenize the prompt to get text inputs.

        Args:
            prompt (`str` or `List[str]`): The prompt to tokenize

        Returns:
            `dict`: Tokenized text inputs with input_ids and attention_mask
        """
        # Handle both string and list inputs
        if isinstance(prompt, str):
            prompt = [prompt]

        text_inputs = self.tokenizer_2(  # Using tokenizer_2 for SDXL
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        return text_inputs

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        attention_store: AttentionStore,
        indices_to_alter: List[int],
        attention_res: int = 32,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        max_iter_to_alter: Optional[int] = 25,
        run_standard_sd: bool = False,
        thresholds: Optional[dict] = {0: 0.05, 10: 0.5, 20: 0.8},
        scale_factor: int = 20,
        scale_range: Tuple[float, float] = (1., 0.5),
        smooth_attentions: bool = True,
        sigma: float = 0.5,
        kernel_size: int = 3,
        normalize_eot: bool = False,
    ):

        # 1. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # Add this line to define batch_size
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        self.prompt = prompt
        device = self._execution_device

        # 2. Set timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device,
        )

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=guidance_scale > 1.0,
            negative_prompt=negative_prompt,
            prompt_2=prompt_2,
            negative_prompt_2=negative_prompt_2,
        )

        text_input = self.get_text_inputs(prompt)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Add image embeds for IP-Adapter (if using)
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds.to(device),
            "time_ids": self._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=self.text_encoder_2.config.projection_dim
            ).to(device)
        }

        # 8. Denoising loop
        scale_range = np.linspace(scale_range[0], scale_range[1], len(timesteps))

        if max_iter_to_alter is None:
            max_iter_to_alter = len(timesteps) + 1

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                with torch.enable_grad():
                    latents = latents.clone().detach().requires_grad_(True)

                    # Forward pass of denoising with text conditioning
                    noise_pred = self.unet(
                        latents,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample
                    self.unet.zero_grad()

                    # Get max activation value for each subject token
                    max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                        attention_store=attention_store,
                        indices_to_alter=indices_to_alter,
                        attention_res=attention_res,
                        smooth_attentions=smooth_attentions,
                        sigma=sigma,
                        kernel_size=kernel_size,
                        normalize_eot=normalize_eot,
                    )

                    if not run_standard_sd:
                        loss = self._compute_loss(max_attention_per_index=max_attention_per_index)

                        # Refinement step if needed
                        if i in thresholds.keys() and loss > 1. - thresholds[i]:
                            del noise_pred
                            torch.cuda.empty_cache()

                            loss, latents, max_attention_per_index = self._perform_iterative_refinement_step(
                                latents=latents,
                                indices_to_alter=indices_to_alter,
                                loss=loss,
                                threshold=thresholds[i],
                                text_embeddings=(negative_prompt_embeds, prompt_embeds, pooled_prompt_embeds, added_cond_kwargs["time_ids"]),
                                text_input=text_input,
                                attention_store=attention_store,
                                step_size=scale_factor * np.sqrt(scale_range[i]),
                                t=t,
                                attention_res=attention_res,
                                smooth_attentions=smooth_attentions,
                                sigma=sigma,
                                kernel_size=kernel_size,
                                normalize_eot=normalize_eot,
                            )

                        # Perform gradient update
                        if i < max_iter_to_alter:
                            loss = self._compute_loss(max_attention_per_index=max_attention_per_index)
                            if loss != 0:
                                latents = self._update_latent(
                                    latents=latents,
                                    loss=loss,
                                    step_size=scale_factor * np.sqrt(scale_range[i])
                                )
                            print(f'Iteration {i} | Loss: {loss:0.4f}')

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]) if guidance_scale > 1.0 else prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                # perform guidance
                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if guidance_scale > 1.0 and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug
                    self.vae = self.vae.to(latents.dtype)

            # Check for latents normalization parameters
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None

            if has_latents_mean and has_latents_std:
                latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)