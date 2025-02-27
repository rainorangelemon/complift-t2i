import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline, StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg
from diffusers.schedulers import DDPMScheduler
from typing import List, Union, Tuple
from torchvision.transforms import GaussianBlur
import matplotlib.pyplot as plt
from copy import deepcopy

THRESHOLD = 200


class UNetForwardLogger:
    def __init__(self, pipe: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
                 component_prompts: List[str],
                 original_size: Tuple[int, int] = (1024, 1024),
                 target_size: Tuple[int, int] = (1024, 1024),
                 crops_coords_top_left: Tuple[int, int] = (0, 0)):
        self.forward_history = []
        self.component_prompts = component_prompts
        self.n_prompts = len(component_prompts) - 1
        self.pipe = pipe
        self.scheduler = deepcopy(pipe.scheduler)
        self.ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=self.scheduler.config.num_train_timesteps,
            beta_start=self.scheduler.config.beta_start,
            beta_end=self.scheduler.config.beta_end,
            beta_schedule=self.scheduler.config.beta_schedule,
        )

        # Check if using SDXL pipeline
        self.is_sdxl = hasattr(pipe, "text_encoder_2")

        # prepare the prompt_embeds
        if self.is_sdxl:
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                component_prompts,
                device=pipe.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            self.prompt_embeds = prompt_embeds
            self.pooled_prompt_embeds = pooled_prompt_embeds
        else:
            self.prompt_embeds, _ = pipe.encode_prompt(
                component_prompts,
                device=pipe.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )

        self.tracked_distances = None
        self.epsilon = 1e-4
        self.n_iterations = 10
        self.step_size = 0.01
        self.timestep_range_to_improve = range(-5, -1)  #range(980, 1000)

        self.original_size = original_size
        self.target_size = target_size
        self.crops_coords_top_left = crops_coords_top_left

    def __call__(self, pipe, step_index, t, callback_kwargs):

        if step_index == 0:
            self.unet = pipe.unet
            self.scheduler.set_timesteps(pipe.num_timesteps, device=pipe.device)

        # Extract relevant arguments from callback_kwargs
        forward_args = {
            'latent_model_input': callback_kwargs['latent_model_input'],
            't': t,
        }

        self.forward_history.append(forward_args)

        # Get the original latents
        latents = callback_kwargs['latents']
        extra_step_kwargs = callback_kwargs['extra_step_kwargs']

        updated_latent = self.test_lift(t=t, **callback_kwargs)

        # check if the updated latents is the same as the latents
        if (updated_latent == latents).all():
            print("No improvement found")
            return {}

        # Add the updated latents to callback_kwargs
        callback_kwargs['latents'] = updated_latent

        self.scheduler = deepcopy(self.pipe.scheduler)

        return callback_kwargs

    def test_lift(self, latent_model_input, t, **callback_kwargs):

        latent_model_input = latent_model_input.chunk(2)[0]
        original_latent_model_input = latent_model_input.clone()
        need_to_update = t in self.timestep_range_to_improve

        best_score = -1
        best_latent_model_input = latent_model_input.clone()

        for _ in range(self.n_iterations):
            # Move this inside the loop to clear memory more aggressively
            torch.cuda.empty_cache()

            tracked_distances, noise_refined, num_satisfied_pixels = self.compute_distances_and_noise(
                latent_model_input, t
            )

            if (not need_to_update) or (num_satisfied_pixels >= THRESHOLD).all():
                best_latent_model_input = latent_model_input.clone()
                best_tracked_distances = tracked_distances.detach()
                break
            else:
                i = num_satisfied_pixels.argmin()
                print(f"Prompt {i} ({self.component_prompts[i]}) has {num_satisfied_pixels[i].item()} satisfied pixels")
                score = num_satisfied_pixels[i].item()
                if score > best_score:
                    best_score = score
                    best_latent_model_input = latent_model_input.clone()
                    best_tracked_distances = tracked_distances.detach()

                # z = torch.randn_like(latent_model_input)
                # epsilon = 2 * self.ddpm_scheduler.alphas.to(latent_model_input.device)[t.to(latent_model_input.device).long()]
                # epsilon = epsilon * ((self.step_size * noise_refined.norm() / z.norm()) ** 0.5)
                # print(f"Epsilon: {epsilon}")

                # latent_model_input = latent_model_input - self.step_size * noise_refined #+ z * ((2 * epsilon) ** 0.5)
                # simply resample a new latent model input
                print(f"Resampling latent model input")
                latent_model_input = torch.randn_like(latent_model_input)
                # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                torch.cuda.empty_cache()

        self.tracked_distances = best_tracked_distances.detach()

        # scale the latents
        scaled_latents = self.scheduler.scale_model_input(best_latent_model_input, t)
        scale_factor = (scaled_latents / best_latent_model_input).mean()

        # check if the scale factor is nan
        if torch.isnan(scale_factor):
            scale_factor = 1.0

        best_latent = best_latent_model_input / scale_factor
        print(f"Scale factor: {scale_factor}")

        # save visualization plots
        plt.clf()
        plt.close("all")
        _, axes = plt.subplots(1, 2, figsize=(10, 5))
        avg = self.tracked_distances.mean(dim=0)
        avg = GaussianBlur(kernel_size=3)(avg)
        axes[0].imshow((avg[1] - avg[0]).relu().cpu().numpy(), cmap="hot")
        axes[0].set_title("first object")
        plt.colorbar(axes[0].images[0], ax=axes[0])
        axes[1].imshow((avg[0] - avg[1]).relu().cpu().numpy(), cmap="hot")
        axes[1].set_title("second object")
        plt.colorbar(axes[1].images[0], ax=axes[1])
        plt.tight_layout()
        plt.savefig(f"tracked_distances_{t}.png")

        plt.clf()
        plt.close("all")
        plt.imshow(best_latent.mean(dim=(0, 1)).cpu().numpy(), cmap="gray")
        plt.savefig(f"best_latent_{t}.png")

        # Handle classifier-free guidance
        if self.pipe.do_classifier_free_guidance:
            # Expand latents for classifier-free guidance
            best_latent_model_input = torch.cat([best_latent_model_input] * 2)

        # Predict noise
        added_cond_kwargs = None
        if self.is_sdxl:
            added_cond_kwargs = callback_kwargs['added_cond_kwargs']

        noise_pred = self.pipe.unet(
            best_latent_model_input,
            t,
            encoder_hidden_states=callback_kwargs['prompt_embeds'],
            cross_attention_kwargs=self.pipe.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        # Apply classifier-free guidance
        if self.pipe.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # if noise_refined is not None:
                # noise_pred = noise_pred_uncond + self.pipe.guidance_scale * (noise_refined - noise_pred_uncond)
            # else:
            noise_pred = noise_pred_uncond + self.pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Apply guidance rescale if enabled
            if self.pipe.guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.pipe.guidance_rescale)

        # Step the scheduler
        best_latent = self.scheduler.step(
            noise_pred,
            t,
            best_latent,
            **callback_kwargs['extra_step_kwargs'],
            return_dict=False,
        )[0]

        # Create new tensor and copy values
        new_latent = torch.zeros_like(best_latent)
        new_latent.copy_(best_latent)
        del best_latent_model_input
        del best_latent
        return new_latent

    def compute_distances_and_noise(self, latent_model_input, t):
        # For SDXL, we need to prepare the additional conditioning
        if self.is_sdxl:
            # Get the composed add_time_ids and text_embeds from the pipe
            add_time_ids = self.pipe._get_add_time_ids(
                original_size=self.original_size,
                crops_coords_top_left=self.crops_coords_top_left,
                target_size=self.target_size,
                dtype=latent_model_input.dtype,
                text_encoder_projection_dim=self.pipe.text_encoder_2.config.projection_dim
            ).to(latent_model_input.device)

            # Prepare the added_cond_kwargs for each component prompt
            added_cond_kwargs = {
                "text_embeds": self.pooled_prompt_embeds,
                "time_ids": add_time_ids.repeat(len(self.component_prompts), 1, 1)
            }

            # Get noise predictions for individual prompts and composed prompt
            packed_noise_preds = self.pipe.unet(
                latent_model_input.repeat(len(self.component_prompts), 1, 1, 1),
                t,
                self.prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

        else:
            # Original SD pipeline logic
            packed_noise_preds = self.pipe.unet(
                latent_model_input.repeat(len(self.component_prompts), 1, 1, 1),
                t,
                self.prompt_embeds,
            ).sample

        noise_preds = packed_noise_preds.chunk(len(self.component_prompts))
        individual_noise_preds = noise_preds[:-1]
        composed_noise_preds = noise_preds[-1]

        dist_to_composed = [(composed_noise_preds - individual_noise_preds[i]).pow(2).mean(dim=(0, 1))
                           for i in range(self.n_prompts)]
        dist_to_composed = torch.stack(dist_to_composed, dim=0)

        if self.tracked_distances is None:
            tracked_distances = dist_to_composed[None]
        else:
            tracked_distances = torch.cat((self.tracked_distances, dist_to_composed[None]), dim=0)

        average_dist = tracked_distances.mean(dim=0)
        average_dist = GaussianBlur(kernel_size=3)(average_dist)

        num_satisfied_pixels = torch.zeros(self.n_prompts, dtype=torch.long, device=average_dist.device)

        suggested_noise = composed_noise_preds.clone()
        for i in range(self.n_prompts):
            other_prompts_mask = torch.ones(self.n_prompts, dtype=torch.bool, device=average_dist.device)
            other_prompts_mask[i] = False

            margin = (average_dist[i] - (average_dist[other_prompts_mask] - self.epsilon))
            is_less_than_others = (margin < 0).all(dim=0)
            print(f"Number of pixels for prompt {i} ({self.component_prompts[i]}) that has average dist alpha less than the other prompts: {is_less_than_others.sum().item()}")

            margin_to_next_closest_prompt = margin.min(dim=0).values
            is_best_k_pixels = margin_to_next_closest_prompt.flatten().topk(k=THRESHOLD, largest=False).indices
            is_best_k_pixels_mask = torch.zeros_like(margin_to_next_closest_prompt, dtype=torch.bool)
            is_best_k_pixels_mask.flatten()[is_best_k_pixels] = True
            is_best_k_pixels_mask = is_best_k_pixels_mask.reshape(margin_to_next_closest_prompt.shape)
            # save the mask to a image
            plt.clf()
            plt.close("all")
            _, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(is_best_k_pixels_mask.cpu().numpy(), cmap="gray")
            axes[0].set_title("is_best_k_pixels_mask")
            axes[1].imshow(margin_to_next_closest_prompt.cpu().numpy(), cmap="hot")
            axes[1].set_title("margin_to_next_closest_prompt")
            plt.tight_layout()
            plt.savefig(f"is_best_k_pixels_mask_{t}_{i}.png")
            if is_less_than_others.sum() < THRESHOLD:
                print(f"Prompt {i} ({self.component_prompts[i]}) has less than {THRESHOLD} satisfied pixels")
                suggested_noise = individual_noise_preds[i] * is_best_k_pixels_mask# + suggested_noise * (~is_best_k_pixels_mask)

            num_satisfied_pixels[i] += is_less_than_others.sum()

        # Add memory cleanup
        del packed_noise_preds
        torch.cuda.empty_cache()

        return tracked_distances, suggested_noise, num_satisfied_pixels


if __name__ == "__main__":
    # Test with either SD or SDXL
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"  # or "CompVis/stable-diffusion-v1-4"
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")

    # Add required tensor inputs based on pipeline type
    base_tensor_inputs = [
        "latent_model_input",
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "timestep_cond",
        "cross_attention_kwargs",
        "added_cond_kwargs",
        "extra_step_kwargs"
    ]

    if hasattr(pipe, "text_encoder_2"):
        # SDXL additional inputs
        base_tensor_inputs.extend(["pooled_prompt_embeds", "add_text_embeds"])
    pipe._callback_tensor_inputs += base_tensor_inputs

    # Initialize logger and generate image
    logger = UNetForwardLogger(pipe, ["black car", "white clock", "a black car and a white clock"])
    prompt = "a black car and a white clock"

    # Set generation parameters
    kwargs = {
        "prompt": prompt,
        "num_inference_steps": 50,
        "callback_on_step_end": logger,
        "callback_on_step_end_tensor_inputs": base_tensor_inputs,
    }

    # Add SDXL-specific parameters if needed
    if hasattr(pipe, "text_encoder_2"):
        kwargs.update({
            "original_size": (1024, 1024),
            "target_size": (1024, 1024),
            "crops_coords_top_left": (0, 0),
        })

    # Generate image
    torch.manual_seed(2)
    torch.cuda.manual_seed(2)
    image = pipe(**kwargs).images[0]
    image.save(f"{prompt}.png")
