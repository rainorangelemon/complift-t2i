# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %autoreload 2

# %cd ..

import pandas as pd
import os

def clean_object_name(object_name):
    if "a " in object_name:
        return object_name.split("a ")[1]
    else:
        return object_name

import torch
all_final_latents = torch.zeros(100, 4, 128, 128)
all_intermediate_latents = torch.zeros(100, 50, 4, 128, 128)
for i in range(10):
    lift_results = torch.load(f"outputs/weak_lift/a black car and a white clock/{i}_lift_results.pt", weights_only=True)
    all_final_latents[i] = lift_results["latents"]
    all_intermediate_latents[i] = torch.stack(lift_results["intermediate_latents"]).squeeze(1)
latest_latents = all_final_latents.to("cuda:0")

from diffusers import DiffusionPipeline
import torch
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True)
pipeline.to("cuda")
pipeline.unet = torch.compile(pipeline.unet, mode="max-autotune", fullgraph=True)
# offload vae and text encoder
pipeline.vae.to("cpu");

from typing import List, Optional, Dict, Any, Union
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDPMScheduler
from lift_callback import LiftCallback
from tqdm import trange

@torch.inference_mode()
def calculate_score(latents,
                    callback: LiftCallback,
                    pipeline: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
                    prompts: List[str],
                    noise: Optional[torch.Tensor] = None,
                    timesteps: Optional[torch.Tensor] = None,
                    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
):

    config = callback.config

    # Compile the unet, might take a while at the first run
    unet = pipeline.unet
    # unet = torch.compile(pipeline.unet, mode="max-autotune")

    device = latents.device
    # print whether the scheduler uses v-prediction or epsilon or others
    assert pipeline.scheduler.config.prediction_type == "epsilon", "Only epsilon prediction is supported for unet"
    scheduler = DDPMScheduler(beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012)
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
        noise = callback.get_noise(latent_shape).to(device, dtype=unet_dtype)
    else:
        noise = noise.to(device, dtype=unet_dtype)
    if timesteps is None:
        ts = callback.get_timesteps(scheduler).to(device)
    else:
        ts = timesteps.to(device)
    # -> (n_trials, *Latent_shape)
    num_latent_dims = len(latents.shape) - 1

    score_results = torch.zeros((B, n_prompts, len(ts), *latent_shape), dtype=unet_dtype)

    image_idxs = torch.arange(B, device=device)[:, None, None].expand(-1, n_prompts, len(ts)).flatten().to(device)
    algebra_idxs = torch.arange(n_prompts, device=device)[None, :, None].expand(B, -1, len(ts)).flatten().to(device)
    trial_idxs = torch.arange(len(ts), device=device)[None, None, :].expand(B, n_prompts, -1).flatten().to(device)

    idx = 0
    for _ in trange(len(trial_idxs) // config.batch_size + int(len(trial_idxs) % config.batch_size != 0), leave=False):
        current_latents = latents[image_idxs[idx:idx + config.batch_size]]
        current_prompts = [prompts[idx] for idx in algebra_idxs[idx:idx + config.batch_size]]
        current_noise = noise[trial_idxs[idx:idx + config.batch_size]]
        current_ts = ts[trial_idxs[idx:idx + config.batch_size]]

        noisy_latents = scheduler.add_noise(current_latents, current_noise, current_ts)

        # Double the batch size to add unconditional prediction
        latent_model_input = noisy_latents
        t = current_ts

        # Encode input prompt
        if isinstance(pipeline, StableDiffusionPipeline):
            prompt_embeds, _ = pipeline.encode_prompt(
                current_prompts,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            current_prompt_embeds = prompt_embeds
            # predict the noise residual
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=current_prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

        elif isinstance(pipeline, StableDiffusionXLPipeline):
            # Encode input prompt
            (prompt_embeds, _, pooled_prompt_embeds, _) = pipeline.encode_prompt(
                current_prompts,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )

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
            add_time_ids = add_time_ids.to(device).repeat(len(current_ts), 1)

            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

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

        noise_pred_text = noise_pred
        # -> (batch_size,)
        score_results[image_idxs[idx:idx + config.batch_size],
                            algebra_idxs[idx:idx + config.batch_size],
                            trial_idxs[idx:idx + config.batch_size]] = noise_pred_text.cpu()

        idx += len(current_ts)
    return score_results

ddpm_scheduler = DDPMScheduler(beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012)

pipeline.scheduler.set_timesteps(num_inference_steps=50, device="cuda")
sqrt_alpha_prod = ddpm_scheduler.alphas_cumprod ** 0.5
sqrt_one_minus_alpha_prod = (1 - ddpm_scheduler.alphas_cumprod) ** 0.5
prev_t = pipeline.scheduler.timesteps[1:]
x0 = all_intermediate_latents[:, -1]
scaled_x0 = torch.einsum("t, b c h w -> b t c h w", sqrt_alpha_prod[prev_t.cpu().long()], x0)

pipeline.scheduler.set_timesteps(timesteps=prev_t.long().cpu(), num_inference_steps=None)
sigma = pipeline.scheduler.sigmas.to(prev_t.device)[:-1]
intended_noise = (all_intermediate_latents[:, :-1] - x0[:, None]) / sigma[:, None, None, None].to(all_intermediate_latents.device)

scaled_intermediate_latents = all_intermediate_latents[:, :-1] / ((sigma**2 + 1) ** 0.5)[:, None, None, None].to(all_intermediate_latents.device)
(ddpm_scheduler.add_noise(x0, intended_noise[:, 1], prev_t[1].long()) - scaled_intermediate_latents[:, 1]).abs().max()

from lift_callback import LiftCallback
from config import LiftConfig
from tqdm import tqdm
config = LiftConfig(prompt="a black car and a white clock", subtract_unconditional=False,
                    batch_size=8,
                    same_noise=False,
)
callback = LiftCallback(config)

# Check if cached scores exist
scores_path = "outputs/weak_lift/a black car and a white clock/scores.pt"
if os.path.exists(scores_path):
    # Load cached scores
    cached_scores = torch.load(scores_path, weights_only=True)
    score_composed_all = cached_scores["composed"]
    score_black_car_all = cached_scores["black_car"]
    score_white_clock_all = cached_scores["white_clock"]
    score_uncond_all = cached_scores["uncond"]
else:
    # Compute scores
    score_composed_all = []
    score_black_car_all = []
    score_white_clock_all = []
    score_uncond_all = []
    for img_idx in tqdm(range(10)):
        score_results = calculate_score(latest_latents[[img_idx]],
                                     callback,
                                     pipeline,
                                     prompts=["a black car and a white clock", "a black car", "a white clock", ""],
                                     # noise=intended_noise[img_idx],
                                     # timesteps=prev_t.long(),
                                     )
        score_composed_all.append(score_results[:, 0])
        score_black_car_all.append(score_results[:, 1])
        score_white_clock_all.append(score_results[:, 2])
        score_uncond_all.append(score_results[:, -1])

    # Concatenate results
    score_composed_all = torch.cat(score_composed_all, dim=0)
    score_black_car_all = torch.cat(score_black_car_all, dim=0)
    score_white_clock_all = torch.cat(score_white_clock_all, dim=0)
    score_uncond_all = torch.cat(score_uncond_all, dim=0)

    # Cache the results
    torch.save({
        "composed": score_composed_all,
        "black_car": score_black_car_all,
        "white_clock": score_white_clock_all,
        "uncond": score_uncond_all
    }, scores_path)

import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from torchvision.transforms import GaussianBlur
import torch
from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.rc('text', usetex = True)
plt.rcParams['xtick.bottom'] = False
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False


@dataclass
class VisualizationConfig:
    use_gaussian_smoothing: bool = False
    draw_pure_lift: bool = False
    draw_trajectories: bool = False
    base_type: str = "composed"
    threshold: float = 5e-5
    specific_indices: Optional[List[int]] = None
    fig_size: Tuple[int, int] = (45, 50)
    output_path: str = "outputs/weak_lift/a black car and a white clock"

class HeatmapGenerator:
    def __init__(self, config: VisualizationConfig):
        self.config = config

    def apply_gaussian_smoothing(self, heatmap: torch.Tensor) -> torch.Tensor:
        if self.config.use_gaussian_smoothing:
            return GaussianBlur(kernel_size=3)(heatmap[None].float())[0]
        return heatmap

    def calculate_pure_lift(self, score_target: torch.Tensor, base: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.config.base_type in ["composed", "uncond"]:
            return (score_target - base).pow(2).mean(dim=(1, 2))
        elif self.config.base_type == "noise":
            return ((base - noise).pow(2) - (score_target - noise).pow(2)).mean(dim=(1, 2))

    def calculate_relative_lift(self, base: torch.Tensor, current_score: torch.Tensor, other_scores: list[torch.Tensor]) -> torch.Tensor:
        """
        Calculate relative lift by comparing current score's MSE against minimum MSE from other scores.

        Args:
            base: Base tensor to calculate MSE against
            current_score: Score tensor for current object
            other_scores: List of score tensors for other objects

        Returns:
            Relative lift tensor
        """
        if torch.allclose(base, data["score_composed_all"]):
            current_mse = (base - current_score).pow(2).mean(dim=(1, 2))

            # Stack all other scores' MSEs and get minimum
            other_mses = torch.stack([
                (base - other_score).pow(2).mean(dim=(1, 2))
                for other_score in other_scores
            ])
            min_other_mse = other_mses.min(dim=0)[0]

            return (min_other_mse - current_mse)

        else:
            current_mse = (base - current_score).pow(2).mean(dim=(1, 2))

            # Stack all other scores' MSEs and get maximum
            other_mses = torch.stack([
                (base - other_score).pow(2).mean(dim=(1, 2))
                for other_score in other_scores
            ])
            max_other_mse = other_mses.max(dim=0)[0]

            return (current_mse - max_other_mse)


    def generate_heatmap(self, data, score_target: torch.Tensor, base: torch.Tensor,
                        noise: Optional[torch.Tensor] = None, idx: int = 0,) -> torch.Tensor:
        if self.config.draw_pure_lift:
            heatmap = self.calculate_pure_lift(score_target, base, noise)[idx]
        else:
                # Get all score tensors except the target one
                other_scores = [
                    data['score_uncond_all']
                    # data[f'score_{name}_all']
                    # for name in ['black_car', 'white_clock']
                    # if not torch.allclose(data[f'score_{name}_all'], score_target)
                ]
                heatmap = self.calculate_relative_lift(base, score_target, other_scores)[idx]

        heatmap = self.apply_gaussian_smoothing(heatmap)
        return (heatmap - self.config.threshold)

class TrajectoryPlotter:
    def __init__(self, config: VisualizationConfig):
        self.config = config

    def calculate_trajectory(self, base: torch.Tensor, score1: torch.Tensor,
                           score2: torch.Tensor, idx: int) -> torch.Tensor:
        diff_along_trajectory = ((base - score1).pow(2) - (base - score2).pow(2))[idx]

        if self.config.use_gaussian_smoothing:
            diff_along_trajectory = GaussianBlur(kernel_size=3)(diff_along_trajectory.float())

        trajectory_range = torch.arange(1, diff_along_trajectory.shape[0] + 1)
        cumsum = diff_along_trajectory.cumsum(dim=0)
        normalized = cumsum / trajectory_range[:, None, None, None].to(diff_along_trajectory.device)
        mean_diff = normalized.mean(dim=1).flatten(1, 2)

        return (mean_diff > self.config.threshold).sum(dim=-1)

class VisualizationPlotter:
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.heatmap_gen = HeatmapGenerator(config)
        self.trajectory_plotter = TrajectoryPlotter(config)
        self.title_fontsize = 30
        self.colorbar_fontsize = 20
        self.n_rows = 2# len(self.config.specific_indices) if self.config.specific_indices else self.config.n_samples
        self.n_cols = 3#8 if self.config.draw_trajectories else 6

    def setup_plot(self) -> Tuple[plt.Figure, np.ndarray]:
        plt.clf()
        plt.close("all")
        width = self.config.fig_size[0] * (self.n_cols / 8)
        height = self.config.fig_size[1] * (self.n_rows / 10)

        # Create figure with GridSpec
        fig = plt.figure(figsize=(width, height))
        gs = fig.add_gridspec(
            self.n_rows,
            self.n_cols,
            # width_ratios=[1, 1, 1, 1, 1],  # Equal width for all columns
            hspace=0.18,
            wspace=0.2  # Default spacing
        )

        # Create axes array
        ax = np.array([[fig.add_subplot(gs[i, j]) for j in range(self.n_cols)]
                       for i in range(self.n_rows)])

        if self.n_rows == 1:
            ax = ax.reshape(1, -1)

        # Adjust specific column spacings after creation
        for i in range(self.n_rows):
            ax[i,0].set_position([ax[i,0].get_position().x0,
                                ax[i,0].get_position().y0,
                                ax[i,0].get_position().width,
                                ax[i,0].get_position().height])

            ax[i,1].set_position([ax[i,1].get_position().x0,
                                ax[i,1].get_position().y0,
                                ax[i,1].get_position().width,
                                ax[i,1].get_position().height])

            ax[i,2].set_position([ax[i,2].get_position().x0 + 0.01,
                                ax[i,2].get_position().y0,
                                ax[i,2].get_position().width,
                                ax[i,2].get_position().height])

            # ax[i,3].set_position([ax[i,3].get_position().x0,
            #                     ax[i,3].get_position().y0,
            #                     ax[i,3].get_position().width,
            #                     ax[i,3].get_position().height])

            # ax[i,4].set_position([ax[i,4].get_position().x0 + 0.011,
            #                     ax[i,4].get_position().y0,
            #                     ax[i,4].get_position().width,
            #                     ax[i,4].get_position().height])

            # ax[i,5].set_position([ax[i,5].get_position().x0 + 0.022,
            #                     ax[i,5].get_position().y0,
            #                     ax[i,5].get_position().width,
            #                     ax[i,5].get_position().height])

        return fig, ax

    def plot_original_image(self, ax, idx: int, row_idx: int):
        # Load and convert image to numpy array
        image = Image.open(f"{self.config.output_path}/{idx}.png")
        image_array = np.array(image)

        # Display image without interpolation to maintain sharpness
        im = ax[0].imshow(image_array, interpolation='nearest')
        title = "Image (1024x1024)"
        ax[0].set_title(title, fontsize=self.title_fontsize)

        # Set aspect to 'auto' to fill the entire subplot
        ax[0].set_aspect('auto')

    def plot_latents(self, ax, ax_idx, latents: torch.Tensor, row_idx: int):
        im = ax[ax_idx].imshow(latents.cpu().numpy(), cmap="gray", interpolation='nearest')
        divider = make_axes_locatable(ax[ax_idx])
        cax = divider.append_axes("right", size="4%", pad=0.2)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=self.colorbar_fontsize)
        cbar.ax.yaxis.offsetText.set_fontsize(self.colorbar_fontsize)
        title = "Latent (128x128)"
        ax[ax_idx].set_title(title, fontsize=self.title_fontsize)

        # Set aspect to 'auto' to fill the entire subplot
        ax[ax_idx].set_aspect('auto')

    def plot_heatmap(self, ax, heatmap: torch.Tensor, title: str, row_idx: int):
        if heatmap.relu().cpu().numpy().max() > 1e-8:
            im = ax.imshow(heatmap.relu().cpu().numpy(), cmap="turbo", interpolation='nearest')
        else:
            im = ax.imshow(heatmap.relu().cpu().numpy(), cmap="turbo", vmin=0, vmax=1e-8, interpolation='nearest')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.2)

        if heatmap.relu().cpu().numpy().max() <= 1e-8:
            cbar = plt.colorbar(im, cax=cax, ticks=[0])
            cbar.ax.tick_params(labelsize=self.colorbar_fontsize)
            cbar.ax.yaxis.get_offset_text().set_fontsize(self.colorbar_fontsize)
        else:
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=self.colorbar_fontsize)
            cbar.ax.yaxis.get_offset_text().set_fontsize(self.colorbar_fontsize)
            cbar.ax.yaxis.set_offset_position('left')

        obj_name = title.split("for ")[-1]
        display_title = title if row_idx == 0 else ""
        ax.set_title(f"{display_title}",#\n\#Activated Pixels: {(heatmap > self.config.threshold).sum().item()}",
                    fontsize=self.title_fontsize)

        # Set aspect to 'auto' to fill the entire subplot
        ax.set_aspect('auto')

    def plot_trajectory(self, ax, trajectory: torch.Tensor):
        ax.plot(trajectory.cpu().numpy(), marker='o')
        ax.tick_params(axis='both', which='major', labelsize=self.colorbar_fontsize)

    def create_visualization(self, data: dict):
        fig, ax = self.setup_plot()
        indices_to_plot = self.config.specific_indices if self.config.specific_indices else range(self.config.n_samples)

        for plot_idx, img_idx in enumerate(tqdm(indices_to_plot)):
            # Plot original image and latents
            self.plot_original_image(ax[plot_idx], img_idx, plot_idx)
            self.plot_latents(ax[1], 0, data['latest_latents'][img_idx].mean(dim=0), plot_idx)

            # Plot heatmaps
            self.config.draw_pure_lift = False
            self.config.base_type = "composed"
            for obj_idx, obj_name in enumerate(['black_car', 'white_clock']):
                heatmap = self.heatmap_gen.generate_heatmap(
                    data,
                    data[f'score_{obj_name}_all'],
                    data['base_to_test'],
                    data.get('noise'),
                    img_idx
                )
                self.plot_heatmap(ax[obj_idx][1], heatmap, f"Eqn.5 for {' '.join([s.capitalize() for s in obj_name.split('_')])}", plot_idx)

            self.config.draw_pure_lift = True
            self.config.base_type = "noise"
            for obj_idx, obj_name in enumerate(['black_car', 'white_clock']):
                heatmap = self.heatmap_gen.generate_heatmap(
                    data,
                    data[f'score_{obj_name}_all'],
                    data['base_to_test'],
                    data.get('noise'),
                    img_idx
                )
                self.plot_heatmap(ax[obj_idx][2], heatmap, f"Eqn.3 for {' '.join([s.capitalize() for s in obj_name.split('_')])}", plot_idx)

            # Plot trajectories if enabled
            if self.config.draw_trajectories:
                for traj_idx, (obj1, obj2) in enumerate([('black_car', 'white_clock')]):
                    trajectory = self.trajectory_plotter.calculate_trajectory(
                        data['base_to_test'],
                        data[f'score_{obj1}_all'],
                        data[f'score_{obj2}_all'],
                        img_idx
                    )
                    self.plot_trajectory(ax[plot_idx][traj_idx + 6], trajectory)

        plt.tight_layout()
        plt.savefig("variance_reduction.pdf", bbox_inches='tight')

# Usage example:
config = VisualizationConfig(
    specific_indices=[9],
)

noise = callback.get_noise(list(latest_latents.shape[1:]))[None].repeat(score_composed_all.shape[0], 1, 1, 1, 1)
base = config.base_type
# Determine base tensor to test
if base == "composed":
    base_to_test = score_composed_all
elif base == "noise":
    base_to_test = noise
elif base == "uncond":
    base_to_test = score_uncond_all

# Prepare data dictionary
data = {
    'latest_latents': latest_latents,
    'score_composed_all': score_composed_all,
    'score_black_car_all': score_black_car_all,
    'score_white_clock_all': score_white_clock_all,
    'score_uncond_all': score_uncond_all,
    'base_to_test': base_to_test,
    'noise': noise,  # if base == "noise" else None,
}

# Create and run visualization
visualizer = VisualizationPlotter(config)
visualizer.create_visualization(data)
