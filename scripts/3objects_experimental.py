# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %autoreload 2

# %cd ..

GOOGLE_RED = "#F44336"
GOOGLE_BLUE = "#2196F3"
GOOGLE_GREEN = "#4CAF50"
GOOGLE_YELLOW = "#FFC107"

import pandas as pd
import os
import torch
from typing import List, Optional, Dict, Any, Union, Tuple
from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline, DDPMScheduler
from lift_callback import LiftCallback
from config import LiftConfig
from tqdm import tqdm, trange
import matplotlib
from PIL import Image
from torchvision.transforms import GaussianBlur
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dataclasses import dataclass

# enable tf 32
torch.set_float32_matmul_precision('high')

matplotlib.rc('text', usetex = True)
pgf_with_latex = {
    "text.usetex": True,            # use LaTeX to write all text
    "pgf.rcfonts": False,           # Ignore Matplotlibrc
    "pgf.preamble": r"""
        \PassOptionsToPackage{svgnames,dvipsnames,x11names,html}{xcolor}
        \usepackage{xcolor}
        \definecolor{GoogleRed}{HTML}{F44336}
        \definecolor{GoogleBlue}{HTML}{2196F3}
        \definecolor{GoogleYellow}{HTML}{FFC107}
        \definecolor{GoogleGreen}{HTML}{4CAF50}
    """
}
matplotlib.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
plt.rcParams['xtick.bottom'] = False
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False


def clean_object_name(object_name):
    if "a " in object_name:
        return object_name.split("a ")[1]
    else:
        return object_name

prompts2objects = {
    "a turtle and a ball in a box": ["a turtle", "a ball", "a box"],
    "a glass bottle with a message drifting past starfish": ["a glass bottle", "a message", "starfish"],
    "a steaming teacup beside an open book and a candle": ["teacup", "an open book", "a candle"],
    "an old gramophone on a windowsill with falling autumn leaves": ["an old gramophone", "a windowsill", "falling autumn leaves"],
    "a rustic wagon wheel leaning against a barn with sunflowers": ["wagon wheel", "a barn", "sunflowers"],
    "A wooden ladder reaching into a treehouse under stars": ["a wooden ladder", "a treehouse", "stars"],
    "A paper boat sailing in a puddle reflecting clouds": ["a paper boat", "a puddle", "clouds"],
}

prompts2titles = {
    "a turtle and a ball in a box": "a turtle\nand a ball\nin a box",
    "a glass bottle with a message drifting past starfish": "a glass bottle\nwith a message\ndrifting past starfish",
    "a steaming teacup beside an open book and a candle": "a steaming teacup\nbesides an open book\nand a candle",
    "an old gramophone on a windowsill with falling autumn leaves": "an old gramophone\non a windowsill with\nfalling autumn leaves",
    "a rustic wagon wheel leaning against a barn with sunflowers": "a rustic wagon wheel\nleaning against a barn\nwith sunflowers",
    "A wooden ladder reaching into a treehouse under stars": "a wooden ladder\nreaching into \na treehouse\nunder stars",
    "A paper boat sailing in a puddle reflecting clouds": "a paper boat\nsailing in a puddle\nreflecting clouds",
}

def convert_prompt_to_title(prompt):
    object0, object1, object2 = prompts2objects[prompt]
    title = prompts2titles[prompt]
    title = title.replace("\\n", "\\\\")
    title = title.replace(object0, r"\textcolor{GoogleRed}{" + object0 + r"}")
    title = title.replace(object1, r"\textcolor{GoogleYellow}{" + object1 + r"}")
    title = title.replace(object2, r"\textcolor{GoogleGreen}{" + object2 + r"}")
    return rf'{title}'


pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True)
pipeline.to("cuda")
# apply dropout to unet

pipeline.unet = torch.compile(pipeline.unet, mode="max-autotune", fullgraph=True)
# offload vae and text encoder
pipeline.vae.to("cpu");

@torch.inference_mode()
def calculate_score(latents,
                    callback: LiftCallback,
                    pipeline: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
                    prompts: List[str],
                    noise: Optional[torch.Tensor] = None,
                    timesteps: Optional[torch.Tensor] = None,
                    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                    alpha: float = 1.0,
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

            added_cond_kwargs = {"text_embeds": add_text_embeds * alpha, "time_ids": add_time_ids}

            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds * alpha,
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


@dataclass
class VisualizationConfig:
    prompt: str
    use_gaussian_smoothing: bool = False
    draw_pure_lift: bool = False
    draw_trajectories: bool = True
    base_type: str = "composed"
    threshold: float = 1e-5 #1e-4
    specific_indices: Optional[List[int]] = None
    fig_size: Tuple[int, int] = (60, 50)
    output_path: str = "outputs/3objects_lift/a turtle and a ball in a box"

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
        current_mse = (base - current_score).pow(2).mean(dim=(1, 2))

        # Stack all other scores' MSEs and get minimum
        other_mses = torch.stack([
            (base - other_score).pow(2).mean(dim=(1, 2))
            for other_score in other_scores
        ])
        min_other_mse = other_mses.min(dim=0)[0]

        return (min_other_mse - current_mse)

    @torch.inference_mode()
    def generate_heatmap(self, data, score_target: torch.Tensor, base: torch.Tensor,
                        noise: Optional[torch.Tensor] = None, idx: int = 0) -> torch.Tensor:
        if self.config.draw_pure_lift:
            heatmap = self.calculate_pure_lift(score_target, base, noise)[idx]
        else:
                # Get all score tensors except the target one
                other_scores = [
                    # data['score_uncond_all']
                    data[f'score_{obj_name}_all']
                    for obj_name in [clean_object_name(obj_name) for obj_name in prompts2objects[self.config.prompt]]
                    if not torch.allclose(data[f'score_{obj_name}_all'], score_target)
                ]
                heatmap = self.calculate_relative_lift(base, score_target, other_scores)[0]

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
        self.n_rows = len(self.config.specific_indices) if self.config.specific_indices else self.config.n_samples
        self.n_cols = 8 if self.config.draw_trajectories else 5

    def setup_plot(self) -> Tuple[plt.Figure, np.ndarray]:
        plt.clf()
        plt.close("all")
        width = self.config.fig_size[0] * (self.n_cols / 8)
        height = self.config.fig_size[1] * (self.n_rows / 10)

        # Create figure with GridSpec and extra space for labels
        fig = plt.figure(figsize=(width, height))
        gs = fig.add_gridspec(
            self.n_rows,
            self.n_cols,
            hspace=0.18,
            wspace=0.2,
            left=0.4,  # Add more space on the left for ylabel
            right=0.95,  # Reduce right margin
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

            ax[i,1].set_position([ax[i,1].get_position().x0 - 0.01,
                                ax[i,1].get_position().y0,
                                ax[i,1].get_position().width,
                                ax[i,1].get_position().height])

            ax[i,2].set_position([ax[i,2].get_position().x0 - 0.011,
                                ax[i,2].get_position().y0,
                                ax[i,2].get_position().width,
                                ax[i,2].get_position().height])

            ax[i,3].set_position([ax[i,3].get_position().x0,
                                ax[i,3].get_position().y0,
                                ax[i,3].get_position().width,
                                ax[i,3].get_position().height])

            ax[i,4].set_position([ax[i,4].get_position().x0 + 0.011,
                                ax[i,4].get_position().y0,
                                ax[i,4].get_position().width,
                                ax[i,4].get_position().height])

        return fig, ax

    def plot_original_image(self, ax, idx: int, row_idx: int, prompt_text: str):
        # Load and convert image to numpy array
        image = Image.open(f"{self.config.output_path}/{idx}.png")
        image_array = np.array(image)

        # Display image without interpolation to maintain sharpness
        im = ax[0].imshow(image_array, interpolation='nearest')
        title = "Image (1024x1024)"
        ax[0].set_title(title, fontsize=self.title_fontsize)

        # Add y axis label with better positioning
        ax[0].set_ylabel(prompt_text, fontsize=self.title_fontsize, rotation=0)
        # Get the ylabel and adjust its position
        ylabel = ax[0].get_yaxis().get_label()
        ylabel.set_verticalalignment('center')
        # Position the label to the left of the plot without overlapping
        ax[0].yaxis.set_label_coords(-0.52, 0.5)

        # Set aspect to 'auto' to fill the entire subplot
        ax[0].set_aspect('auto')

    def plot_latents(self, ax, latents: torch.Tensor, row_idx: int):
        im = ax[1].imshow(latents.cpu().numpy(), cmap="gray", interpolation='nearest')
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="4%", pad=0.2)
        cbar = plt.colorbar(im, cax=cax, format="%.1f")
        cbar.ax.tick_params(labelsize=self.colorbar_fontsize)
        cbar.ax.yaxis.offsetText.set_fontsize(self.colorbar_fontsize)
        title = "Latent (128x128)"
        ax[1].set_title(title, fontsize=self.title_fontsize)

        # Set aspect to 'auto' to fill the entire subplot
        ax[1].set_aspect('auto')

    def plot_heatmap(self, ax, heatmap: torch.Tensor, title: str, row_idx: int, border_color: str):
        if heatmap.relu().cpu().numpy().max() > 1e-8:
            im = ax.imshow(heatmap.relu().cpu().numpy(), cmap="turbo", interpolation='nearest')
        else:
            im = ax.imshow(heatmap.relu().cpu().numpy(), cmap="turbo", vmin=0, vmax=1e-8, interpolation='nearest')

        # Add colored border to the plot
        for spine in ax.spines.values():
            spine.set_linewidth(10)  # Make border thicker
            spine.set_edgecolor(border_color)  # Set border color - you can change this to any color
            spine.set_visible(True)

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
        display_title = ""
        ax.set_title(f"{display_title}\#Activated Pixels: {(heatmap > self.config.threshold).sum().item()}",
                    fontsize=self.title_fontsize)

        # Set aspect to 'auto' to fill the entire subplot
        ax.set_aspect('auto')

    def plot_trajectory(self, ax, trajectory: torch.Tensor):
        ax.plot(trajectory.cpu().numpy(), marker='o')
        ax.tick_params(axis='both', which='major', labelsize=self.colorbar_fontsize)

    @torch.inference_mode()
    def create_visualization(self, data: dict, list_of_prompt_and_indices: List[Tuple[str, int]]=None):
        fig, ax = self.setup_plot()
        if list_of_prompt_and_indices is None:
            indices_to_plot = self.config.specific_indices if self.config.specific_indices else range(self.config.n_samples)
        else:
            indices_to_plot = range(len(list_of_prompt_and_indices))

        for plot_idx, img_idx in enumerate(tqdm(indices_to_plot)):
            # Plot original image and latents
            if list_of_prompt_and_indices is None:
                prompt_text = convert_prompt_to_title(self.config.prompt)
            else:
                prompt_text = convert_prompt_to_title(list_of_prompt_and_indices[plot_idx][0])
                self.config.prompt = list_of_prompt_and_indices[plot_idx][0]
                self.config.output_path = f"outputs/3objects_lift/{list_of_prompt_and_indices[plot_idx][0]}"
            self.plot_original_image(ax[plot_idx], img_idx if list_of_prompt_and_indices is None else list_of_prompt_and_indices[plot_idx][1], plot_idx, prompt_text)
            self.plot_latents(ax[plot_idx], data['latest_latents'][img_idx].mean(dim=0), plot_idx)

            # Plot heatmaps
            for obj_idx, obj_name in enumerate(prompts2objects[self.config.prompt]):
                obj_name = clean_object_name(obj_name)
                sub_data = {k: v[[img_idx]].to("cuda:0") for k, v in data.items()}
                heatmap = self.heatmap_gen.generate_heatmap(
                    sub_data,
                    sub_data[f'score_{obj_name}_all'],
                    sub_data['base_to_test'],
                    sub_data.get('noise'),
                    img_idx
                )

                border_color = [GOOGLE_RED, GOOGLE_YELLOW, GOOGLE_GREEN][obj_idx]
                self.plot_heatmap(ax[plot_idx][obj_idx + 2], heatmap, f"$\it{{Lift}}$ for {obj_name.capitalize()}", plot_idx, border_color)

            # Plot trajectories if enabled
            if self.config.draw_trajectories:
                raise NotImplementedError("Trajectories not implemented for 3 objects")
                for traj_idx, (obj1, obj2) in enumerate([('turtle', 'ball'), ('ball', 'turtle'), ('box', 'turtle')]):
                    trajectory = self.trajectory_plotter.calculate_trajectory(
                        data['base_to_test'],
                        data[f'score_{obj1}_all'],
                        data[f'score_{obj2}_all'],
                        img_idx
                    )
                    self.plot_trajectory(ax[plot_idx][traj_idx + 5], trajectory)

        plt.tight_layout()
        # Use bbox_inches='tight' and pad_inches to automatically adjust boundaries
        plt.savefig(
            f"{self.config.output_path}/t2i_example.pdf",
            backend='pgf',
            bbox_inches='tight',
            pad_inches=0.5  # Add padding around the figure
        )



def compute_and_cache_scores(latest_latents: torch.Tensor,
                             config: LiftConfig,
                             callback: LiftCallback,
                             pipeline: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
                             target_indices: List[int] = None) -> Dict[str, torch.Tensor]:
    """
    Compute and cache scores for a given prompt with multiple objects.

    Args:
        latest_latents: Tensor of latent representations
        config: Configuration object containing prompt and other settings
        callback: Callback object for the lift operation
        pipeline: The diffusion pipeline to use
        target_indices: List of specific image indices to compute scores for. If None, zeros will be used.

    Returns:
        Dictionary containing all computed scores
    """
    # Parse objects from the prompt
    objects = prompts2objects.get(config.prompt, [])
    if not objects:
        raise ValueError(f"Prompt '{config.prompt}' not found in prompts2objects dictionary")

    # Generate score paths
    scores_path = f"outputs/3objects_lift/{config.prompt}/scores.pt"

    if os.path.exists(scores_path):
        # Load cached scores
        return torch.load(scores_path, weights_only=True)

    # Initialize score containers
    n_images = len(latest_latents)
    score_containers = {
        f"score_{name}_all": []
        for name in ['composed'] + [clean_object_name(obj) for obj in objects] + ['uncond']
    }

    # Prepare prompts list - composed prompt first, then individual objects, then empty string
    prompts = [config.prompt] + objects + [""]

    # Compute scores for each image
    for img_idx in tqdm(range(n_images)):
        if target_indices is None or img_idx not in target_indices:
            # Create zero tensors for non-target indices
            score_shape = (1, len(prompts), config.n_samples, *latest_latents.shape[1:])
            score_results = torch.zeros(score_shape)
        else:
            # Compute actual scores for target indices
            score_results = calculate_score(
                latest_latents[[img_idx]],
                callback,
                pipeline,
                prompts=prompts,
                alpha=1.0,
            )

        # Append results to containers
        for i, name in enumerate(['composed'] + [clean_object_name(obj) for obj in objects] + ['uncond']):
            score_containers[f'score_{name}_all'].append(score_results[:, i])

    # Concatenate all results
    results = {
        key: torch.cat(value, dim=0)
        for key, value in score_containers.items()
    }

    # Cache results
    torch.save(results, scores_path)

    return results


def run_visualization(prompt="a turtle and a ball in a box", target_indices=[2, 3, 7, 8]):
    # Load latents
    all_final_latents = torch.zeros(10, 4, 128, 128)
    for i in range(10):
        lift_results = torch.load(f"outputs/3objects_lift/{prompt}/{i}_lift_results.pt", weights_only=True)
        all_final_latents[i] = lift_results["latents"]
    latest_latents = all_final_latents.to("cuda:0")

    # Compute scores
    config = LiftConfig(
        prompt=prompt,
        subtract_unconditional=False,
        batch_size=8,
        same_noise=False,
        n_samples=2000,
    )
    scores = compute_and_cache_scores(
        latest_latents=latest_latents,
        config=config,
        callback=LiftCallback(config),
        pipeline=pipeline,
        target_indices=target_indices
    )

    # Setup visualization
    vis_config = VisualizationConfig(
        prompt=prompt,
        specific_indices=target_indices,
        draw_trajectories=False,
        output_path=f"outputs/3objects_lift/{prompt}"
    )

    # Prepare data for visualization
    noise = LiftCallback(config).get_noise(list(latest_latents.shape[1:]))[None].repeat(scores["score_composed_all"].shape[0], 1, 1, 1, 1)
    base_to_test = {
        "composed": scores["score_composed_all"],
        "noise": noise,
        "uncond": scores["score_uncond_all"]
    }[vis_config.base_type]

    data = {
        'latest_latents': latest_latents,
        'base_to_test': base_to_test,
        'noise': noise,
        **scores  # Unpack all scores directly into data dict
    }

    # Create and run visualization
    VisualizationPlotter(vis_config).create_visualization(data)



def run_mixed_visualization(list_of_prompt_and_indices: List[Tuple[str, List[int]]]):
    # Load latents
    all_final_latents = torch.zeros(len(list_of_prompt_and_indices), 4, 128, 128)
    score_results = []
    for i in range(len(list_of_prompt_and_indices)):
        prompt, index = list_of_prompt_and_indices[i]
        lift_results = torch.load(f"outputs/3objects_lift/{prompt}/{index}_lift_results.pt", weights_only=True)
        all_final_latents[i] = lift_results["latents"]

        # Compute scores
        config = LiftConfig(
            prompt=prompt,
            subtract_unconditional=False,
            batch_size=8,
            same_noise=False,
            n_samples=2000,
        )
        scores = compute_and_cache_scores(
            latest_latents=None,
            config=config,
            callback=LiftCallback(config),
            pipeline=pipeline,
        )
        score_results.append({k: v[index] for k, v in scores.items()})

    latest_latents = all_final_latents.to("cuda:0")
    all_keys = set([k for s in score_results for k in s.keys()])
    scores = {k: torch.zeros(len(list_of_prompt_and_indices), *score_results[0]["score_composed_all"].shape) for k in all_keys}
    for i in range(len(score_results)):
        for k in score_results[i].keys():
            scores[k][i] = score_results[i][k]

    # Setup visualization
    vis_config = VisualizationConfig(
        prompt=prompt,
        specific_indices=list(range(len(list_of_prompt_and_indices))),
        draw_trajectories=False,
        output_path=f"outputs/3objects_lift/{prompt}"
    )

    # Prepare data for visualization
    noise = LiftCallback(config).get_noise(list(latest_latents.shape[1:]))[None].repeat(scores["score_composed_all"].shape[0], 1, 1, 1, 1)
    base_to_test = {
        "composed": scores["score_composed_all"],
        "noise": noise,
        "uncond": scores["score_uncond_all"]
    }[vis_config.base_type]

    data = {
        'latest_latents': latest_latents,
        'base_to_test': base_to_test,
        'noise': noise,
        **scores  # Unpack all scores directly into data dict
    }

    # Create and run visualization
    VisualizationPlotter(vis_config).create_visualization(data, list_of_prompt_and_indices)


if __name__ == "__main__":
    # for prompt in prompts2objects.keys():
    #     try:
    #         run_visualization(prompt, target_indices=list(range(10)))
    #     except Exception as e:
    #         print(f"Error for prompt {prompt}: {e}")
    #         continue
    run_mixed_visualization([
        ("a glass bottle with a message drifting past starfish", 2),
        ("a steaming teacup beside an open book and a candle", 5),
        ("A wooden ladder reaching into a treehouse under stars", 9),
        ("an old gramophone on a windowsill with falling autumn leaves", 7),
    ])
