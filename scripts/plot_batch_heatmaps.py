from tqdm import tqdm
import gc
from torchvision.transforms import GaussianBlur
import torch
from scipy import ndimage
# run the metrics for original data
import os
import json
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


# original_folder = "outputs/weak_lift"
original_folder = "outputs/standard_sd_xl_lift"
# original_folder = "outputs/standard_sd_2_1_lift"
# original_folder = "outputs/standard_sd_1_4_lift"


# run the clip metrics
os.system(f"python metrics/compute_clip_similarity.py --output_path={original_folder} --metrics_save_path={original_folder} > /dev/null 2>&1")
# load the metrics results
with open(f"{original_folder}/clip_raw_metrics.json", "r") as f:
    raw_metrics_results = json.load(f)

# create a df, each row is a sample, each column includes: prompt, image_name, clip_score
import pandas as pd
# Create a list of dictionaries for all rows
data = [
    {
        "prompt": prompt,
        "image_name": image_name,
        "clip_score": full_text_score,
        "first_half_text_score": first_half_text_score,
        "second_half_text_score": second_half_text_score
    }
    for prompt in raw_metrics_results.keys()
    for image_name, full_text_score, first_half_text_score, second_half_text_score in zip(
        raw_metrics_results[prompt]['image_names'],
        raw_metrics_results[prompt]['full_text'],
        raw_metrics_results[prompt]['first_half'],
        raw_metrics_results[prompt]['second_half']
    )
]

# Create DataFrame from the list of dictionaries
df = pd.DataFrame(data)
df = df[~df['image_name'].str.contains("heatmap")]

def apply_median_filter(tensor, kernel_size=3):
    """
    Apply median filter to a PyTorch tensor

    Parameters:
    - tensor: Input PyTorch tensor
    - kernel_size: Size of the median filter kernel (default: 3)

    Returns:
    - Filtered tensor
    """
    # Store original device
    device = tensor.device

    # Convert to numpy array, ensuring it's float32
    if tensor.dim() == 2:
        numpy_array = tensor.detach().cpu().float().numpy()
    else:
        numpy_array = tensor.detach().cpu().float().squeeze().numpy()

    # Ensure the array is contiguous and in the correct format
    numpy_array = np.ascontiguousarray(numpy_array)

    # Apply median filter
    filtered = ndimage.median_filter(numpy_array, size=kernel_size)

    # Convert back to torch tensor
    return torch.from_numpy(filtered).float().to(device)


def plot_heatmaps(lift_object_1, lift_object_2, prompt, seed, filename,threshold=2e-4, pixels_threshold=250):
    plt.clf()
    plt.close("all")

    # Create figure with 1 row and 3 columns
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Load and display original image
    image = Image.open(f"{original_folder}/{prompt}/{seed}.png")
    ax[0].imshow(image)
    clip_score = df.loc[(df['prompt'] == prompt) & (df['image_name'] == f"{seed}.png"), 'clip_score'].values[0]
    ax[0].set_title(f"Image (1024x1024)\nCLIP Score: {clip_score:.4f}", fontsize=12)

    if "and" in prompt:
        name_object_1 = prompt.split("and")[0]
        name_object_2 = prompt.split("and")[1]
    else:
        assert "with" in prompt
        name_object_1 = prompt.split("with")[0]
        name_object_2 = prompt.split("with")[1]

    # Plot heatmaps for each object
    for idx, (heatmap, title) in enumerate([(lift_object_1, name_object_1), (lift_object_2, name_object_2)]):
        heatmap = heatmap - threshold

        if heatmap.relu().cpu().numpy().max() > 1e-8:
            im = ax[idx+1].imshow(heatmap.relu().cpu().numpy(), cmap="turbo", interpolation='nearest')
        else:
            im = ax[idx+1].imshow(heatmap.relu().cpu().numpy(), cmap="turbo", vmin=0, vmax=1e-8, interpolation='nearest')

        num_of_activated_pixels = (heatmap > 0).sum().item()
        decision = "accept" if num_of_activated_pixels > pixels_threshold else "reject"
        title = f"{title}\n#Activated Pixels: {num_of_activated_pixels}\nDecision: {decision}"
        # Add colorbar
        divider = make_axes_locatable(ax[idx+1])
        cax = divider.append_axes("right", size="4%", pad=0.2)

        if heatmap.relu().cpu().numpy().max() <= 1e-8:
            cbar = plt.colorbar(im, cax=cax, ticks=[0])
        else:
            cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.yaxis.get_offset_text().set_fontsize(10)

        ax[idx+1].set_title(f"Lift for {title}", fontsize=12)

    # Remove ticks
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
        a.set_aspect('auto')

    plt.tight_layout()

    # Save the figure
    output_dir = f"{original_folder}/{prompt}/heatmaps"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/heatmap_{seed}{filename}.png", bbox_inches='tight')
    plt.close()


def bootstrap_heatmap(heatmap, num_of_timesteps, num_of_bootstraps):
    bootstrap_heatmaps = []
    for _ in range(num_of_bootstraps):
        bootstrap_heatmap = heatmap[:, torch.randint(0, num_of_timesteps, (num_of_timesteps,))]
        # -> (1, 200, 4, 128, 128)
        bootstrap_heatmap = bootstrap_heatmap.mean(dim=(1, 2))
        # -> (1, 128, 128)
        bootstrap_heatmaps.append(bootstrap_heatmap)
    return torch.cat(bootstrap_heatmaps, dim=0)


# load the score results

thresholds = [0, 9e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3]

for prompt in tqdm(df['prompt'].unique()):
    gc.collect()
    torch.cuda.empty_cache()
    num_of_samples = len(df[df['prompt'] == prompt])

    if "weak_lift" in original_folder:
        score_results_all = torch.load(f"{original_folder}/score_results_{prompt}_same_noise=False.pt", weights_only=True, map_location="cpu")

    for seed in range(5 if "weak_lift" not in original_folder else 100):
        if "weak_lift" in original_folder:
            score_results = score_results_all[seed]
        else:
            score_results = torch.load(f"{original_folder}/{prompt}/{seed}_score_results.pt", weights_only=True, map_location="cpu")

        score_results = score_results.float()
        if score_results.shape[1] > 3:
            score_composed = score_results[:, 0]
            score_object_1 = score_results[:, 1]
            score_object_2 = score_results[:, 2]
            score_uncond = score_results[:, 3]

            lift_object_1 = ((score_uncond - score_composed).pow(2) - (score_object_1 - score_composed).pow(2))
            lift_object_2 = ((score_uncond - score_composed).pow(2) - (score_object_2 - score_composed).pow(2))

            # lift_object_1_bootstrap = bootstrap_heatmap(lift_object_1, 200, 1000)
            # lift_object_2_bootstrap = bootstrap_heatmap(lift_object_2, 200, 1000)

            # %95 confidence interval
            lift_object_1 = lift_object_1.mean(dim=(0, 1, 2))
            lift_object_2 = lift_object_2.mean(dim=(0, 1, 2))

            # Gaussian blur
            # lift_object_1 = GaussianBlur(kernel_size=3)(lift_object_1.float()[None])[0]
            # lift_object_2 = GaussianBlur(kernel_size=3)(lift_object_2.float()[None])[0]

            plot_heatmaps(lift_object_1, lift_object_2, prompt, seed, filename="", threshold=2e-4)

            lift_old_object_1 = ((score_object_2 - score_composed).pow(2) - (score_object_1 - score_composed).pow(2)).mean(dim=(0, 1, 2))
            lift_old_object_2 = ((score_object_1 - score_composed).pow(2) - (score_object_2 - score_composed).pow(2)).mean(dim=(0, 1, 2))

            plot_heatmaps(lift_old_object_1, lift_old_object_2, prompt, seed, filename="_old", threshold=2e-4)

        else:
            lift_object_1 = torch.ones_like(score_results[:, 0]).mean(dim=(1, 2))
            lift_object_2 = torch.ones_like(score_results[:, 0]).mean(dim=(1, 2))

        # lift_object_1 = GaussianBlur(kernel_size=3)(lift_object_1.float()[None])[0]
        # lift_object_2 = GaussianBlur(kernel_size=3)(lift_object_2.float()[None])[0]

        df.loc[(df['prompt'] == prompt) & (df['image_name'] == f"{seed}.png"), 'maximum_lift_object_1'] = lift_object_1.max()
        df.loc[(df['prompt'] == prompt) & (df['image_name'] == f"{seed}.png"), 'maximum_lift_object_2'] = lift_object_2.max()

        for threshold in thresholds:
            threshold_str = f"{threshold:.0e}"

            activated_pixels_for_object_1 = (lift_object_1 > threshold).sum()
            activated_pixels_for_object_2 = (lift_object_2 > threshold).sum()

            activated_pixels_for_object_1_old = (lift_old_object_1 > threshold).sum()
            activated_pixels_for_object_2_old = (lift_old_object_2 > threshold).sum()

            df.loc[(df['prompt'] == prompt) & (df['image_name'] == f"{seed}.png"), f'activated_pixels_for_object_1_{threshold_str}'] = activated_pixels_for_object_1
            df.loc[(df['prompt'] == prompt) & (df['image_name'] == f"{seed}.png"), f'activated_pixels_for_object_2_{threshold_str}'] = activated_pixels_for_object_2

            df.loc[(df['prompt'] == prompt) & (df['image_name'] == f"{seed}.png"), f'activated_pixels_for_object_1_old_{threshold_str}'] = activated_pixels_for_object_1_old
            df.loc[(df['prompt'] == prompt) & (df['image_name'] == f"{seed}.png"), f'activated_pixels_for_object_2_old_{threshold_str}'] = activated_pixels_for_object_2_old

df.to_csv(f"{original_folder}/metrics.csv", index=False)
