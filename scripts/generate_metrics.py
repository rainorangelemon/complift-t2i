# run the metrics for original data
import os
import json
from tqdm import tqdm
import gc
from torchvision.transforms import GaussianBlur
import torch
from copy import deepcopy
import ImageReward as RM
import pandas as pd


def add_image_reward_scores(folder_path):
    """Add ImageReward scores to the metrics.csv file in the given folder."""

    model = RM.load("ImageReward-v1.0")

    # Read the metrics CSV
    csv_path = os.path.join(folder_path, 'metrics.csv')
    df = pd.read_csv(csv_path)

    # Initialize the new column
    df['image_reward_score'] = 0.0

    # Calculate scores for each row
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row['prompt']
        image_path = os.path.join(folder_path, prompt, row['image_name'])

        if os.path.exists(image_path):
            # Calculate ImageReward score
            reward = model.score(prompt, [image_path])
            df.at[idx, 'image_reward_score'] = reward
        else:
            print(f"Image not found: {image_path}")
            assert False

    # Save the updated CSV
    df.to_csv(csv_path, index=False)
    print(f"Updated metrics saved to {csv_path}")


def cummean(x, dim=0):
    """
    Compute cumulative mean along a dimension.

    Args:
        x (torch.Tensor): Input tensor
        dim (int): Dimension along which to compute cumulative mean (default: 0)

    Returns:
        torch.Tensor: Tensor of same shape as input containing cumulative means
    """
    cumsum = torch.cumsum(x, dim=dim)
    arange = torch.arange(1, x.shape[dim] + 1, device=x.device)

    # Reshape arange to match the dimension of cumsum for broadcasting
    shape = [1] * len(x.shape)
    shape[dim] = x.shape[dim]
    arange = arange.view(shape)

    return cumsum / arange


def generate_metrics(original_folder):
    """
    Generate metrics for the given output folder.

    Args:
        original_folder (str): Path to the folder containing the output data
    """
    if not os.path.exists(f"{original_folder}/clip_raw_metrics.json"):
        # run the clip metrics
        os.system(f"python metrics/compute_clip_similarity.py --output_path={original_folder} --metrics_save_path={original_folder} > /dev/null 2>&1")
        # load the metrics results
        with open(f"{original_folder}/clip_raw_metrics.json", "r") as f:
            metrics_results = json.load(f)
    else:
        with open(f"{original_folder}/clip_raw_metrics.json", "r") as f:
            raw_metrics_results = json.load(f)

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
    early_stop_df = pd.DataFrame(data)

    # thresholds for the metrics
    thresholds = [0, 9e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3]

    for prompt in tqdm(df['prompt'].unique()):
        gc.collect()
        torch.cuda.empty_cache()
        num_of_samples = 5 if "weak_lift" not in original_folder else 100

        if "weak_lift" in original_folder:
            score_results_all = torch.load(f"{original_folder}/score_results_{prompt}_same_noise=False.pt", weights_only=True, map_location="cuda")

        for seed in range(num_of_samples):
            if "weak_lift" in original_folder:
                score_results = score_results_all[seed]
            elif "ae" not in original_folder:
                score_results = torch.load(f"{original_folder}/{prompt}/{seed}_score_results.pt", weights_only=True, map_location="cuda")
                cached_score_results = torch.load(f"{original_folder}/{prompt}/{seed}_cached_score_results.pt", weights_only=True, map_location="cuda")
                intermediate_ts = torch.load(f"{original_folder}/{prompt}/{seed}_lift_results.pt", weights_only=True, map_location="cuda")['intermediate_ts']
                intermediate_ts = [t.item() for t in intermediate_ts]
            else:
                continue

            score_results = score_results.float()
            if score_results.shape[1] > 3:
                score_composed = score_results[:, 0]
                score_object_1 = score_results[:, 1]
                score_object_2 = score_results[:, 2]
                score_uncond = score_results[:, 3]

                cached_score_composed = cached_score_results[:, 0]
                cached_score_object_1 = cached_score_results[:, 1]
                cached_score_object_2 = cached_score_results[:, 2]
                cached_score_uncond = cached_score_results[:, 3]

                lift_object_1 = ((score_uncond - score_composed).pow(2) - (score_object_1 - score_composed).pow(2)).mean(dim=(1, 2))
                lift_object_2 = ((score_uncond - score_composed).pow(2) - (score_object_2 - score_composed).pow(2)).mean(dim=(1, 2))
                lift_object_1_old = ((score_object_2 - score_composed).pow(2) - (score_object_1 - score_composed).pow(2)).mean(dim=(1, 2))
                lift_object_2_old = ((score_object_1 - score_composed).pow(2) - (score_object_2 - score_composed).pow(2)).mean(dim=(1, 2))
                lift_object_1_cached = ((cached_score_uncond - cached_score_composed).pow(2) - (cached_score_object_1 - cached_score_composed).pow(2)).mean(dim=(1, 2))
                lift_object_2_cached = ((cached_score_uncond - cached_score_composed).pow(2) - (cached_score_object_2 - cached_score_composed).pow(2)).mean(dim=(1, 2))

                cummean_lift_object_1_cached = ((cached_score_uncond - cached_score_composed).pow(2) - (cached_score_object_1 - cached_score_composed).pow(2)).mean(dim=(0, 2))
                cummean_lift_object_1_cached = cummean(cummean_lift_object_1_cached, dim=0)

                cummean_lift_object_2_cached = ((cached_score_uncond - cached_score_composed).pow(2) - (cached_score_object_2 - cached_score_composed).pow(2)).mean(dim=(0, 2))
                cummean_lift_object_2_cached = cummean(cummean_lift_object_2_cached, dim=0)

            else:
                assert False

            df.loc[(df['prompt'] == prompt) & (df['image_name'] == f"{seed}.png"), 'maximum_lift_object_1'] = lift_object_1.max().item()
            df.loc[(df['prompt'] == prompt) & (df['image_name'] == f"{seed}.png"), 'maximum_lift_object_2'] = lift_object_2.max().item()

            for threshold in thresholds:
                threshold_str = f"{threshold:.0e}"

                activated_pixels_for_object_1 = (lift_object_1 > threshold).sum().item()
                activated_pixels_for_object_2 = (lift_object_2 > threshold).sum().item()
                activated_pixels_for_object_1_old = (lift_object_1_old > threshold).sum().item()
                activated_pixels_for_object_2_old = (lift_object_2_old > threshold).sum().item()
                activated_pixels_for_object_1_cached = (lift_object_1_cached > threshold).sum().item()
                activated_pixels_for_object_2_cached = (lift_object_2_cached > threshold).sum().item()

                df.loc[(df['prompt'] == prompt) & (df['image_name'] == f"{seed}.png"), f'activated_pixels_for_object_1_{threshold_str}'] = activated_pixels_for_object_1
                df.loc[(df['prompt'] == prompt) & (df['image_name'] == f"{seed}.png"), f'activated_pixels_for_object_2_{threshold_str}'] = activated_pixels_for_object_2
                df.loc[(df['prompt'] == prompt) & (df['image_name'] == f"{seed}.png"), f'activated_pixels_for_object_1_old_{threshold_str}'] = activated_pixels_for_object_1_old
                df.loc[(df['prompt'] == prompt) & (df['image_name'] == f"{seed}.png"), f'activated_pixels_for_object_2_old_{threshold_str}'] = activated_pixels_for_object_2_old
                df.loc[(df['prompt'] == prompt) & (df['image_name'] == f"{seed}.png"), f'activated_pixels_for_object_1_cached_{threshold_str}'] = activated_pixels_for_object_1_cached
                df.loc[(df['prompt'] == prompt) & (df['image_name'] == f"{seed}.png"), f'activated_pixels_for_object_2_cached_{threshold_str}'] = activated_pixels_for_object_2_cached

                for i, t in enumerate(intermediate_ts):
                    activated_pixels_for_object_1_cached_t = (cummean_lift_object_1_cached[i] > threshold).sum().item()
                    activated_pixels_for_object_2_cached_t = (cummean_lift_object_2_cached[i] > threshold).sum().item()
                    early_stop_df.loc[(early_stop_df['prompt'] == prompt) & (early_stop_df['image_name'] == f"{seed}.png"), f'activated_pixels_for_object_1_cached_{threshold_str}_{t}'] = activated_pixels_for_object_1_cached_t
                    early_stop_df.loc[(early_stop_df['prompt'] == prompt) & (early_stop_df['image_name'] == f"{seed}.png"), f'activated_pixels_for_object_2_cached_{threshold_str}_{t}'] = activated_pixels_for_object_2_cached_t

    df = df[~df['image_name'].str.contains("heatmap")]
    df.to_csv(f'{original_folder}/metrics.csv', index=False)

    early_stop_df = early_stop_df[~early_stop_df['image_name'].str.contains("heatmap")]
    early_stop_df.to_csv(f'{original_folder}/early_stop_metrics.csv', index=False)

    return df, early_stop_df

if __name__ == "__main__":
    # Example usage
    for folder in ["outputs/standard_sd_1_4",
                   "outputs/standard_sd_2_1",
                   "outputs/standard_sd_xl",
                   ]:
        df, early_stop_df = generate_metrics(folder)
        add_image_reward_scores(folder)
