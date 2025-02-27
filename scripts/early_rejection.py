import pandas as pd
import numpy as np
from collections import defaultdict
import json
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


def analyze_early_stop_metrics(original_folder):
    """
    Analyze early stop metrics from a CSV file and return the best settings for each prompt class and timestep.
    """
    # Debug: Print file path and verify file exists
    csv_path = f'{original_folder}/early_stop_metrics.csv'
    metrics_path = f'{original_folder}/metrics.csv'
    print(f"Reading CSV from: {csv_path}")
    print(f"Reading metrics from: {metrics_path}")

    # Read both CSVs
    df = pd.read_csv(csv_path)
    metrics_df = pd.read_csv(metrics_path)

    # Merge ImageReward scores from metrics.csv into early_stop_metrics.csv
    df = df.merge(
        metrics_df[['prompt', 'image_name', 'image_reward_score']],
        on=['prompt', 'image_name'],
        how='left'
    )

    print(f"CSV loaded with {len(df)} rows")
    print(f"ImageReward scores available for {df['image_reward_score'].notna().sum()} rows")

    thresholds = [2e-4] #[0, 9e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3]
    pixel_thresholds = [250] #[1, 10, 25, 50, 100, 125, 150, 175, 200, 225, 250, 300, 500, 1200, 2000, 3000, 16384]

    # Get all timesteps from column names - modified pattern matching
    timestep_columns = [col for col in df.columns if 'activated_pixels_for_object_1_cached_' in col]
    print(f"\nFound timestep columns: {timestep_columns[:5]}...")  # Show first 5 for brevity

    # Extract timesteps using the new pattern
    timesteps = []
    for col in timestep_columns:
        try:
            # Get the last part and convert to integer
            timestamp = col.split('_')[-1]
            timesteps.append(timestamp)
        except ValueError:
            continue

    timesteps = sorted(list(set(timesteps)), key=lambda x: float(x))
    print(f"Extracted timesteps: {timesteps[:10]}...")  # Show first 10 for brevity

    # Read prompt classes
    prompt_classes_path = "a.e_prompts.txt"
    print(f"\nReading prompt classes from: {prompt_classes_path}")

    with open(prompt_classes_path, "r") as f:
        prompt_classes = json.load(f)

    print(f"Loaded prompt classes: {list(prompt_classes.keys())}")

    results = {}
    best_settings = {}

    for prompt_class in prompt_classes.keys():
        print(f"\nProcessing prompt class: {prompt_class}")
        print(f"Number of prompts in class: {len(prompt_classes[prompt_class])}")

        best_settings[prompt_class] = {}

        for t in timesteps:
            metrics = []

            for threshold in tqdm(thresholds, desc=f"Processing {prompt_class} at t={t}"):
                threshold_str = f"{threshold:.0e}"

                for pixel_threshold in pixel_thresholds:
                    n_wins = 0
                    n_comparisons = 0
                    sum_filtered_clip_scores = []
                    sum_filtered_ir_scores = []
                    sum_all_clip_scores = []
                    sum_all_ir_scores = []
                    n_reject = 0

                    for prompt in prompt_classes[prompt_class]:
                        sub_df = df[df['prompt'] == prompt]

                        if len(sub_df) == 0:
                            print(f"Warning: No data found for prompt: {prompt}")
                            continue

                        # Get column names for this timestep
                        col1 = f'activated_pixels_for_object_1_cached_{threshold_str}_{t}'
                        col2 = f'activated_pixels_for_object_2_cached_{threshold_str}_{t}'

                        if col1 not in sub_df.columns or col2 not in sub_df.columns:
                            print(f"Warning: Columns {col1} or {col2} not found")
                            continue

                        # Count rejections
                        k = (sub_df[[col1, col2]].min(axis=1) >= pixel_threshold).sum()

                        if (k > 0) and (k < len(sub_df)):
                            n_reject += (len(sub_df) - k)

                            filter_condition = sub_df[[col1, col2]].min(axis=1) >= pixel_threshold

                            avg_clip_score_filtered = sub_df[filter_condition]['clip_score'].mean()
                            avg_clip_score_all = sub_df['clip_score'].mean()
                            avg_ir_score_filtered = sub_df[filter_condition]['image_reward_score'].mean()
                            avg_ir_score_all = sub_df['image_reward_score'].mean()

                            n_wins += (avg_clip_score_filtered > avg_clip_score_all)
                            n_comparisons += 1

                            sum_filtered_clip_scores.append(avg_clip_score_filtered)
                            sum_all_clip_scores.append(avg_clip_score_all)
                            sum_filtered_ir_scores.append(avg_ir_score_filtered)
                            sum_all_ir_scores.append(avg_ir_score_all)

                        else:
                            avg_clip_score_filtered = sub_df['clip_score'].mean()
                            avg_clip_score_all = sub_df['clip_score'].mean()
                            avg_ir_score_filtered = sub_df['image_reward_score'].mean()
                            avg_ir_score_all = sub_df['image_reward_score'].mean()

                            sum_filtered_clip_scores.append(avg_clip_score_filtered)
                            sum_all_clip_scores.append(avg_clip_score_all)
                            sum_filtered_ir_scores.append(avg_ir_score_filtered)
                            sum_all_ir_scores.append(avg_ir_score_all)

                    if n_comparisons > 0:
                        metrics.append({
                            'threshold': threshold_str,
                            'pixel_threshold': pixel_threshold,
                            'clip_ratio': np.mean(sum_filtered_clip_scores) / np.mean(sum_all_clip_scores),
                            'ir_ratio': np.mean(sum_filtered_ir_scores) - np.mean(sum_all_ir_scores),
                            'mean_filtered_clip': np.mean(sum_filtered_clip_scores),
                            'mean_all_clip': np.mean(sum_all_clip_scores),
                            'mean_filtered_ir': np.mean(sum_filtered_ir_scores),
                            'mean_all_ir': np.mean(sum_all_ir_scores),
                            'n_reject_ratio': n_reject / (len(sub_df) * len(prompt_classes[prompt_class])),
                            'n_wins': n_wins,
                            'n_comparisons': n_comparisons,
                            'win_rate': n_wins / n_comparisons if n_comparisons > 0 else 0
                        })

            if metrics:
                # Find best setting based on CLIP ratio
                best_metric = max(metrics, key=lambda x: x['clip_ratio'])
                best_settings[prompt_class][t] = best_metric
            else:
                print(f"Warning: No valid metrics found for {prompt_class} at timestep {t}")

    return best_settings

def format_results(best_settings):
    """Format results in a readable way."""
    formatted_results = {}

    for prompt_class, timestep_results in best_settings.items():
        formatted_results[prompt_class] = {}

        for t, metrics in timestep_results.items():
            formatted_results[prompt_class][t] = {
                'threshold': metrics['threshold'],
                'pixel_threshold': metrics['pixel_threshold'],
                'clip_improvement': f"{(metrics['clip_ratio'] - 1) * 100:.2f}%",
                'ir_improvement': f"{metrics['ir_ratio']:.2f}",
                'rejection_rate': f"{metrics['n_reject_ratio'] * 100:.2f}%",
                'win_rate': f"{metrics['win_rate'] * 100:.2f}%",
                'mean_filtered_clip': f"{metrics['mean_filtered_clip']:.2f}",
                'mean_all_clip': f"{metrics['mean_all_clip']:.2f}",
                'mean_filtered_ir': f"{metrics['mean_filtered_ir']:.2f}",
                'mean_all_ir': f"{metrics['mean_all_ir']:.2f}"
            }

    return formatted_results

if __name__ == "__main__":
    # Example usage for different models
    models = {
        # "SD 1.4": "outputs/standard_sd_1_4_lift",
        "SD 2.1": "outputs/standard_sd_2_1_lift",
        # "SD XL": "outputs/standard_sd_xl_lift"
    }

    for model_name, folder in models.items():
        if os.path.exists(f'{folder}/early_rejection_results.json'):
            formatted_results = json.load(open(f'{folder}/early_rejection_results.json', 'r'))

        else:
            print(f"\nAnalyzing {model_name}...")
            best_settings = analyze_early_stop_metrics(folder)
            formatted_results = format_results(best_settings)

            # Print results
            for prompt_class, timestep_results in formatted_results.items():
                print(f"\n{prompt_class}:")
                for t, metrics in timestep_results.items():
                    print(f"\nTimestep {t}:")
                    print(f"  Threshold: {metrics['threshold']}")
                    print(f"  Pixel Threshold: {metrics['pixel_threshold']}")
                    print(f"  CLIP Improvement: {metrics['clip_improvement']}")
                    print(f"  IR Improvement: {metrics['ir_improvement']}")
                    print(f"  Rejection Rate: {metrics['rejection_rate']}")
                    print(f"  Win Rate: {metrics['win_rate']}")
                    print(f"  Mean Filtered CLIP: {metrics['mean_filtered_clip']}")
                    print(f"  Mean All CLIP: {metrics['mean_all_clip']}")
                    print(f"  Mean Filtered IR: {metrics['mean_filtered_ir']}")
                    print(f"  Mean All IR: {metrics['mean_all_ir']}")

            # save results to json
            with open(f'{folder}/early_rejection_results.json', 'w') as f:
                json.dump(formatted_results, f, indent=4)

        # draw plots - horizontal axis is timesteps, vertical axis is filtered image reward score
        for prompt_class, timestep_results in formatted_results.items():
            plt.figure()
            # sort timestep_results from largest to smallest
            timestep_results = sorted(timestep_results.items(), key=lambda x: float(x[0]), reverse=True)
            ts = [float(t) for t, _ in timestep_results]
            irs = []
            for _, metrics in timestep_results:
                if "mean_filtered_ir" in metrics:
                    irs.append(float(metrics['mean_filtered_ir']))
                else:
                    import pdb; pdb.set_trace()
                    assert False
            plt.plot(ts, irs, label=f'{prompt_class}')
            plt.gca().invert_xaxis()
            plt.legend()
            plt.savefig(f'figures/{model_name.replace(" ", "_")}_{prompt_class}.png')
