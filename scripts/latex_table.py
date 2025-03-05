import numpy as np
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import argparse

def analyze_metrics(original_folder):
    """
    Analyze metrics from a CSV file and return the best settings for each prompt class.
    """
    df = pd.read_csv(f'{original_folder}/metrics.csv')
    thresholds = [0, 9e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3]
    pixel_thresholds = [1, 10, 25, 50, 100, 125, 150, 175, 200, 225, 250, 300, 500, 1200, 2000, 3000, 16384]

    # Read prompt classes
    with open("a.e_prompts.txt", "r") as f:
        prompt_classes = json.load(f)

    metrics = defaultdict(list)
    metrics_cached = defaultdict(list)  # Add metrics for cached version
    results = {}

    for threshold in tqdm(thresholds):
        threshold_str = f"{threshold:.0e}"
        for prompt_class in prompt_classes.keys():
            for pixel_threshold in pixel_thresholds:
                # Initialize metrics for both regular and cached
                n_wins = n_wins_cached = 0
                n_comparisons = n_comparisons_cached = 0
                n_reject = n_reject_cached = 0
                sum_filtered_clip_scores = []
                sum_filtered_clip_scores_cached = []
                sum_all_clip_scores = []
                sum_filtered_image_reward_scores = []
                sum_filtered_image_reward_scores_cached = []
                sum_all_image_reward_scores = []

                for prompt in prompt_classes[prompt_class]:
                    sub_df = df[df['prompt'] == prompt]

                    # Regular Lift processing
                    k = (sub_df[[f'activated_pixels_for_object_1_{threshold_str}',
                               f'activated_pixels_for_object_2_{threshold_str}']].min(axis=1) >= pixel_threshold).sum()
                    n_reject += (len(sub_df) - k)

                    # Cached Lift processing
                    k_cached = (sub_df[[f'activated_pixels_for_object_1_cached_{threshold_str}',
                                      f'activated_pixels_for_object_2_cached_{threshold_str}']].min(axis=1) >= pixel_threshold).sum()
                    n_reject_cached += (len(sub_df) - k_cached)

                    if (k > 0) and (k < len(sub_df)):
                        # Regular Lift filter condition
                        filter_condition = sub_df[[f'activated_pixels_for_object_1_{threshold_str}',
                                                 f'activated_pixels_for_object_2_{threshold_str}']].min(axis=1) >= pixel_threshold

                        # Calculate averages for filtered samples
                        avg_clip_score_filtered = sub_df[filter_condition]['clip_score'].mean()
                        avg_ir_score_filtered = sub_df[filter_condition]['image_reward_score'].mean()

                        n_wins += (avg_clip_score_filtered > avg_clip_score_all)
                        n_comparisons += 1

                        sum_filtered_clip_scores.append(avg_clip_score_filtered)
                        sum_filtered_image_reward_scores.append(avg_ir_score_filtered)

                    else:
                        avg_clip_score_filtered = sub_df['clip_score'].mean()
                        avg_ir_score_filtered = sub_df['image_reward_score'].mean()

                        sum_filtered_clip_scores.append(avg_clip_score_filtered)
                        sum_filtered_image_reward_scores.append(avg_ir_score_filtered)

                    # Process cached version
                    if (k_cached > 0) and (k_cached < len(sub_df)):
                        # Cached Lift filter condition
                        filter_condition_cached = sub_df[[f'activated_pixels_for_object_1_cached_{threshold_str}',
                                                        f'activated_pixels_for_object_2_cached_{threshold_str}']].min(axis=1) >= pixel_threshold

                        # Calculate averages for cached filtered samples
                        avg_clip_score_filtered_cached = sub_df[filter_condition_cached]['clip_score'].mean()
                        avg_ir_score_filtered_cached = sub_df[filter_condition_cached]['image_reward_score'].mean()

                        n_wins_cached += (avg_clip_score_filtered_cached > avg_clip_score_all)
                        n_comparisons_cached += 1

                        sum_filtered_clip_scores_cached.append(avg_clip_score_filtered_cached)
                        sum_filtered_image_reward_scores_cached.append(avg_ir_score_filtered_cached)

                    else:
                        avg_clip_score = sub_df['clip_score'].mean()
                        avg_ir_score = sub_df['image_reward_score'].mean()

                        sum_filtered_clip_scores_cached.append(avg_clip_score)
                        sum_filtered_image_reward_scores_cached.append(avg_ir_score)

                    avg_clip_score_all = sub_df['clip_score'].mean()
                    avg_ir_score_all = sub_df['image_reward_score'].mean()
                    sum_all_clip_scores.append(avg_clip_score_all)
                    sum_all_image_reward_scores.append(avg_ir_score_all)

                if n_comparisons:
                    # Store regular Lift metrics
                    metrics[prompt_class].append((
                        np.mean(sum_filtered_clip_scores) / np.mean(sum_all_clip_scores),
                        threshold_str,
                        pixel_threshold
                    ))
                    results[prompt_class, threshold_str, pixel_threshold] = {
                        "mean_filtered_clip_score": np.mean(sum_filtered_clip_scores),
                        "mean_all_clip_score": np.mean(sum_all_clip_scores),
                        "mean_filtered_image_reward_score": np.mean(sum_filtered_image_reward_scores),
                        "mean_all_image_reward_score": np.mean(sum_all_image_reward_scores),
                        "n_reject": n_reject / (len(sub_df) * len(prompt_classes[prompt_class])),
                        "n_wins": n_wins,
                        "n_comparisons": n_comparisons,
                        "n_wins_ratio": n_wins / n_comparisons,
                        "n_comparisons_ratio": n_comparisons / len(prompt_classes[prompt_class])
                    }

                if n_comparisons_cached:
                    # Store cached Lift metrics
                    metrics_cached[prompt_class].append((
                        np.mean(sum_filtered_clip_scores_cached) / np.mean(sum_all_clip_scores),
                        threshold_str,
                        pixel_threshold
                    ))
                    results[f"{prompt_class}_cached", threshold_str, pixel_threshold] = {
                        "mean_filtered_clip_score": np.mean(sum_filtered_clip_scores_cached),
                        "mean_all_clip_score": np.mean(sum_all_clip_scores),
                        "mean_filtered_image_reward_score": np.mean(sum_filtered_image_reward_scores_cached),
                        "mean_all_image_reward_score": np.mean(sum_all_image_reward_scores),
                        "n_reject": n_reject_cached / (len(sub_df) * len(prompt_classes[prompt_class])),
                        "n_wins": n_wins_cached,
                        "n_comparisons": n_comparisons_cached,
                        "n_wins_ratio": n_wins_cached / n_comparisons_cached,
                        "n_comparisons_ratio": n_comparisons_cached / len(prompt_classes[prompt_class])
                    }

    # Prepare return dictionary
    best_settings = {}
    for prompt_class in prompt_classes.keys():
        # Regular Lift
        best_metric = max(metrics[prompt_class], key=lambda x: x[0])
        best_settings[prompt_class] = {
            'best_metric': best_metric,
            'detailed_results': results[prompt_class, best_metric[1], best_metric[2]]
        }

        # Cached Lift
        best_metric_cached = max(metrics_cached[prompt_class], key=lambda x: x[0])
        best_settings[f"{prompt_class}_cached"] = {
            'best_metric': best_metric_cached,
            'detailed_results': results[f"{prompt_class}_cached", best_metric_cached[1], best_metric_cached[2]]
        }

        # Print the best threshold and pixel threshold for each method
        print(f"Best threshold and pixel threshold for {prompt_class}:")
        print(f"Threshold: {best_settings[prompt_class]['best_metric'][1]}")
        print(f"Pixel threshold: {best_settings[prompt_class]['best_metric'][2]}")
        print("for cached:")
        print(f"Threshold: {best_settings[f'{prompt_class}_cached']['best_metric'][1]}")
        print(f"Pixel threshold: {best_settings[f'{prompt_class}_cached']['best_metric'][2]}")

    return best_settings


def format_value(value, decimal_places=3):
    """Format a numeric value with specified decimal places."""
    try:
        val = float(value)
        if np.isnan(val):
            return '-'
        return f"{val:.{decimal_places}f}"
    except:
        return '-'

def process_metrics(folder_path):
    """Process metrics for both base and Lift variants from a single Lift analysis."""
    best_settings = analyze_metrics(folder_path)

    base_results = {}
    lift_results = {}
    cached_lift_results = {}  # Add cached results

    # Map the metric keys to the categories in our table
    category_map = {
        'Animals': 'animals',
        'Object&Animal': 'animals_objects',
        'Objects': 'objects'
    }

    # Process each display category
    for display_category, metric_key in category_map.items():
        if metric_key in best_settings:
            # Get both regular and cached settings
            settings = best_settings[metric_key]
            cached_settings = best_settings[f"{metric_key}_cached"]  # Get cached settings

            # Base model results
            base_results[display_category] = {
                'clip': settings['detailed_results']['mean_all_clip_score'],
                'win_rate': 100 - settings['detailed_results']['n_wins_ratio'] * 100,
                'image_reward': settings['detailed_results']['mean_all_image_reward_score']
            }

            # Lift model results
            lift_results[display_category] = {
                'clip': settings['detailed_results']['mean_filtered_clip_score'],
                'win_rate': settings['detailed_results']['n_wins_ratio'] * 100,
                'image_reward': settings['detailed_results']['mean_filtered_image_reward_score']
            }

            # Cached Lift results
            cached_lift_results[display_category] = {
                'clip': cached_settings['detailed_results']['mean_filtered_clip_score'],
                'win_rate': cached_settings['detailed_results']['n_wins_ratio'] * 100,
                'image_reward': cached_settings['detailed_results']['mean_filtered_image_reward_score']
            }

    return base_results, lift_results, cached_lift_results

def generate_comparison_table(lift_folders, show_win=False):
    """
    Generate a LaTeX table comparing different model versions and their metrics.

    Args:
        lift_folders: Dictionary mapping Lift model names to their output folders
        show_win: Boolean to control whether to show win rate columns
    """
    # Process metrics for all models
    all_metrics = {}
    for lift_model, folder in lift_folders.items():
        # Get metrics for this folder
        base_metrics, lift_metrics, cached_metrics = process_metrics(folder)

        # Store metrics with correct model names
        if "SD 1.4" in lift_model:
            base_name = "Stable Diffusion 1.4"
            cached_name = "SD 1.4 + Cached Rejection w/ {\\em Lift}"
        elif "SD 2.1" in lift_model:
            base_name = "Stable Diffusion 2.1"
            cached_name = "SD 2.1 + Cached Rejection w/ {\\em Lift}"
        elif "SD XL" in lift_model:
            base_name = "Stable Diffusion XL"
            cached_name = "SD XL + Cached Rejection w/ {\\em Lift}"
        else:
            base_name = lift_model.replace(" + Rejection w/ {\\em Lift}", "")
            cached_name = lift_model.replace("Rejection w/ {\\em Lift}", "Cached Rejection w/ {\\\em Lift}")

        all_metrics[base_name] = base_metrics
        all_metrics[lift_model] = lift_metrics
        all_metrics[cached_name] = cached_metrics  # Store cached metrics with the correct name

    latex = []

    # Begin table
    latex.append(r"\begin{table}[!h]")
    latex.append(r"\small\centering")
    latex.append(r"\setlength{\tabcolsep}{3pt}")
    latex.append(r"\resizebox{\linewidth}{!}{")
    latex.append(r"\begin{tabular}{lccccccccc}")
    latex.append(r"\toprule")

    # Headers
    if show_win:
        latex.append(r"\bf Method & \multicolumn{3}{c}{\bf Animals} & " +
                    r"\multicolumn{3}{c}{\bf Object\&Animal} & " +
                    r"\multicolumn{3}{c}{\bf Objects} \\")
        latex.append(r" & CLIP $\uparrow$ & IR $\uparrow$ & Win Rate $\uparrow$ & " +
                    r"CLIP $\uparrow$ & IR $\uparrow$ & Win Rate $\uparrow$ & " +
                    r"CLIP $\uparrow$ & IR $\uparrow$ & Win Rate $\uparrow$ \\")
    else:
        latex.append(r"\bf Method & \multicolumn{2}{c}{\bf Animals} & " +
                    r"\multicolumn{2}{c}{\bf Object\&Animal} & " +
                    r"\multicolumn{2}{c}{\bf Objects} \\")
        latex.append(r" & CLIP $\uparrow$ & IR $\uparrow$ & " +
                    r"CLIP $\uparrow$ & IR $\uparrow$ & " +
                    r"CLIP $\uparrow$ & IR $\uparrow$ \\")

    latex.append(r"\midrule")

    # Model groups and their names
    model_groups = [
        ("Stable Diffusion 1.4", "SD 1.4 + Rejection w/ {\\em Lift}", "SD 1.4 + Cached Rejection w/ {\\em Lift}"),
        ("Stable Diffusion 2.1", "SD 2.1 + Rejection w/ {\\em Lift}", "SD 2.1 + Cached Rejection w/ {\\em Lift}"),
        ("Stable Diffusion XL", "SD XL + Rejection w/ {\\em Lift}", "SD XL + Cached Rejection w/ {\\em Lift}")
    ]

    # Categories to process
    categories = ['Animals', 'Object&Animal', 'Objects']

    # Add data rows
    for i, (base_model, lift_model, cached_lift_model) in enumerate(model_groups):
        if i > 0:
            latex.append(r"\midrule")

        # Process all three variants
        base_metrics = all_metrics[base_model]
        lift_metrics = all_metrics[lift_model]
        cached_metrics = all_metrics[cached_lift_model]

        # Create rows
        base_row = [base_model]
        lift_row = [lift_model]
        cached_row = [cached_lift_model]

        # Add metrics for each category
        for category in categories:
            base_clip = format_value(base_metrics[category]['clip'])
            base_ir = format_value(base_metrics[category]['image_reward'])
            lift_clip = format_value(lift_metrics[category]['clip'])
            lift_ir = format_value(lift_metrics[category]['image_reward'])
            cached_clip = format_value(cached_metrics[category]['clip'])
            cached_ir = format_value(cached_metrics[category]['image_reward'])

            # First add the values to rows
            if show_win:
                base_win = format_value(base_metrics[category]['win_rate'], 1)
                lift_win = format_value(lift_metrics[category]['win_rate'], 1)
                cached_win = format_value(cached_metrics[category]['win_rate'], 1)
                base_row.extend([base_clip, base_ir, base_win])
                lift_row.extend([lift_clip, lift_ir, lift_win])
                cached_row.extend([cached_clip, cached_ir, cached_win])
            else:
                base_row.extend([base_clip, base_ir])
                lift_row.extend([lift_clip, lift_ir])
                cached_row.extend([cached_clip, cached_ir])

            # Then apply bold formatting to better values
            current_index = len(base_row) - (3 if show_win else 2)
            if float(lift_clip) >= max(float(base_clip), float(cached_clip)):
                lift_row[current_index] = rf"{{\bf {lift_clip}}}"
            if float(lift_ir) >= max(float(base_ir), float(cached_ir)):
                lift_row[current_index + 1] = rf"{{\bf {lift_ir}}}"
            if show_win and float(lift_win) >= max(float(base_win), float(cached_win)):
                lift_row[current_index + 2] = rf"{{\bf {lift_win}}}"
            if float(cached_clip) >= max(float(base_clip), float(lift_clip)):
                cached_row[current_index] = rf"{{\bf {cached_clip}}}"
            if float(cached_ir) >= max(float(base_ir), float(lift_ir)):
                cached_row[current_index + 1] = rf"{{\bf {cached_ir}}}"
            if show_win and float(cached_win) >= max(float(base_win), float(lift_win)):
                cached_row[current_index + 2] = rf"{{\bf {cached_win}}}"

        # Add rows to latex
        latex.append(" & ".join(base_row) + r" \\")
        latex.append(r"\rowcolor{gray!10} " + " & ".join(cached_row) + r" \\")
        latex.append(r"\rowcolor{gray!20} " + " & ".join(lift_row) + r" \\")

    # Close table
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"}")
    latex.append(r"\vspace{-6pt}")
    latex.append(r"\caption{\small \label{tab:t2i_comp} Quantitative results on Attend-and-Excite Benchmarks. IR represents the ImageReward score \cite{xu2024imagereward}.}")
    latex.append(r"\end{table}")

    return "\n".join(latex)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--win', action='store_true', help='Include win rate columns in the table')
    args = parser.parse_args()

    # Define folders for Lift models only
    lift_folders = {
        "SD 1.4 + Rejection w/ {\\em Lift}": "outputs/standard_sd_1_4",
        "SD 2.1 + Rejection w/ {\\em Lift}": "outputs/standard_sd_2_1",
        "SD XL + Rejection w/ {\\em Lift}": "outputs/standard_sd_xl"
    }

    latex_table = generate_comparison_table(lift_folders, show_win=args.win)
    print(latex_table)

    # Optionally save to file
    with open("model_comparison.tex", "w") as f:
        f.write(latex_table)
