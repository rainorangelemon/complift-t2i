GOOGLE_RED = "#F44336"
GOOGLE_BLUE = "#2196F3"
GOOGLE_GREEN = "#4CAF50"
GOOGLE_YELLOW = "#FFC107"
GOOGLE_GREY = "#9E9E9E"
GOOGLE_LIGHT_GREY = "#E0E0E0"
GOOGLE_TILTED_GREY = "#BDBDBD"

# read df
original_folder = "outputs/weak_lift"

import pandas as pd
df = pd.read_csv(f'{original_folder}/metrics.csv')
df = df[~df['image_name'].str.contains("heatmap")]
thresholds = [0, 9e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3]

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rc('text', usetex = True)

threshold = 2e-4
str_activated_pixels_1 = f'activated_pixels_for_object_1_{threshold:.0e}'
str_activated_pixels_2 = f'activated_pixels_for_object_2_{threshold:.0e}'

str_activated_pixels_1_old = f'activated_pixels_for_object_1_old_{threshold:.0e}'
str_activated_pixels_2_old = f'activated_pixels_for_object_2_old_{threshold:.0e}'

for prompt in df['prompt'].unique()[:5]:
    print(f"prompt: {prompt}")
    # only plot the first prompt
    sub_df = df[df['prompt'] == prompt]

    # remove the row with the maximum activated pixels for object 2
    sub_df = sub_df[sub_df[str_activated_pixels_2] != sub_df[str_activated_pixels_2].max()]
    # remove the row with the minimum clip score
    sub_df = sub_df[sub_df['clip_score'] != sub_df['clip_score'].min()]

    # print the ratio of samples with activated pixels larger than 200 for both objects
    print(f"ratio of samples with activated pixels larger than 200 for {prompt}: {(sub_df[[str_activated_pixels_1, str_activated_pixels_2]].min(axis=1) > 200).mean():.4f}")
    # print the clip score for all samples
    print(f"clip score for all samples: {sub_df['clip_score'].mean():.4f}")
    print(f"clip score for samples with activated pixels larger than 200: {sub_df[sub_df[[str_activated_pixels_1, str_activated_pixels_2]].min(axis=1) > 200]['clip_score'].mean():.4f}")

    # scatter activated pixels for both objects and clip score
    plt.clf()
    plt.close("all")
    fig = plt.figure(figsize=(20, 12))

    # Set font sizes
    plt.rcParams.update({'font.size': 40})

    # Create a special layout: left column spans 2 rows, right side is 2x2
    gs = plt.GridSpec(2, 3, figure=fig, width_ratios=[2, 1, 1], height_ratios=[1, 1])  # Add height_ratios

    # Add spacing between subplots
    gs.update(wspace=0.1, hspace=0.17)

    ax_scatter = fig.add_subplot(gs[:, 0])  # Span both rows in first column
    ax_images = [
        fig.add_subplot(gs[0, 1]),  # Top middle
        fig.add_subplot(gs[0, 2]),  # Top right
        fig.add_subplot(gs[1, 1]),  # Bottom middle
        fig.add_subplot(gs[1, 2])   # Bottom right
    ]

    # Find the index of the lowest clip score
    min_clip_row = sub_df.loc[sub_df['second_half_text_score'].idxmin()]
    min_pixel_num_row = sub_df.loc[sub_df[[str_activated_pixels_2, str_activated_pixels_2]].min(axis=1).idxmin()]
    max_clip_row = sub_df.loc[sub_df['second_half_text_score'].idxmax()]
    max_pixel_num_row = sub_df.loc[sub_df[[str_activated_pixels_2, str_activated_pixels_2]].min(axis=1).idxmax()]

    # remove the above rows from the dataframe
    other_rows = sub_df[~sub_df.index.isin([min_clip_row.name, min_pixel_num_row.name, max_clip_row.name, max_pixel_num_row.name])]

    # Define border colors matching scatter plot colors
    border_colors = {
        'lowest_clip_score': GOOGLE_GREEN,
        'lowest_pixel_num': GOOGLE_RED,
        'highest_clip_score': GOOGLE_YELLOW,
        'highest_pixel_num': GOOGLE_BLUE
    }

    for name, row in [('lowest_clip_score', min_clip_row), ('lowest_pixel_num', min_pixel_num_row), ('highest_clip_score', max_clip_row), ('highest_pixel_num', max_pixel_num_row)]:
        lowest_clip_image_name = row['image_name']
        lowest_clip_image_path = f"{original_folder}/{prompt}/{lowest_clip_image_name}"

        # Open and add border using PIL
        from PIL import Image
        img = Image.open(lowest_clip_image_path)
        border_width = 40  # Adjust border width as needed
        border_color = tuple(int(border_colors[name].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))  # Convert hex to RGB
        img_with_border = Image.new('RGB', (img.width + 2*border_width, img.height + 2*border_width), border_color)
        img_with_border.paste(img, (border_width, border_width))
        img_with_border.save(f"figures/{name}_{prompt}_image.png")

    ax_scatter.scatter(other_rows[str_activated_pixels_2], other_rows['second_half_text_score'], c=GOOGLE_TILTED_GREY, s=100)
    ax_scatter.scatter(min_clip_row[str_activated_pixels_2], min_clip_row['second_half_text_score'], c=GOOGLE_GREEN, s=300, zorder=10)
    ax_scatter.scatter(min_pixel_num_row[str_activated_pixels_2], min_pixel_num_row['second_half_text_score'], c=GOOGLE_RED, s=300, zorder=10)
    ax_scatter.scatter(max_clip_row[str_activated_pixels_2], max_clip_row['second_half_text_score'], c=GOOGLE_YELLOW, s=300, zorder=10)
    ax_scatter.scatter(max_pixel_num_row[str_activated_pixels_2], max_pixel_num_row['second_half_text_score'], c=GOOGLE_BLUE, s=300, zorder=10)

    # Calculate the confidence interval
    from scipy import stats

    # Get the data
    x = sub_df[str_activated_pixels_2]
    y = sub_df['second_half_text_score']

    # Fit a linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Create prediction interval
    # Get the prediction for the line
    y_pred = slope * x + intercept

    # Calculate standard error of prediction
    n = len(x)
    x_mean = np.mean(x)
    # Calculate confidence interval
    conf = 0.95
    alpha = 1 - conf
    t = stats.t.ppf(1 - alpha/2, n-2)  # t-value for confidence interval
    std_dev = np.sqrt(sum((y - y_pred) ** 2) / (n-2))

    # Plot the confidence band
    x_sorted = np.sort(x)
    y_pred_line = slope * x_sorted + intercept

    ci = t * std_dev * np.sqrt(1/n + (x_sorted - x_mean)**2 / sum((x - x_mean)**2))

    # Plot the band
    ax_scatter.fill_between(x_sorted,
                    y_pred_line - ci,
                    y_pred_line + ci,
                    color=GOOGLE_GREY,
                    alpha=0.2)

    # Plot the regression line
    ax_scatter.plot(x_sorted, y_pred_line, '--', color=GOOGLE_GREY)

    ax_scatter.set_title(f"Pearson Correlation: {sub_df[str_activated_pixels_2].corr(sub_df['second_half_text_score']):.4f}", fontsize=40)
    ax_scatter.set_xlabel("\#CompLift Activated Pixels of \\em{{A Mouse}}", fontsize=40)
    ax_scatter.set_ylabel("CLIP Score of \\em{{A Mouse}}", fontsize=40)
    ax_scatter.tick_params(axis='both', which='major', labelsize=40)

    ax_scatter.grid(False)
    ax_scatter.grid(True, which="major", ls="-", alpha=0.2)
    ax_scatter.grid(True, which="minor", ls=":", alpha=0.2)

    # Show images in 2x2 grid on the right
    for i, image_name in enumerate([min_pixel_num_row['image_name'], min_clip_row['image_name'], max_clip_row['image_name'], max_pixel_num_row['image_name']]):
        name = ['lowest_pixel_num', 'lowest_clip_score', 'highest_clip_score', 'highest_pixel_num']
        image_path = f"figures/{name[i]}_{prompt}_image.png"
        img = Image.open(image_path)
        ax_images[i].imshow(img)
        ax_images[i].axis('off')
        # make adjustments to the image
        ax_images[i].set_aspect('auto')

        if i <= 1:
            ax_images[i].set_title(f"Rejected", fontsize=40)
        else:
            ax_images[i].set_title(f"Accepted", fontsize=40)

    # Make the scatter plot square
    # ax_scatter.set_aspect('equal', adjustable='box')

    # Adjust the layout to be more compact
    plt.tight_layout(pad=3.0)
    plt.savefig(f"figures/pixel_based_lift_correlation_{prompt}.pdf", bbox_inches='tight')
