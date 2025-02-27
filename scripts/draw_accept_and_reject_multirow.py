import os
import json
from PIL import Image

GOOGLE_RED = "#F44336"
GOOGLE_BLUE = "#2196F3"
GOOGLE_GREEN = "#4CAF50"
GOOGLE_YELLOW = "#FFC107"
GOOGLE_GREY = "#9E9E9E"
GOOGLE_LIGHT_GREY = "#E0E0E0"
GOOGLE_TILTED_GREY = "#BDBDBD"

# original_folder = "outputs/weak_lift"
original_folder = "outputs/standard_sd_xl_lift"
# original_folder = "outputs/standard_sd_2_1_lift"
# original_folder = "outputs/standard_sd_1_4_lift"

import pandas as pd
df = pd.read_csv(f'{original_folder}/metrics.csv')
thresholds = [0, 9e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3]

# %%
# plot a few samples which are rejected
optimal_threshold = 2e-4
optimal_pixel_threshold = 300

threshold_str = f"{optimal_threshold:.0e}"

# plot the samples which are rejected with the prompt
sub_df_rejected = df[df[[f'activated_pixels_for_object_1_{threshold_str}', f'activated_pixels_for_object_2_{threshold_str}']].min(axis=1) < optimal_pixel_threshold]

# 1. First get the value counts
prompt_counts = sub_df_rejected['prompt'].value_counts()

# 2. Create the boolean mask for prompts that aren't fully rejected
not_fully_rejected_and_not_fully_accepted_prompts = prompt_counts[(prompt_counts != 5) & (prompt_counts != 0)].index

# 3. Filter the DataFrame using isin()
sub_df_not_fully = sub_df_rejected[sub_df_rejected['prompt'].isin(not_fully_rejected_and_not_fully_accepted_prompts)]


# %%
import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

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

# Create a 2x20 subplot layout
plt.clf()
plt.close("all")

designated_prompts = [
 'a white bow and a white car',
 'a purple chair and a red bow',
 'a horse and a mouse',
 'a yellow backpack and a gray apple',
 'a elephant with a bow',
 'a cat and a elephant',
 'a turtle and a mouse',
 'a bear and a mouse',
 'a bird and a horse',
 'a elephant and a rabbit',
 'a green backpack and a purple bench',
 'a red glasses and a red suitcase',
 'a lion with a bow',
 'a elephant and a turtle',
 'a frog and a rabbit',
 'a gray backpack and a yellow glasses',
 'a cat and a mouse',
 'a gray crown and a white clock',
 'a red backpack and a yellow bowl',
 'a bird with a glasses',
#  'a rabbit and a blue bowl',
#  'a cat and a yellow car',
 'a lion and a elephant',
 'a elephant and a pink backpack',
 'a bird and a yellow car',
 'a horse and a rabbit',
 'a purple car and a pink apple',
 'a white glasses and a orange balloon',
 'a green balloon and a pink bowl',
 'a white chair and a gray balloon',
 'a yellow glasses and a brown bow',
 'a blue balloon and a blue bow',
 'a bird and a black backpack',
 'a turtle and a gray backpack',
 'a white bow and a black clock',
 'a lion and a mouse',
 'a dog and a rabbit',
 'a pink crown and a purple bow',
#  'a monkey with a bow',
 'a mouse and a red car',
 'a white suitcase and a white chair',
 'a dog and a frog',
 'a frog and a turtle',
 'a gray crown and a purple apple',
 'a frog and a pink bench',
 'a yellow suitcase and a yellow car',
 'a yellow glasses and a gray bowl',
#  'a cat with a bow',
#  'a cat and a frog',
 'a horse with a bow',
#  'a bear and a red balloon',
 'a monkey and a mouse',
#  'a mouse and a red bench',
#  'a bird and a bear',
#  'a bear and a white car',
 'a black backpack and a pink balloon',
#  'a elephant and a mouse',
#  'a dog and a gray bowl',
#  'a monkey and a blue chair',
#  'a lion with a glasses',
#  'a cat and a red apple',
#  'a mouse with a glasses',
#  'a elephant and a frog',
#  'a lion and a white chair',
#  'a dog and a pink bench',
 'a lion and a horse',
 'a bird and a black suitcase',
 'a cat and a blue backpack',
 'a orange suitcase and a brown bench',
 'a cat and a green clock',
 'a bear and a frog',
 'a horse and a blue bench',
 'a bear with a bow',
 'a dog with a glasses',
 'a elephant and a monkey',
 'a lion and a monkey',
 'a dog and a bear',
 'a cat with a glasses'
]

border_colors = {
    'accepted': GOOGLE_BLUE,
    'rejected': GOOGLE_YELLOW,
}

designated_prompts = designated_prompts[24:48]

def convert_a_to_an_for_object(object):
    if object[0].lower() in "aeiou":
        result = f"an {object}"
    else:
        result = f"a {object}"
    if result.endswith("es"):
        result = " ".join(result.split(" ")[1:])
    return rf"\textcolor{{GoogleBlue}}{{{result}}}"

def convert_prompt_to_title(prompt):
    if "and" in prompt:
        object_1, object_2 = prompt.split(" and ")
        object_1 = convert_a_to_an_for_object(" ".join(object_1.split(" ")[1:]))
        object_2 = convert_a_to_an_for_object(" ".join(object_2.split(" ")[1:]))
        return rf"{object_1}" + "\n"+ r"and" + "\n" + rf"{object_2}"
    else:
        assert "with" in prompt
        object_1, object_2 = prompt.split(" with ")
        object_1 = convert_a_to_an_for_object(" ".join(object_1.split(" ")[1:]))
        object_2 = convert_a_to_an_for_object(" ".join(object_2.split(" ")[1:]))
        return rf"{object_1}" + "\n" + r"with" + "\n" + rf"{object_2}"

# Calculate dimensions for the subplot grid
n_prompts = len(designated_prompts)
n_cols = 6
n_rows = 3 * ((n_prompts + n_cols - 1) // n_cols)  # Now multiply by 3 instead of 2

# Create the subplot layout
fig = plt.figure(figsize=(30, 4*n_rows))  # Keep original total height

# Create a gridspec with different row heights
# Calculate the number of actual rows needed (n_prompts rows × 3 rows each)
actual_rows = (n_prompts + n_cols - 1) // n_cols * 3

from matplotlib import gridspec
gs = gridspec.GridSpec(actual_rows, n_cols, height_ratios=[0.6 if i % 3 == 0 else 1.3 for i in range(actual_rows)])
axes = []
for i in range(actual_rows):
    row = []
    for j in range(n_cols):
        row.append(fig.add_subplot(gs[i, j]))
    axes.append(row)
axes = np.array(axes)

# set the wspace and hspace
plt.subplots_adjust(wspace=0.01, hspace=0.01)

for idx, prompt in enumerate(designated_prompts):
    # Calculate row and column position
    base_row = (idx // n_cols) * 3  # Multiply by 3 instead of 2
    col = idx % n_cols

    # Rest of the image loading code remains the same
    all_image_names = [f'{i}.png' for i in range(5)]

    # Get worst (rejected) image
    worst_row = sub_df_not_fully[sub_df_not_fully['prompt'] == prompt].loc[
        sub_df_not_fully[sub_df_not_fully['prompt'] == prompt]['clip_score'].idxmin()
    ]
    rejected_image_name = worst_row['image_name']

    # Remove rejected image names from candidates for best image
    for image_name in sub_df_not_fully[sub_df_not_fully['prompt'] == prompt]['image_name']:
        all_image_names.remove(image_name)

    # Get best (accepted) image
    sub_df_all = df[df['prompt'] == prompt]
    sub_df_all = sub_df_all[sub_df_all['image_name'].isin(all_image_names)]
    best_row = sub_df_all.loc[sub_df_all['clip_score'].idxmax()]
    accepted_image_name = best_row['image_name']

    # Add text-only row
    axes[base_row, col].axis('off')
    axes[base_row, col].text(0.5, 0.5, convert_prompt_to_title(prompt),
                           ha='center', va='center', fontsize=32)

    # Plot accepted image in second row
    accepted_img = Image.open(f"outputs/standard_sd_xl_lift/{prompt}/{accepted_image_name}")
    border_width = 15  # Adjust border width as needed
    border_color = tuple(int(border_colors['accepted'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))  # Convert hex to RGB
    img_with_border = Image.new('RGB', (accepted_img.width + 2*border_width, accepted_img.height + 2*border_width), border_color)
    img_with_border.paste(accepted_img, (border_width, border_width))
    ax = axes[base_row + 1, col]
    ax.imshow(img_with_border)
    ax.axis('off')

    # Plot rejected image in third row
    rejected_img = Image.open(f"outputs/standard_sd_xl_lift/{prompt}/{rejected_image_name}")
    border_width = 15  # Adjust border width as needed
    border_color = tuple(int(border_colors['rejected'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))  # Convert hex to RGB
    img_with_border = Image.new('RGB', (rejected_img.width + 2*border_width, rejected_img.height + 2*border_width), border_color)
    img_with_border.paste(rejected_img, (border_width, border_width))
    ax = axes[base_row + 2, col]
    ax.imshow(img_with_border)
    ax.axis('off')

# Hide empty subplots
for idx in range(n_prompts, n_rows * n_cols):
    row = idx // n_cols
    col = idx % n_cols
    axes[row, col].axis('off')

plt.savefig("accepted_vs_rejected_more.pdf", backend='pgf', bbox_inches='tight')

plt.clf()
plt.close("all")

plt.figure()
plt.ylabel(r'\textcolor{red}{Today} '+
           r'\textcolor{green}{is} '+
           r'\textcolor{blue}{cloudy.}')
plt.savefig("test.pdf", backend='pgf')


