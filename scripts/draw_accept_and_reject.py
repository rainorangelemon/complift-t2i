import os
import json
from PIL import Image
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
    "a turtle and a blue clock",
    "a frog with a bow",
    "a orange backpack and a purple car",
    "a elephant with a glasses",
    "a cat and a horse",
    "a black crown and a red car",
]

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

fig, axes = plt.subplots(2, len(designated_prompts), figsize=(5*len(designated_prompts), 10))
# set the wspace and hspace
plt.subplots_adjust(wspace=0.01, hspace=0.01)

# Add these lines to create vertical labels
fig.text(0.09, 0.7, r'Accepted' + "\n" + "with" + "\n" + r'\textit{CompLift}', va='center', ha='center', fontsize=32)
fig.text(0.09, 0.3, r'Rejected' + "\n" + "with" + "\n" + r'\textit{CompLift}', va='center', ha='center', fontsize=32)

for idx, prompt in enumerate(designated_prompts):
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

    # Plot accepted image in first row
    accepted_img = Image.open(f"outputs/standard_sd_xl_lift/{prompt}/{accepted_image_name}")
    axes[0, idx].imshow(accepted_img)
    axes[0, idx].axis('off')
    axes[0, idx].set_title(convert_prompt_to_title(prompt), fontsize=32, wrap=True)

    # Plot rejected image in second row
    rejected_img = Image.open(f"outputs/standard_sd_xl_lift/{prompt}/{rejected_image_name}")
    axes[1, idx].imshow(rejected_img)
    axes[1, idx].axis('off')

plt.savefig("accepted_vs_rejected.pdf", backend='pgf', bbox_inches='tight')

plt.clf()
plt.close("all")

plt.figure()
plt.ylabel(r'\textcolor{red}{Today} '+
           r'\textcolor{green}{is} '+
           r'\textcolor{blue}{cloudy.}')
plt.savefig("test.pdf", backend='pgf')


