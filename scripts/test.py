import json
import pandas as pd

with open("a.e_prompts.txt", "r") as f:
    prompt_classes = json.load(f)

df = pd.read_csv("outputs/ae_sd_2_1_lift/metrics.csv")
print(df.head())

# compute the average of the image reward score for each prompt class
for prompt_class, prompts in prompt_classes.items():
    print(prompt_class)
    sub_df = df[df['prompt'].isin(prompts)]
    # only keep the rows with f"{i}_png" where i<=4
    image_idx = [int(i.split(".png")[0]) for i in sub_df['image_name']]
    sub_df = sub_df[pd.Series(image_idx, index=sub_df.index) <= 4]
    print(len(prompts), len(sub_df))
