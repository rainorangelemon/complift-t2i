import ImageReward as RM
import pandas as pd
import os
from tqdm import tqdm

model = RM.load("ImageReward-v1.0")

def add_image_reward_scores(folder_path):
    """Add ImageReward scores to the metrics.csv file in the given folder."""

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

if __name__ == "__main__":
    folders = [
        "outputs/standard_sd_1_4_lift",
        "outputs/standard_sd_2_1_lift",
        "outputs/standard_sd_xl_lift"
    ]

    for folder in folders:
        print(f"\nProcessing {folder}...")
        add_image_reward_scores(folder)
