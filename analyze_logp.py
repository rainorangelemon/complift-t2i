import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Optional, Dict, Any, Union
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DDPMScheduler
)
from tqdm import tqdm, trange
from lift_callback import LiftCallback
from config import LiftConfig, ModelConfigs
import pyrallis
import glob
from pathlib import Path
import os

def get_two_objects(prompt: str):
    if "with" in prompt:
        return clean_object_name(prompt.split("with")[0]), clean_object_name(prompt.split("with")[1])
    return clean_object_name(prompt.split("and")[0]), clean_object_name(prompt.split("and")[1])

def clean_object_name(object_name):
    if "a " in object_name:
        return object_name.split("a ")[1].strip()
    return object_name.strip()

def load_and_prepare_data(prompt: str):
    df = pd.read_csv(f"outputs/weak_lift/{prompt}.csv")
    image_names = df["image"].tolist()
    df = df.sort_values(by="image", key=lambda x: [int(image_name.split("-")[-1].split(".")[0]) for image_name in x])
    object_1, object_2 = get_two_objects(prompt)
    object_1 = clean_object_name(object_1)
    object_2 = clean_object_name(object_2)
    is_object_1 = (df["choice"] == f"{object_1} + {object_2}") | (df["choice"] == f"{object_1} + no {object_2}")
    is_object_2 = (df["choice"] == f"{object_1} + {object_2}") | (df["choice"] == f"no {object_1} + {object_2}")

    # Load tensors
    all_lift_results = torch.zeros(100, 2, 200)
    all_final_latents = torch.zeros(100, 4, 128, 128)
    all_intermediate_latents = torch.zeros(100, 50, 4, 128, 128)

    for i in range(100):
        lift_results = torch.load(f"outputs/weak_lift/{prompt}/{i}_lift_results.pt", weights_only=True)
        all_lift_results[i] = lift_results["log_lift_results"]
        all_final_latents[i] = lift_results["latents"]
        all_intermediate_latents[i] = torch.stack(lift_results["intermediate_latents"]).squeeze(1)

    return df, is_object_1, is_object_2, all_lift_results, all_final_latents, all_intermediate_latents

def load_and_prepare_data_from_folder(folder_path: Path):
    pt_files = glob.glob(f"{str(folder_path)}/*_lift_results.pt")
    pt_files.sort(key=lambda x: int(x.split("/")[-1].split("_")[0]))
    seeds = [int(pt_file.split("/")[-1].split("_")[0]) for pt_file in pt_files]
    all_final_latents = []
    for i, pt_file in enumerate(pt_files):
        all_final_latents.append(torch.load(pt_file, weights_only=True)["latents"])
    all_final_latents = torch.cat(all_final_latents, dim=0)
    all_intermediate_latents = []
    all_intermediate_ts = []
    for i, pt_file in enumerate(pt_files):
        all_intermediate_latents.append(torch.stack(torch.load(pt_file, weights_only=True)["intermediate_latents"]))
        all_intermediate_ts.append(torch.stack(torch.load(pt_file, weights_only=True)["intermediate_ts"]))
    all_intermediate_latents = torch.stack(all_intermediate_latents, dim=0)
    all_intermediate_ts = torch.stack(all_intermediate_ts, dim=0)
    return all_final_latents, all_intermediate_latents, all_intermediate_ts, seeds

def setup_pipeline(config: LiftConfig):
    model_type, model_config = ModelConfigs().get_model_config(config.sd_2_1, config.sd_xl)
    pipeline = DiffusionPipeline.from_pretrained(
        model_config['version'],
        **model_config['params']
    )
    pipeline.to("cuda")
    pipeline.unet = torch.compile(pipeline.unet, mode="max-autotune", fullgraph=True)
    pipeline.vae.to("cpu")
    return pipeline


@pyrallis.wrap()
def main(config_input: LiftConfig):
    all_final_latents, all_intermediate_latents, all_intermediate_ts, seeds = load_and_prepare_data_from_folder(config_input.output_path / config_input.prompt)

    if ("xl" in config_input.output_path.name) or config_input.sd_xl:
        config_input.sd_xl = True
        config_input.sd_2_1 = False
    elif ("2_1" in config_input.output_path.name) or config_input.sd_2_1:
        config_input.sd_2_1 = True
        config_input.sd_xl = False
    else:
        config_input.sd_2_1 = False
        config_input.sd_xl = False

    # Setup pipeline
    pipeline = setup_pipeline(config_input)

    # Setup config and callback
    config = LiftConfig(
        prompt=config_input.prompt,
        subtract_unconditional=False,
        batch_size=16,
        same_noise=config_input.same_noise,
        n_samples=config_input.n_samples,
    )
    callback = LiftCallback(config)

    if config_input.components is None:
        object_1, object_2 = get_two_objects(config_input.prompt)
        prompts = [config_input.prompt, object_1, object_2, ""]
    else:
        prompts = [config_input.prompt, *config_input.components, ""]

    # Calculate scores
    for img_idx in tqdm(range(len(all_final_latents))):
        score_result_path = f"{str(config_input.output_path)}/{config_input.prompt}/{seeds[img_idx]}_score_results.pt"
        # if not os.path.exists(score_result_path):
        score_result = callback.calculate_score(
            all_final_latents[[img_idx]].to("cuda:0"),
            pipeline,
            prompts=prompts
        )
        torch.save(score_result, score_result_path)

        cached_score_result_path = f"{str(config_input.output_path)}/{config_input.prompt}/{seeds[img_idx]}_cached_score_results.pt"
        # if not os.path.exists(cached_score_result_path):
        cached_score_result = callback.calculate_score_with_latent_model_inputs(
            pipeline,
            prompts=prompts,
            latent_model_inputs=all_intermediate_latents[[img_idx]].to("cuda:0"),
            timesteps=all_intermediate_ts[img_idx],
        )
        torch.save(cached_score_result, cached_score_result_path)


if __name__ == "__main__":
    main()
