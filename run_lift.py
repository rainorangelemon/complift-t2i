import pprint
from typing import List

import pyrallis
import torch
from PIL import Image
from pathlib import Path

from config import RunConfig, LiftConfig
from pipeline_attend_and_excite import AttendAndExcitePipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore
from lift_callback import LiftCallback

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_model(config: RunConfig):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if config.sd_2_1:
        stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
    else:
        stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
    stable = AttendAndExcitePipeline.from_pretrained(stable_diffusion_version).to(device)
    torch.set_float32_matmul_precision('high')
    return stable


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices


def run_on_prompt(prompt: List[str],
                  model: AttendAndExcitePipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  config: RunConfig,
                  callback: LiftCallback) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)

    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1,
                    callback=callback)
    image = outputs.images[0]
    return image


@pyrallis.wrap()
def main(config: LiftConfig):
    callback = LiftCallback(config)
    stable = load_model(config)
    token_indices = get_indices_to_alter(stable, config.prompt) if config.token_indices is None else config.token_indices

    images = []
    seeds_to_process = config.seeds.copy()  # Create a working copy
    while seeds_to_process:
        seed = seeds_to_process.pop(0)  # Get next seed
        print(f"Seed: {seed}")
        g = torch.Generator('cuda').manual_seed(seed)
        controller = AttentionStore()
        image = run_on_prompt(prompt=config.prompt,
                              model=stable,
                              controller=controller,
                              token_indices=token_indices,
                              seed=g,
                              config=config,
                              callback=callback)
        log_lift_results, is_valid = callback.calculate_lift(pipeline=stable,
                                prompts=config.prompts,
                                algebras=["product"]*len(config.prompts),
                                return_intermediate_results=True,
                                cross_attention_kwargs=None)

        if not is_valid.all():
            output_path_to_save = Path(str(config.output_path) + "_invalid")
        else:
            output_path_to_save = config.output_path

        prompt_output_path = output_path_to_save / config.prompt
        prompt_output_path.mkdir(exist_ok=True, parents=True)
        image.save(prompt_output_path / f'{seed}.png')
        images.append(image)

        # save the log_lift_results
        torch.save(log_lift_results, prompt_output_path / f'{seed}_log_lift_results.pt')

        if not is_valid.all():
            print(f"Seed {seed} is not valid")
            new_seed = seeds_to_process[-1] + 1 if seeds_to_process else seed + 1
            seeds_to_process.append(new_seed)

    # save a grid of results across all seeds
    joined_image = vis_utils.get_image_grid(images)
    joined_image.save(config.output_path / f'{config.prompt}.png')


if __name__ == '__main__':
    main()
