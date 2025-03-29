import pprint
from typing import List

import pyrallis
import torch
from PIL import Image
from pathlib import Path

from config import RunConfig, ModelConfigs, LiftConfig
from pipeline_attend_and_excite import AttendAndExcitePipeline
from pipeline_attend_and_excite_xl import AttendAndExcitePipelineXL
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore
from lift_callback import LiftCallback


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_model(config: RunConfig):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Get model configuration
    model_type, model_config = ModelConfigs().get_model_config(config.sd_2_1, config.sd_xl)
    config.attention_res = model_config['attention_res']

    # Initialize the appropriate pipeline
    pipeline_class = (AttendAndExcitePipelineXL if config.sd_xl else AttendAndExcitePipeline) if not config.run_standard_sd \
        else (StableDiffusionXLPipeline if config.sd_xl else StableDiffusionPipeline)

    stable = pipeline_class.from_pretrained(
        model_config['version'],
        **model_config['params']
    ).to(device)

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
                  callback: LiftCallback = None) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    with torch.inference_mode(mode=config.run_standard_sd):
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
                        normalize_eot=config.normalize_eot,
                        sd_2_1=config.sd_2_1,
                        callback_on_step_end=callback,
                        callback_on_step_end_tensor_inputs=model._callback_tensor_inputs,
                        )
    image = outputs.images[0]
    return image


def save_results(image: Image.Image,
                output_path: Path,
                prompt: str,
                seed: int,
                data_to_save: dict):
    """Save image and associated data for a given seed."""
    prompt_output_path = output_path / prompt
    prompt_output_path.mkdir(exist_ok=True, parents=True)

    # Save image
    image.save(prompt_output_path / f'{seed}.png')

    # Save associated data
    torch.save(data_to_save, prompt_output_path / f'{seed}_lift_results.pt')


def process_and_save_predictions(callback: LiftCallback,
                           config: LiftConfig,
                           seed: int,
                           image: Image.Image) -> Image.Image:
    data_to_save = {
        "latents": callback.latest_latents.clone() if config.save_intermediate_latent else None,
        "intermediate_latents": callback.intermediate_latents if config.save_intermediate_latent else None,
        "intermediate_ts": callback.intermediate_ts if config.save_intermediate_latent else None,
    }

    save_results(image, config.output_path, config.prompt, seed, data_to_save)
    callback.clear()
    return image


@pyrallis.wrap()
def main(config: LiftConfig):

    callback = LiftCallback(config)

    stable = load_model(config)
    token_indices = get_indices_to_alter(stable, config.prompt) if config.token_indices is None else config.token_indices

    base_tensor_inputs = [
        "latent_model_input",
    ]
    stable._callback_tensor_inputs += base_tensor_inputs

    images = []
    seeds_to_process = config.seeds.copy()
    while seeds_to_process:
        seed = seeds_to_process.pop(0)
        print(f"Seed: {seed}")

        # # Skip if image already exists
        # if (config.output_path / config.prompt / f'{seed}.png').exists():
        #     print(f"Image for seed {seed} already exists, skipping...")
        #     continue

        g = torch.Generator('cuda').manual_seed(seed)
        controller = AttentionStore()
        image = run_on_prompt(prompt=config.prompt,
                            model=stable,
                            controller=controller,
                            token_indices=token_indices,
                            seed=g,
                            config=config,
                            callback=callback)

        image = process_and_save_predictions(callback, config, seed, image)
        images.append(image)

    if len(images) > 0:
        # save a grid of results across all seeds
        joined_image = vis_utils.get_image_grid(images)
        joined_image.save(config.output_path / f'{config.prompt}.png')


if __name__ == '__main__':
    main()
