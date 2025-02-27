from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import torch

@dataclass
class ModelConfigs:
    sd_1_4: Dict = field(default_factory=lambda: {
        'version': "CompVis/stable-diffusion-v1-4",
        'attention_res': 16,
        'params': {}
    })
    sd_2_1: Dict = field(default_factory=lambda: {
        'version': "stabilityai/stable-diffusion-2-1-base",
        'attention_res': 16,
        'params': {}
    })
    sd_xl: Dict = field(default_factory=lambda: {
        'version': "stabilityai/stable-diffusion-xl-base-1.0",
        'attention_res': 32,
        'params': {
            'torch_dtype': torch.float16,
            'use_safetensors': True,
        }
    })

    def get_model_config(self, sd_2_1, sd_xl):
        """Returns the model type and configuration based on the current settings."""
        if sd_2_1:
            return 'sd_2_1', self.sd_2_1
        elif sd_xl:
            return 'sd_xl', self.sd_xl
        return 'sd_1_4', self.sd_1_4


@dataclass
class RunConfig:
    # Guiding text prompt
    prompt: str = None
    # Whether to use Stable Diffusion v2.1
    sd_2_1: bool = False
    # Whether to use Stable Diffusion XL
    sd_xl: bool = False
    # Whether to normalize the EOT token
    normalize_eot: bool = False
    # Which token indices to alter with attend-and-excite
    token_indices: List[int] = None
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [42])
    # Path to save all outputs to
    output_path: Path = None
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Number of denoising steps to apply attend-and-excite
    max_iter_to_alter: int = 25
    # Resolution of UNet to compute attention maps over
    attention_res: int = 16
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(default_factory=lambda: {0: 0.05, 10: 0.5, 20: 0.8})
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 20
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.5))
    # Whether to apply the Gaussian smoothing before computing the maximum attention value for each subject token
    smooth_attentions: bool = True
    # Standard deviation for the Gaussian smoothing
    sigma: float = 0.5
    # Kernel size for the Gaussian smoothing
    kernel_size: int = 3
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False
    # Add model configs
    model_configs: ModelConfigs = field(default_factory=ModelConfigs)

    def make_output_path(self):
        if self.output_path is None:
            if self.run_standard_sd:
                self.output_path = "./outputs/standard"
            else:
                self.output_path = "./outputs/ae"
            # add model version to the output path
            self.output_path = self.output_path + "_" + ModelConfigs().get_model_config(self.sd_2_1, self.sd_xl)[0]
            self.output_path = Path(self.output_path)

    def __post_init__(self):
        self.make_output_path()
        self.output_path.mkdir(exist_ok=True, parents=True)
        if self.sd_xl or self.sd_2_1:
            self.normalize_eot = True
        else:
            self.normalize_eot = False


@dataclass
class LiftConfig(RunConfig):
    # batch size to calculate the lift score
    batch_size: int = 1
    # whether noise is randomized for each t
    same_noise: bool = False
    # number of samples to calculate the lift score
    n_samples: int = 200
    # schedule of timesteps, interleave or random
    t_schedule: str = "interleave"
    # path to noise
    noise_path: Path = Path('./noise.pt')
    # prompts to calculate the lift score
    prompts: List[str] = None
    # whether to use lift
    use_lift: bool = True
    # whether to calculate the lift score
    lift_calculate: bool = True
    # number of steps to call the callback
    callback_steps: int = 1
    # whether to save the intermediate latent
    save_intermediate_latent: bool = False
    # whether to subtract the unconditional noise
    subtract_unconditional: bool = False

    def __post_init__(self):
        self.make_output_path()
        if self.use_lift:
            self.output_path = Path(str(self.output_path) + "_lift")
        super().__post_init__()
