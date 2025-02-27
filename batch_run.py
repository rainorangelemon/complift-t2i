import json
import os
import sys
import glob
import argparse
from config import LiftConfig
from pathlib import Path

# Add argument parser for known arguments
parser = argparse.ArgumentParser(description='Batch run lift processing')
parser.add_argument('--config_file', type=str, default='a.e_prompts.txt',
                    help='Path to the config file containing prompts')
parser.add_argument('--num_seeds', type=int, default=64,
                    help='Number of seeds to generate')
parser.add_argument('--output_path', type=str, default=None,
                    help='Path to save all outputs to')
parser.add_argument('--mode', type=str, default="run.py", choices=["run.py", "analyze_logp.py"],
                    help='Mode to run the script')

# Capture all unknown arguments
args, unknown_args = parser.parse_known_args()

# Convert unknown arguments back to command line format
additional_args = ' '.join(unknown_args)

# load a file
with open(args.config_file) as f:
    config = json.load(f)

prompts = [p for k, v in config.items() for p in v]
combination_types = [k for k, v in config.items() for p in v]
print(combination_types)
print(len(prompts))

for prompt, combination in zip(prompts, combination_types):
    token_to_combine = []
    if combination == "animals":
        token_to_combine = "[2,5]"
    elif combination == "animals_objects":
        token_to_combine = f"[2,{len(prompt.split())}]"
    elif combination == "objects":
        token_to_combine = "[3,7]"
    print([prompt.split()[idx-1] for idx in eval(token_to_combine)])
    word_to_split = "and" if ("and" in prompt) else "with"
    prompts = [s.strip().replace("a ", "") for s in prompt.split(word_to_split)]
    prompts = '[' + ','.join([f'"{p}"' for p in prompts]) + ']'
    seeds=f"[{','.join([str(i) for i in range(args.num_seeds)])}]"
    if args.output_path is not None:
        command = f"python {args.mode} --prompt \"{prompt}\" --seeds {seeds} --token_indices {token_to_combine} --prompts {prompts} {additional_args} --output_path {args.output_path}"
    else:
        command = f"python {args.mode} --prompt \"{prompt}\" --seeds {seeds} --token_indices {token_to_combine} --prompts {prompts} {additional_args}"
    print(command)
    result = os.system(command)
    if result != 0:
        print("Error")
        sys.exit(1)
