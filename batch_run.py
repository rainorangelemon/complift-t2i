import json
import os
import sys
import glob

# load a file
with open('a.e_prompts.txt') as f:
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
    seeds_to_generate = []
    for seed_to_generate in range(64):
        if os.path.exists(f"outputs/{prompt}/{seed_to_generate}.png"):
            continue
        else:
            seeds_to_generate.append(seed_to_generate)
    if len(seeds_to_generate) == 0:
        continue
    seeds=f"[{','.join([str(i) for i in seeds_to_generate])}]"
    command = f"python run.py --prompt \"{prompt}\" --seeds {seeds} --token_indices {token_to_combine}"
    print(command)
    result = os.system(command)
    if result != 0:
        print("Error")
        sys.exit(1)
