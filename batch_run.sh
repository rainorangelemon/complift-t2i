#!/bin/bash
pip install opencv-python transformers diffusers["torch"]
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs
git lfs install
git lfs fetch --all
python batch_run.py --sd_xl=True --run_standard_sd=True --use_lift=True --num_seeds=5
# python batch_run.py --sd_2_1=True --run_standard_sd=True --use_lift=True --num_seeds=5
# python batch_run.py --sd_2_1=True --use_lift=True --num_seeds=5
