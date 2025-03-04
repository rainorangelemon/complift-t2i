!/bin/bash
# Check and install required Python packages
for package in "opencv-python" "transformers" "diffusers[torch]"; do
    if ! pip show $package &> /dev/null; then
        echo "Installing $package..."
        pip install $package
    else
        echo "$package is already installed"
    fi
done
# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "Installing Git LFS..."
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
    apt-get install git-lfs
    git lfs install
else
    echo "Git LFS is already installed"
fi
git lfs fetch --all

python batch_run.py               --run_standard_sd=True --num_seeds=5 --save_intermediate_latent=True
python batch_run.py --sd_2_1=True --run_standard_sd=True --num_seeds=5 --save_intermediate_latent=True
python batch_run.py --sd_xl=True  --run_standard_sd=True --num_seeds=5 --save_intermediate_latent=True

python batch_run.py --mode="analyze_logp.py" --output_path="outputs/standard_sd_1_4"
python batch_run.py --mode="analyze_logp.py" --output_path="outputs/standard_sd_2_1"
python batch_run.py --mode="analyze_logp.py" --output_path="outputs/standard_sd_xl"
