#!/bin/bash
#SBATCH --mem=180G 
#SBATCH --cpus-per-task=8 
#SBATCH --time=40:0:0 
#SBATCH --gres=gpu:v100l:2
#SBATCH --output=hard_neg_results/bm25_dev_ce.out
#SBATCH --error=hard_neg_results/bm25_dev_ce.err
export CUDA_AVAILABLE_DEVICES=0,1
module load java
source ~/ENV/bin/activate

python3 run.py --train True --eval True --config_path yaml_config/2021-02-06-triple.small.topics.tripletsampler.yaml
