#!/bin/bash
#SBATCH -J  xmem-segswap
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12  # You can give each process multiple threads/cpus
#SBATCH --time=06-23:00:00     # DD-HH:MM:SS
#SBATCH --mem=32G           # Max memory (CPU) 16GB
#SBATCH --output=/localscratch/yma50/experiments/logs/xmem-segswap-%N-%j.out # Terminal output to file named (hostname)-(jobid).out
#SBATCH --partition=long     # long partition (allows up to 7 days runtime)
#SBATCH --nodelist=cs-venus-09   # if needed, set the node you want (similar to -w xyz)

ulimit -Sv unlimited
ulimit -Su unlimited

MAIN_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MAIN_ADDR
source /home/yma50/miniconda3/etc/profile.d/conda.sh
cd /localscratch/yma50/XMemSegSwap
conda activate xmem
export OPENBLAS_NUM_THREADS=1

echo $CUDA_AVAILABLE_DEVICES

echo "Starting script..."

set -x
srun python -m torch.distributed.launch --master_port 25754 --nproc_per_node=1 train.py --exp_id xmem_segswap --stage 3 --load_network pretrained/XMem.pth --egoexo_root ./data/correspondence/ --num_workers 4 --segswap_model pretrained/segswap_egoexo.pth --save_network_interval 1000 --save_checkpoint_interval 2000
# git pull
# python eval.py --model pretrained/XMem.pth --save_all --output egoexo_pretrained_test_24 --e23_path data/correspondence --split test
