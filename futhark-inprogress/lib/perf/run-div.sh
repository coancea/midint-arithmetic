#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=39000M
# we run on the gpu partition and we allocate 1 A100 GPUs
#SBATCH -p gpu --gres=gpu:a100:1
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=0:10:00

#your script, in this case: write the hostname and the ids of the chosen gpus.
module load cuda
module load futhark
hostname
echo $CUDA_VISIBLE_DEVICES
futhark bench --backend=cuda perf-div.fut -r 3 &> res-div.txt
