#!/bin/bash
#SBATCH -A m4521
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -n 1
#SBATCH --time=1:30:00
#SBATCH --array=0-19
#SBATCH -e /pscratch/sd/n/niranjan/error/clf_error_rep/opta_20_trials_09/17/optatune.%A_%a 
#SBATCH -o /pscratch/sd/n/niranjan/error/clf_error_rep/opta_20_trials_09/17/optatune.%A_%a
#SBATCH -J optatune_20_trials_09/17

echo $SLURM_ARRAY_TASK_ID
lmcontrol opta-tune \
    -t 1 \
    --training \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT1/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT4/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT7/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT10/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT1/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT4/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT7/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT10/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT1/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT4/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT7/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT10/all_processed.npz \
    --validation \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT2/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT5/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT8/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT11/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT2/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT5/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT8/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT11/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT2/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT5/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT8/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT11/all_processed.npz \
    -e 50 \
    -n 9500 \
    /pscratch/sd/n/niranjan/output/optatune/20_trials_09/17 \
    time

    # do 20 epochs and 9500 points and ALWAYS REMOVE THE OPTATUNE FILE
    # TrainingPermutations.ipynb stored in /pscratch/sd/n/niranjan/output/optatune
