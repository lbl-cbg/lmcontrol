#!/bin/bash
#SBATCH -A m3513
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -n 1
#SBATCH --time=2:30:00
#SBATCH --array=0  # Only one job in the array (since we're only using one name)
#SBATCH -e /pscratch/sd/n/niranjan/error/clf_error_rep/opta_1296_combis/optatune.%A_10_18_03_00.err
#SBATCH -o /pscratch/sd/n/niranjan/error/clf_error_rep/opta_1296_combis/optatune.%A_10_18_03_00.out
#SBATCH -J opta_10_18_03_00  # Job name

# Assign custom name (SLURM_ARRAY_TASK_ID is replaced with the custom name)
task_name="10_22_03_00"

# SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-10_15_03_00}

lmcontrol opta-clf-train \
    --training \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S6/S6_HT1/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S6/S6_HT3/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S6/S6_HT5/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S6/S6_HT7/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S6/S6_HT9/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S6/S6_HT11/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S13/S13_HT1/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S13/S13_HT3/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S13/S13_HT5/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S13/S13_HT7/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S13/S13_HT9/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S13/S13_HT11/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S17/S17_HT1/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S17/S17_HT3/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S17/S17_HT5/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S17/S17_HT7/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S17/S17_HT9/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S17/S17_HT11/all_processed.npz \
    --val_frac 0.2 \
    --seed 42 \
    --checkpoint /pscratch/sd/n/niranjan/output/optatune/ambr03/opta_${task_name}/ \
    --epochs 5 \
    --outdir /pscratch/sd/n/niranjan/output/optatune/ambr03/opta_${task_name}/ \
    -n 95 \
    --early_stopping \
    testing0 \
    feed

best_ckpt=$(ls /pscratch/sd/n/niranjan/output/optatune/ambr03/opta_${task_name}/ | \
    grep 'checkpoint-' | \
    sort -t '-' -k3,3 -g | \
    tail -n 1)

best_ckpt_path="/pscratch/sd/n/niranjan/output/optatune/ambr03/opta_${task_name}/$best_ckpt"

lmcontrol opta-clf-predict \
    --prediction \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S6/S6_HT2/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S6/S6_HT4/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S6/S6_HT6/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S6/S6_HT8/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S6/S6_HT10/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S6/S6_HT12/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S13/S13_HT2/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S13/S13_HT4/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S13/S13_HT6/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S13/S13_HT8/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S13/S13_HT10/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S13/S13_HT12/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S17/S17_HT2/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S17/S17_HT4/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S17/S17_HT6/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S17/S17_HT8/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S17/S17_HT10/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/ambr_03/S17/S17_HT12/all_processed.npz \
    --checkpoint $best_ckpt_path \
    -o /pscratch/sd/n/niranjan/output/ambr03/prediction_${task_name}.npz \
    -n 95 \
    feed \
    # --save_confusion /pscratch/sd/n/niranjan/output/optatune/ambr03_misclassify/ \
    # --save_misclassified /pscratch/sd/n/niranjan/output/optatune/ambr03_misclassify/misclassified_${SLURM_ARRAY_TASK_ID} \
   
# Here time is expected to below -n in any case