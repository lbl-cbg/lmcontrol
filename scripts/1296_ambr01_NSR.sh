#!/bin/bash
#SBATCH -A m3513
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -n 1
#SBATCH --time=2:30:00
#SBATCH --array=0  # Only one job in the array (since we're only using one name)
#SBATCH -e /pscratch/sd/n/niranjan/error/clf_error_rep/opta_1296_combis/optatune.%A_10_18_00_00.err
#SBATCH -o /pscratch/sd/n/niranjan/error/clf_error_rep/opta_1296_combis/optatune.%A_10_18_00_00.out
#SBATCH -J opta_10_18_00_00  # Job name

# Assign custom name (SLURM_ARRAY_TASK_ID is replaced with the custom name)
task_name="10_22_00_00"

INPUT_DIR="/pscratch/sd/n/niranjan/tar_ball/ABF_FA_AMBR01-NSR"

# Run the script with the task_name
lmcontrol opta-clf-train \
    --training \
    $INPUT_DIR/S1/S1_HT2/all_processed.npz \
    $INPUT_DIR/S1/S1_HT7/all_processed.npz \
    $INPUT_DIR/S1/S1_HT10/all_processed.npz \
    $INPUT_DIR/S6/S6_HT2/all_processed.npz \
    $INPUT_DIR/S6/S6_HT7/all_processed.npz \
    $INPUT_DIR/S6/S6_HT10/all_processed.npz \
    $INPUT_DIR/S15/S15_HT2/all_processed.npz \
    $INPUT_DIR/S15/S15_HT7/all_processed.npz \
    $INPUT_DIR/S15/S15_HT10/all_processed.npz \
    $INPUT_DIR/S24/S24_HT2/all_processed.npz \
    $INPUT_DIR/S24/S24_HT7/all_processed.npz \
    $INPUT_DIR/S24/S24_HT10/all_processed.npz \
    $INPUT_DIR/S27/S27_HT2/all_processed.npz \
    $INPUT_DIR/S27/S27_HT7/all_processed.npz \
    $INPUT_DIR/S27/S27_HT10/all_processed.npz \
    $INPUT_DIR/S28/S28_HT2/all_processed.npz \
    $INPUT_DIR/S28/S28_HT7/all_processed.npz \
    $INPUT_DIR/S28/S28_HT10/all_processed.npz \
    --validation \
    $INPUT_DIR/S1/S1_HT4/all_processed.npz \
    $INPUT_DIR/S1/S1_HT8/all_processed.npz \
    $INPUT_DIR/S1/S1_HT11/all_processed.npz \
    $INPUT_DIR/S6/S6_HT4/all_processed.npz \
    $INPUT_DIR/S6/S6_HT8/all_processed.npz \
    $INPUT_DIR/S6/S6_HT11/all_processed.npz \
    $INPUT_DIR/S15/S15_HT4/all_processed.npz \
    $INPUT_DIR/S15/S15_HT8/all_processed.npz \
    $INPUT_DIR/S15/S15_HT11/all_processed.npz \
    $INPUT_DIR/S24/S24_HT4/all_processed.npz \
    $INPUT_DIR/S24/S24_HT8/all_processed.npz \
    $INPUT_DIR/S24/S24_HT11/all_processed.npz \
    $INPUT_DIR/S27/S27_HT4/all_processed.npz \
    $INPUT_DIR/S27/S27_HT8/all_processed.npz \
    $INPUT_DIR/S27/S27_HT11/all_processed.npz \
    $INPUT_DIR/S28/S28_HT4/all_processed.npz \
    $INPUT_DIR/S28/S28_HT8/all_processed.npz \
    $INPUT_DIR/S28/S28_HT11/all_processed.npz \
    --checkpoint /pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/ \
    --epochs 5 \
    --outdir /pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/ \
    -n 95 \
    testing0 \
    conditions

# Get the best checkpoint
best_ckpt=$(ls /pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/ | \
    grep 'checkpoint-' | \
    sort -t '-' -k3,3 -g | \
    tail -n 1)

best_ckpt_path="/pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/$best_ckpt"

# Run the prediction step using the best checkpoint
lmcontrol opta-clf-predict \
    --prediction \
    $INPUT_DIR/S1/S1_HT5/all_processed.npz \
    $INPUT_DIR/S1/S1_HT9/all_processed.npz \
    $INPUT_DIR/S1/S1_HT12/all_processed.npz \
    $INPUT_DIR/S6/S6_HT5/all_processed.npz \
    $INPUT_DIR/S6/S6_HT9/all_processed.npz \
    $INPUT_DIR/S6/S6_HT12/all_processed.npz \
    $INPUT_DIR/S15/S15_HT5/all_processed.npz \
    $INPUT_DIR/S15/S15_HT9/all_processed.npz \
    $INPUT_DIR/S15/S15_HT12/all_processed.npz \
    $INPUT_DIR/S24/S24_HT5/all_processed.npz \
    $INPUT_DIR/S24/S24_HT9/all_processed.npz \
    $INPUT_DIR/S24/S24_HT12/all_processed.npz \
    $INPUT_DIR/S27/S27_HT5/all_processed.npz \
    $INPUT_DIR/S27/S27_HT9/all_processed.npz \
    $INPUT_DIR/S27/S27_HT12/all_processed.npz \
    $INPUT_DIR/S28/S28_HT5/all_processed.npz \
    $INPUT_DIR/S28/S28_HT9/all_processed.npz \
    $INPUT_DIR/S28/S28_HT12/all_processed.npz \
    --checkpoint $best_ckpt_path \
    -o /pscratch/sd/n/niranjan/output/prediction_${task_name}.npz \
    -n 95 \
    conditions \
    # -save_emb \
    # --save_misclassified /pscratch/sd/n/niranjan/output/optatune/ambr01_NSR_misclassify/misclassified_${task_name} \
    # --save_confusion /pscratch/sd/n/niranjan/output/optatune/ambr01_NSR_misclassify/

    ## 10_14_00_00 , here its mm_dd_00/01/03 in case of ambr01_NSR, ambr01, ambr03 resp., followed by codes for sunning accordign to user