#!/bin/bash
#SBATCH -A m3513
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -n 1
#SBATCH --time=1:30:00
#SBATCH --array=0  # Only one job in the array (since we're only using one name)
#SBATCH -e /pscratch/sd/n/niranjan/error/clf_error_rep/opta_1296_combis/optatune.%A_multilabels_1e-3_40e.err
#SBATCH -o /pscratch/sd/n/niranjan/error/clf_error_rep/opta_1296_combis/optatune.%A_multilabels_1e-3_40e.out
#SBATCH -J multilabels_1e-3_40e  # Job name

# Assign custom name (SLURM_ARRAY_TASK_ID is replaced with the custom name)
task_name="byol_delete"
INPUT_DIR="$SCRATCH/tar_ball/segmented_square_96"

# # Run the script with the task_name
# lmcontrol train-byol \
#     --training \
#     $INPUT_DIR/S4/S4_HT1/all_processed.npz \
#     $INPUT_DIR/S10/S10_HT4/all_processed.npz \
#     $INPUT_DIR/S14/S14_HT7/all_processed.npz \
#     --validation \
#     $INPUT_DIR/S4/S4_HT2/all_processed.npz \
#     $INPUT_DIR/S10/S10_HT5/all_processed.npz \
#     $INPUT_DIR/S14/S14_HT8/all_processed.npz \
#     --checkpoint /pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/ \
#     --epochs 5 \
#     --outdir /pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/ \
#     -n 95 \
#     --stop_wandb \

# Get the best checkpoint
best_ckpt=$(ls /pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/ | \
    grep 'checkpoint-' | \
    sort -t '-' -k3,3 -g | \
    tail -n 1)

best_ckpt_path="/pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/$best_ckpt"

# Run the prediction step using the best checkpoint
lmcontrol infer-byol \
    --prediction \
    $INPUT_DIR/S4/S4_HT3/all_processed.npz \
    $INPUT_DIR/S10/S10_HT6/all_processed.npz \
    $INPUT_DIR/S14/S14_HT9/all_processed.npz \
    --checkpoint $best_ckpt_path \
    -o /pscratch/sd/n/niranjan/output/prediction_${task_name}.npz \
    -n 95 \
    # -save_emb \
    # --save_misclassified /pscratch/sd/n/niranjan/output/optatune/ambr01_misclassify/misclassified_${task_name} \
    # --save_confusion /pscratch/sd/n/niranjan/output/optatune/ambr01_misclassify/


