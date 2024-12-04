#!/bin/bash
#SBATCH -A m3513
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -c 32
#SBATCH --gpus=4
#SBATCH -N 1
#SBATCH --ntasks-per-node=4  #the script mentioned 4 originally, but we do not have 4 GPUs with us
#SBATCH --time=20:00 
#SBATCH -e /pscratch/sd/n/niranjan/error/clf_error_rep/opta_1296_combis/optatune.%A_byol_combined_200e_version1.err
#SBATCH -o ls .%A_byol_combined_200e_version1.out
#SBATCH -J test_lightning  # Job name

export SLURM_CPU_BIND="cores"

# Assign custom name (SLURM_ARRAY_TASK_ID is replaced with the custom name)
task_name="byol_combined_200e_version1"
INPUT_DIR="$SCRATCH/tar_ball/segmented_square_96"
INPUT_DIR1="$SCRATCH/tar_ball/ambr_03/"
INPUT_DIR2="/pscratch/sd/n/niranjan/tar_ball/ABF_FA_AMBR01-NSR"
INPUT_DIR3="/pscratch/sd/n/niranjan/tar_ball/bg_noise_images"
INPUT_DIR4="/pscratch/sd/n/niranjan/tar_ball/Images_from_water"

# Run the script with the task_name
srun lmcontrol train-byol \
    --training \
    $INPUT_DIR/S4/S4_HT1/all_processed.npz \
    $INPUT_DIR/S10/S10_HT4/all_processed.npz \
    $INPUT_DIR/S14/S14_HT7/all_processed.npz \
    $INPUT_DIR/S4/S4_HT2/all_processed.npz \
    $INPUT_DIR/S4/S4_HT6/all_processed.npz \
    $INPUT_DIR4/Water_10/all_processed.npz \
    $INPUT_DIR4/Water_11/all_processed.npz \
    $INPUT_DIR4/Water_12/all_processed.npz \
    --val_frac 0.2 \
    --seed 42 \
    --checkpoint /pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/ \
    --epochs 5 \
    --outdir /pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/ \
    -n 160 \
    --accelerator "gpu" \
    --strategy "ddp" \
    --devices 4 \
    --num_nodes 1 \
    --stop_wandb

# Get the best checkpoint
best_ckpt=$(ls /pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/ | \
    grep 'checkpoint-' | \
    sort -t '-' -k3,3 -g | \
    tail -n 1)

best_ckpt_path="/pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/$best_ckpt"

# # Run the prediction step using the best checkpoint
# lmcontrol infer-byol \
#     --prediction \
#     $INPUT_DIR/S4/S4_HT1/all_processed.npz \
#     $INPUT_DIR/S4/S4_HT4/all_processed.npz \
#     $INPUT_DIR/S4/S4_HT7/all_processed.npz \
#     $INPUT_DIR/S14/S14_HT1/all_processed.npz \
#     $INPUT_DIR3/HT1.2/all_processed.npz \
#     $INPUT_DIR3/HT1.3/all_processed.npz \
#     $INPUT_DIR4/Water_1/all_processed.npz \
#     --checkpoint $best_ckpt_path \
#     -o /pscratch/sd/n/niranjan/output/prediction_${task_name}.npz \
#     -n 100 \
#     # -save_emb \
#     # --save_misclassified /pscratch/sd/n/niranjan/output/optatune/ambr01_misclassify/misclassified_${task_name} \
#     # --save_confusion /pscratch/sd/n/niranjan/output/optatune/ambr01_misclassify/


