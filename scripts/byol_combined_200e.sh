#!/bin/bash
#SBATCH -A m3513
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -c 32
#SBATCH --gpus=4
#SBATCH -N 2
#SBATCH --ntasks-per-node=4  #the script mentioned 4 originally, but we do not have 4 GPUs with us
#SBATCH --time=24:00:00
#SBATCH -e /pscratch/sd/n/niranjan/error/clf_error_rep/opta_1296_combis/optatune.%A_byol_combined_200e_version2.err
#SBATCH -o /pscratch/sd/n/niranjan/error/clf_error_rep/opta_1296_combis/optatune.%A_byol_combined_200e_version2.out
#SBATCH -J byol_combined_200e_version2  # Job name

export SLURM_CPU_BIND="cores"

# Assign custom name (SLURM_ARRAY_TASK_ID is replaced with the custom name)
task_name="byol_combined_200e_version2"
INPUT_DIR="$SCRATCH/tar_ball/segmented_square_96"
INPUT_DIR1="$SCRATCH/tar_ball/ambr_03/"
INPUT_DIR2="/pscratch/sd/n/niranjan/tar_ball/ABF_FA_AMBR01-NSR"
INPUT_DIR3="/pscratch/sd/n/niranjan/tar_ball/bg_noise_images"
INPUT_DIR4="/pscratch/sd/n/niranjan/tar_ball/Images_from_water"

# Run the script with the task_name
srun lmcontrol train-byol \
    --training \
    $INPUT_DIR/S4/S4_HT1/all_processed.npz \
    $INPUT_DIR/S4/S4_HT4/all_processed.npz \
    $INPUT_DIR/S4/S4_HT7/all_processed.npz \
    $INPUT_DIR/S4/S4_HT10/all_processed.npz \
    $INPUT_DIR/S10/S10_HT1/all_processed.npz \
    $INPUT_DIR/S10/S10_HT4/all_processed.npz \
    $INPUT_DIR/S10/S10_HT7/all_processed.npz \
    $INPUT_DIR/S10/S10_HT10/all_processed.npz \
    $INPUT_DIR/S14/S14_HT1/all_processed.npz \
    $INPUT_DIR/S14/S14_HT4/all_processed.npz \
    $INPUT_DIR/S14/S14_HT7/all_processed.npz \
    $INPUT_DIR/S14/S14_HT10/all_processed.npz \
    $INPUT_DIR/S4/S4_HT2/all_processed.npz \
    $INPUT_DIR/S4/S4_HT5/all_processed.npz \
    $INPUT_DIR/S4/S4_HT8/all_processed.npz \
    $INPUT_DIR/S4/S4_HT11/all_processed.npz \
    $INPUT_DIR/S10/S10_HT2/all_processed.npz \
    $INPUT_DIR/S10/S10_HT5/all_processed.npz \
    $INPUT_DIR/S10/S10_HT8/all_processed.npz \
    $INPUT_DIR/S10/S10_HT11/all_processed.npz \
    $INPUT_DIR/S14/S14_HT2/all_processed.npz \
    $INPUT_DIR/S14/S14_HT5/all_processed.npz \
    $INPUT_DIR/S14/S14_HT8/all_processed.npz \
    $INPUT_DIR/S14/S14_HT11/all_processed.npz \
    $INPUT_DIR/S4/S4_HT3/all_processed.npz \
    $INPUT_DIR/S4/S4_HT6/all_processed.npz \
    $INPUT_DIR/S4/S4_HT9/all_processed.npz \
    $INPUT_DIR/S4/S4_HT12/all_processed.npz \
    $INPUT_DIR/S10/S10_HT3/all_processed.npz \
    $INPUT_DIR/S10/S10_HT6/all_processed.npz \
    $INPUT_DIR/S10/S10_HT9/all_processed.npz \
    $INPUT_DIR/S10/S10_HT12/all_processed.npz \
    $INPUT_DIR/S14/S14_HT3/all_processed.npz \
    $INPUT_DIR/S14/S14_HT6/all_processed.npz \
    $INPUT_DIR/S14/S14_HT9/all_processed.npz \
    $INPUT_DIR/S14/S14_HT12/all_processed.npz \
    $INPUT_DIR1/S6/S6_HT1/all_processed.npz \
    $INPUT_DIR1/S6/S6_HT3/all_processed.npz \
    $INPUT_DIR1/S6/S6_HT5/all_processed.npz \
    $INPUT_DIR1/S6/S6_HT7/all_processed.npz \
    $INPUT_DIR1/S6/S6_HT9/all_processed.npz \
    $INPUT_DIR1/S6/S6_HT11/all_processed.npz \
    $INPUT_DIR1/S13/S13_HT1/all_processed.npz \
    $INPUT_DIR1/S13/S13_HT3/all_processed.npz \
    $INPUT_DIR1/S13/S13_HT5/all_processed.npz \
    $INPUT_DIR1/S13/S13_HT7/all_processed.npz \
    $INPUT_DIR1/S13/S13_HT9/all_processed.npz \
    $INPUT_DIR1/S13/S13_HT11/all_processed.npz \
    $INPUT_DIR1/S17/S17_HT1/all_processed.npz \
    $INPUT_DIR1/S17/S17_HT3/all_processed.npz \
    $INPUT_DIR1/S17/S17_HT5/all_processed.npz \
    $INPUT_DIR1/S17/S17_HT7/all_processed.npz \
    $INPUT_DIR1/S17/S17_HT9/all_processed.npz \
    $INPUT_DIR1/S17/S17_HT11/all_processed.npz \
    $INPUT_DIR1/S6/S6_HT2/all_processed.npz \
    $INPUT_DIR1/S6/S6_HT4/all_processed.npz \
    $INPUT_DIR1/S6/S6_HT6/all_processed.npz \
    $INPUT_DIR1/S6/S6_HT8/all_processed.npz \
    $INPUT_DIR1/S6/S6_HT10/all_processed.npz \
    $INPUT_DIR1/S6/S6_HT12/all_processed.npz \
    $INPUT_DIR1/S13/S13_HT2/all_processed.npz \
    $INPUT_DIR1/S13/S13_HT4/all_processed.npz \
    $INPUT_DIR1/S13/S13_HT6/all_processed.npz \
    $INPUT_DIR1/S13/S13_HT8/all_processed.npz \
    $INPUT_DIR1/S13/S13_HT10/all_processed.npz \
    $INPUT_DIR1/S13/S13_HT12/all_processed.npz \
    $INPUT_DIR1/S17/S17_HT2/all_processed.npz \
    $INPUT_DIR1/S17/S17_HT4/all_processed.npz \
    $INPUT_DIR1/S17/S17_HT6/all_processed.npz \
    $INPUT_DIR1/S17/S17_HT8/all_processed.npz \
    $INPUT_DIR1/S17/S17_HT10/all_processed.npz \
    $INPUT_DIR1/S17/S17_HT12/all_processed.npz \
    $INPUT_DIR2/S1/S1_HT2/all_processed.npz \
    $INPUT_DIR2/S1/S1_HT7/all_processed.npz \
    $INPUT_DIR2/S1/S1_HT10/all_processed.npz \
    $INPUT_DIR2/S6/S6_HT2/all_processed.npz \
    $INPUT_DIR2/S6/S6_HT7/all_processed.npz \
    $INPUT_DIR2/S6/S6_HT10/all_processed.npz \
    $INPUT_DIR2/S15/S15_HT2/all_processed.npz \
    $INPUT_DIR2/S15/S15_HT7/all_processed.npz \
    $INPUT_DIR2/S15/S15_HT10/all_processed.npz \
    $INPUT_DIR2/S24/S24_HT2/all_processed.npz \
    $INPUT_DIR2/S24/S24_HT7/all_processed.npz \
    $INPUT_DIR2/S24/S24_HT10/all_processed.npz \
    $INPUT_DIR2/S27/S27_HT2/all_processed.npz \
    $INPUT_DIR2/S27/S27_HT7/all_processed.npz \
    $INPUT_DIR2/S27/S27_HT10/all_processed.npz \
    $INPUT_DIR2/S28/S28_HT2/all_processed.npz \
    $INPUT_DIR2/S28/S28_HT7/all_processed.npz \
    $INPUT_DIR2/S28/S28_HT10/all_processed.npz \
    $INPUT_DIR2/S1/S1_HT4/all_processed.npz \
    $INPUT_DIR2/S1/S1_HT8/all_processed.npz \
    $INPUT_DIR2/S1/S1_HT11/all_processed.npz \
    $INPUT_DIR2/S6/S6_HT4/all_processed.npz \
    $INPUT_DIR2/S6/S6_HT8/all_processed.npz \
    $INPUT_DIR2/S6/S6_HT11/all_processed.npz \
    $INPUT_DIR2/S15/S15_HT4/all_processed.npz \
    $INPUT_DIR2/S15/S15_HT8/all_processed.npz \
    $INPUT_DIR2/S15/S15_HT11/all_processed.npz \
    $INPUT_DIR2/S24/S24_HT4/all_processed.npz \
    $INPUT_DIR2/S24/S24_HT8/all_processed.npz \
    $INPUT_DIR2/S24/S24_HT11/all_processed.npz \
    $INPUT_DIR2/S27/S27_HT4/all_processed.npz \
    $INPUT_DIR2/S27/S27_HT8/all_processed.npz \
    $INPUT_DIR2/S27/S27_HT11/all_processed.npz \
    $INPUT_DIR2/S28/S28_HT4/all_processed.npz \
    $INPUT_DIR2/S28/S28_HT8/all_processed.npz \
    $INPUT_DIR2/S28/S28_HT11/all_processed.npz \
    $INPUT_DIR2/S1/S1_HT5/all_processed.npz \
    $INPUT_DIR2/S1/S1_HT9/all_processed.npz \
    $INPUT_DIR2/S1/S1_HT12/all_processed.npz \
    $INPUT_DIR2/S6/S6_HT5/all_processed.npz \
    $INPUT_DIR2/S6/S6_HT9/all_processed.npz \
    $INPUT_DIR2/S6/S6_HT12/all_processed.npz \
    $INPUT_DIR2/S15/S15_HT5/all_processed.npz \
    $INPUT_DIR2/S15/S15_HT9/all_processed.npz \
    $INPUT_DIR2/S15/S15_HT12/all_processed.npz \
    $INPUT_DIR2/S24/S24_HT5/all_processed.npz \
    $INPUT_DIR2/S24/S24_HT9/all_processed.npz \
    $INPUT_DIR2/S24/S24_HT12/all_processed.npz \
    $INPUT_DIR2/S27/S27_HT5/all_processed.npz \
    $INPUT_DIR2/S27/S27_HT9/all_processed.npz \
    $INPUT_DIR2/S27/S27_HT12/all_processed.npz \
    $INPUT_DIR2/S28/S28_HT5/all_processed.npz \
    $INPUT_DIR2/S28/S28_HT9/all_processed.npz \
    $INPUT_DIR2/S28/S28_HT12/all_processed.npz \
    $INPUT_DIR3/HT1.1/all_processed.npz \
    $INPUT_DIR3/HT1.2/all_processed.npz \
    $INPUT_DIR3/HT1.3/all_processed.npz \
    $INPUT_DIR4/Water_1/all_processed.npz \
    $INPUT_DIR4/Water_2/all_processed.npz \
    $INPUT_DIR4/Water_3/all_processed.npz \
    $INPUT_DIR4/Water_4/all_processed.npz \
    $INPUT_DIR4/Water_5/all_processed.npz \
    $INPUT_DIR4/Water_6/all_processed.npz \
    $INPUT_DIR4/Water_7/all_processed.npz \
    $INPUT_DIR4/Water_8/all_processed.npz \
    $INPUT_DIR4/Water_9/all_processed.npz \
    $INPUT_DIR4/Water_10/all_processed.npz \
    $INPUT_DIR4/Water_11/all_processed.npz \
    $INPUT_DIR4/Water_12/all_processed.npz \
    --val_frac 0.2 \
    --seed 42 \
    --checkpoint /pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/ \
    --epochs 200 \
    --outdir /pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/ \
    -n 10000 \
    --accelerator "gpu" \
    --strategy "ddp" \
    --devices 4 \
    --num_nodes 2 \
    --stop_wandb \

# Get the best checkpoint
best_ckpt=$(ls /pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/ | \
    grep 'checkpoint-' | \
    sort -t '-' -k3,3 -g | \
    tail -n 1)

best_ckpt_path="/pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/$best_ckpt"

# Run the prediction step using the best checkpoint
# lmcontrol infer-byol \
#     --prediction \
#     $INPUT_DIR/S4/S4_HT1/all_processed.npz \
#     $INPUT_DIR/S4/S4_HT4/all_processed.npz \
#     $INPUT_DIR/S4/S4_HT7/all_processed.npz \
#     $INPUT_DIR/S4/S4_HT10/all_processed.npz \
#     $INPUT_DIR/S10/S10_HT1/all_processed.npz \
#     $INPUT_DIR/S10/S10_HT4/all_processed.npz \
#     $INPUT_DIR/S10/S10_HT7/all_processed.npz \
#     $INPUT_DIR/S10/S10_HT10/all_processed.npz \
#     $INPUT_DIR/S14/S14_HT1/all_processed.npz \
#     $INPUT_DIR/S14/S14_HT4/all_processed.npz \
#     $INPUT_DIR/S14/S14_HT7/all_processed.npz \
#     $INPUT_DIR/S14/S14_HT10/all_processed.npz \
#     $INPUT_DIR/S4/S4_HT2/all_processed.npz \
#     $INPUT_DIR/S4/S4_HT5/all_processed.npz \
#     $INPUT_DIR/S4/S4_HT8/all_processed.npz \
#     $INPUT_DIR/S4/S4_HT11/all_processed.npz \
#     $INPUT_DIR/S10/S10_HT2/all_processed.npz \
#     $INPUT_DIR/S10/S10_HT5/all_processed.npz \
#     $INPUT_DIR/S10/S10_HT8/all_processed.npz \
#     $INPUT_DIR/S10/S10_HT11/all_processed.npz \
#     $INPUT_DIR/S14/S14_HT2/all_processed.npz \
#     $INPUT_DIR/S14/S14_HT5/all_processed.npz \
#     $INPUT_DIR/S14/S14_HT8/all_processed.npz \
#     $INPUT_DIR/S14/S14_HT11/all_processed.npz \
#     $INPUT_DIR/S4/S4_HT3/all_processed.npz \
#     $INPUT_DIR/S4/S4_HT6/all_processed.npz \
#     $INPUT_DIR/S4/S4_HT9/all_processed.npz \
#     $INPUT_DIR/S4/S4_HT12/all_processed.npz \
#     $INPUT_DIR/S10/S10_HT3/all_processed.npz \
#     $INPUT_DIR/S10/S10_HT6/all_processed.npz \
#     $INPUT_DIR/S10/S10_HT9/all_processed.npz \
#     $INPUT_DIR/S10/S10_HT12/all_processed.npz \
#     $INPUT_DIR/S14/S14_HT3/all_processed.npz \
#     $INPUT_DIR/S14/S14_HT6/all_processed.npz \
#     $INPUT_DIR/S14/S14_HT9/all_processed.npz \
#     $INPUT_DIR/S14/S14_HT12/all_processed.npz \
#     $INPUT_DIR1/S6/S6_HT1/all_processed.npz \
#     $INPUT_DIR1/S6/S6_HT3/all_processed.npz \
#     $INPUT_DIR1/S6/S6_HT5/all_processed.npz \
#     $INPUT_DIR1/S6/S6_HT7/all_processed.npz \
#     $INPUT_DIR1/S6/S6_HT9/all_processed.npz \
#     $INPUT_DIR1/S6/S6_HT11/all_processed.npz \
#     $INPUT_DIR1/S13/S13_HT1/all_processed.npz \
#     $INPUT_DIR1/S13/S13_HT3/all_processed.npz \
#     $INPUT_DIR1/S13/S13_HT5/all_processed.npz \
#     $INPUT_DIR1/S13/S13_HT7/all_processed.npz \
#     $INPUT_DIR1/S13/S13_HT9/all_processed.npz \
#     $INPUT_DIR1/S13/S13_HT11/all_processed.npz \
#     $INPUT_DIR1/S17/S17_HT1/all_processed.npz \
#     $INPUT_DIR1/S17/S17_HT3/all_processed.npz \
#     $INPUT_DIR1/S17/S17_HT5/all_processed.npz \
#     $INPUT_DIR1/S17/S17_HT7/all_processed.npz \
#     $INPUT_DIR1/S17/S17_HT9/all_processed.npz \
#     $INPUT_DIR1/S17/S17_HT11/all_processed.npz \
#     $INPUT_DIR1/S6/S6_HT2/all_processed.npz \
#     $INPUT_DIR1/S6/S6_HT4/all_processed.npz \
#     $INPUT_DIR1/S6/S6_HT6/all_processed.npz \
#     $INPUT_DIR1/S6/S6_HT8/all_processed.npz \
#     $INPUT_DIR1/S6/S6_HT10/all_processed.npz \
#     $INPUT_DIR1/S6/S6_HT12/all_processed.npz \
#     $INPUT_DIR1/S13/S13_HT2/all_processed.npz \
#     $INPUT_DIR1/S13/S13_HT4/all_processed.npz \
#     $INPUT_DIR1/S13/S13_HT6/all_processed.npz \
#     $INPUT_DIR1/S13/S13_HT8/all_processed.npz \
#     $INPUT_DIR1/S13/S13_HT10/all_processed.npz \
#     $INPUT_DIR1/S13/S13_HT12/all_processed.npz \
#     $INPUT_DIR1/S17/S17_HT2/all_processed.npz \
#     $INPUT_DIR1/S17/S17_HT4/all_processed.npz \
#     $INPUT_DIR1/S17/S17_HT6/all_processed.npz \
#     $INPUT_DIR1/S17/S17_HT8/all_processed.npz \
#     $INPUT_DIR1/S17/S17_HT10/all_processed.npz \
#     $INPUT_DIR1/S17/S17_HT12/all_processed.npz \
#     $INPUT_DIR2/S1/S1_HT2/all_processed.npz \
#     $INPUT_DIR2/S1/S1_HT7/all_processed.npz \
#     $INPUT_DIR2/S1/S1_HT10/all_processed.npz \
#     $INPUT_DIR2/S6/S6_HT2/all_processed.npz \
#     $INPUT_DIR2/S6/S6_HT7/all_processed.npz \
#     $INPUT_DIR2/S6/S6_HT10/all_processed.npz \
#     $INPUT_DIR2/S15/S15_HT2/all_processed.npz \
#     $INPUT_DIR2/S15/S15_HT7/all_processed.npz \
#     $INPUT_DIR2/S15/S15_HT10/all_processed.npz \
#     $INPUT_DIR2/S24/S24_HT2/all_processed.npz \
#     $INPUT_DIR2/S24/S24_HT7/all_processed.npz \
#     $INPUT_DIR2/S24/S24_HT10/all_processed.npz \
#     $INPUT_DIR2/S27/S27_HT2/all_processed.npz \
#     $INPUT_DIR2/S27/S27_HT7/all_processed.npz \
#     $INPUT_DIR2/S27/S27_HT10/all_processed.npz \
#     $INPUT_DIR2/S28/S28_HT2/all_processed.npz \
#     $INPUT_DIR2/S28/S28_HT7/all_processed.npz \
#     $INPUT_DIR2/S28/S28_HT10/all_processed.npz \
#     $INPUT_DIR2/S1/S1_HT4/all_processed.npz \
#     $INPUT_DIR2/S1/S1_HT8/all_processed.npz \
#     $INPUT_DIR2/S1/S1_HT11/all_processed.npz \
#     $INPUT_DIR2/S6/S6_HT4/all_processed.npz \
#     $INPUT_DIR2/S6/S6_HT8/all_processed.npz \
#     $INPUT_DIR2/S6/S6_HT11/all_processed.npz \
#     $INPUT_DIR2/S15/S15_HT4/all_processed.npz \
#     $INPUT_DIR2/S15/S15_HT8/all_processed.npz \
#     $INPUT_DIR2/S15/S15_HT11/all_processed.npz \
#     $INPUT_DIR2/S24/S24_HT4/all_processed.npz \
#     $INPUT_DIR2/S24/S24_HT8/all_processed.npz \
#     $INPUT_DIR2/S24/S24_HT11/all_processed.npz \
#     $INPUT_DIR2/S27/S27_HT4/all_processed.npz \
#     $INPUT_DIR2/S27/S27_HT8/all_processed.npz \
#     $INPUT_DIR2/S27/S27_HT11/all_processed.npz \
#     $INPUT_DIR2/S28/S28_HT4/all_processed.npz \
#     $INPUT_DIR2/S28/S28_HT8/all_processed.npz \
#     $INPUT_DIR2/S28/S28_HT11/all_processed.npz \
#     $INPUT_DIR2/S1/S1_HT5/all_processed.npz \
#     $INPUT_DIR2/S1/S1_HT9/all_processed.npz \
#     $INPUT_DIR2/S1/S1_HT12/all_processed.npz \
#     $INPUT_DIR2/S6/S6_HT5/all_processed.npz \
#     $INPUT_DIR2/S6/S6_HT9/all_processed.npz \
#     $INPUT_DIR2/S6/S6_HT12/all_processed.npz \
#     $INPUT_DIR2/S15/S15_HT5/all_processed.npz \
#     $INPUT_DIR2/S15/S15_HT9/all_processed.npz \
#     $INPUT_DIR2/S15/S15_HT12/all_processed.npz \
#     $INPUT_DIR2/S24/S24_HT5/all_processed.npz \
#     $INPUT_DIR2/S24/S24_HT9/all_processed.npz \
#     $INPUT_DIR2/S24/S24_HT12/all_processed.npz \
#     $INPUT_DIR2/S27/S27_HT5/all_processed.npz \
#     $INPUT_DIR2/S27/S27_HT9/all_processed.npz \
#     $INPUT_DIR2/S27/S27_HT12/all_processed.npz \
#     $INPUT_DIR2/S28/S28_HT5/all_processed.npz \
#     $INPUT_DIR2/S28/S28_HT9/all_processed.npz \
#     $INPUT_DIR2/S28/S28_HT12/all_processed.npz \
#     $INPUT_DIR3/HT1.1/all_processed.npz \
#     $INPUT_DIR3/HT1.2/all_processed.npz \
#     $INPUT_DIR3/HT1.3/all_processed.npz \
#     $INPUT_DIR4/Water_1/all_processed.npz \
#     $INPUT_DIR4/Water_2/all_processed.npz \
#     $INPUT_DIR4/Water_3/all_processed.npz \
#     $INPUT_DIR4/Water_4/all_processed.npz \
#     $INPUT_DIR4/Water_5/all_processed.npz \
#     $INPUT_DIR4/Water_6/all_processed.npz \
#     $INPUT_DIR4/Water_7/all_processed.npz \
#     $INPUT_DIR4/Water_8/all_processed.npz \
#     $INPUT_DIR4/Water_9/all_processed.npz \
#     $INPUT_DIR4/Water_10/all_processed.npz \
#     $INPUT_DIR4/Water_11/all_processed.npz \
#     $INPUT_DIR4/Water_12/all_processed.npz \
#     --checkpoint $best_ckpt_path \
#     -o /pscratch/sd/n/niranjan/output/prediction_${task_name}.npz \
#     -n 10000 \
#     # -save_emb \
#     # --save_misclassified /pscratch/sd/n/niranjan/output/optatune/ambr01_misclassify/misclassified_${task_name} \
#     # --save_confusion /pscratch/sd/n/niranjan/output/optatune/ambr01_misclassify/


