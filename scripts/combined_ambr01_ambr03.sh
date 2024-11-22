#!/bin/bash
#SBATCH -A m3513
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -n 1
#SBATCH --time=1:30:00
#SBATCH --array=0  # Only one job in the array (since we're only using one name)
#SBATCH -e /pscratch/sd/n/niranjan/error/clf_error_rep/opta_1296_combis/optatune.%A_combined_ambr01_ambr03_40e.err
#SBATCH -o /pscratch/sd/n/niranjan/error/clf_error_rep/opta_1296_combis/optatune.%A_combined_ambr01_ambr03_40e.out
#SBATCH -J combined_ambr01_ambr03_40e  # Job name

# Assign custom name (SLURM_ARRAY_TASK_ID is replaced with the custom name)
task_name="combined_ambr01_ambr03_40e"
INPUT_DIR="$SCRATCH/tar_ball/segmented_square_96"
INPUT_DIR1="$SCRATCH/tar_ball/ambr_03/"
INPUT_DIR2="/pscratch/sd/n/niranjan/tar_ball/ABF_FA_AMBR01-NSR"

# # Run the script with the task_name
# lmcontrol sup-train \
#     --training \
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
#     --val_frac 0.2 \
#     --seed 42 \
#     --checkpoint /pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/ \
#     --epochs 40 \
#     --outdir /pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/ \
#     -n 5000 \
#     testing0 \
#     --time_weight 1 \
#     time \

# Get the best checkpoint
best_ckpt=$(ls /pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/ | \
    grep 'checkpoint-' | \
    sort -t '-' -k3,3 -g | \
    tail -n 1)

best_ckpt_path="/pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/$best_ckpt"

# Run the prediction step using the best checkpoint
lmcontrol sup-predict \
    --prediction \
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
    --checkpoint $best_ckpt_path \
    -o /pscratch/sd/n/niranjan/output/prediction_${task_name}.npz \
    -n 5000 \
    time \
    #--return_embeddings \
    # --save_misclassified /pscratch/sd/n/niranjan/output/optatune/ambr01_misclassify/misclassified_${task_name} \
    # --save_confusion /pscratch/sd/n/niranjan/output/optatune/ambr01_misclassify/


    ### NEW ##: CHANGE THIS ##

    # $INPUT_DIR/S4/S4_HT3/all_processed.npz \
    # $INPUT_DIR/S10/S10_HT6/all_processed.npz \
    # $INPUT_DIR/S14/S14_HT9/all_processed.npz \
    # $INPUT_DIR1/S6/S6_HT2/all_processed.npz \
    # $INPUT_DIR1/S13/S13_HT4/all_processed.npz \
    # $INPUT_DIR1/S17/S17_HT6/all_processed.npz \

    # $INPUT_DIR/S4/S4_HT3/all_processed.npz \
    # $INPUT_DIR/S4/S4_HT6/all_processed.npz \
    # $INPUT_DIR/S4/S4_HT9/all_processed.npz \
    # $INPUT_DIR/S4/S4_HT12/all_processed.npz \
    # $INPUT_DIR/S10/S10_HT3/all_processed.npz \
    # $INPUT_DIR/S10/S10_HT6/all_processed.npz \
    # $INPUT_DIR/S10/S10_HT9/all_processed.npz \
    # $INPUT_DIR/S10/S10_HT12/all_processed.npz \
    # $INPUT_DIR/S14/S14_HT3/all_processed.npz \
    # $INPUT_DIR/S14/S14_HT6/all_processed.npz \
    # $INPUT_DIR/S14/S14_HT9/all_processed.npz \
    # $INPUT_DIR/S14/S14_HT12/all_processed.npz \
    # $INPUT_DIR1/S6/S6_HT2/all_processed.npz \
    # $INPUT_DIR1/S6/S6_HT4/all_processed.npz \
    # $INPUT_DIR1/S6/S6_HT6/all_processed.npz \
    # $INPUT_DIR1/S6/S6_HT8/all_processed.npz \
    # $INPUT_DIR1/S6/S6_HT10/all_processed.npz \
    # $INPUT_DIR1/S6/S6_HT12/all_processed.npz \
    # $INPUT_DIR1/S13/S13_HT2/all_processed.npz \
    # $INPUT_DIR1/S13/S13_HT4/all_processed.npz \
    # $INPUT_DIR1/S13/S13_HT6/all_processed.npz \
    # $INPUT_DIR1/S13/S13_HT8/all_processed.npz \
    # $INPUT_DIR1/S13/S13_HT10/all_processed.npz \
    # $INPUT_DIR1/S13/S13_HT12/all_processed.npz \
    # $INPUT_DIR1/S17/S17_HT2/all_processed.npz \
    # $INPUT_DIR1/S17/S17_HT4/all_processed.npz \
    # $INPUT_DIR1/S17/S17_HT6/all_processed.npz \
    # $INPUT_DIR1/S17/S17_HT8/all_processed.npz \
    # $INPUT_DIR1/S17/S17_HT10/all_processed.npz \
    # $INPUT_DIR1/S17/S17_HT12/all_processed.npz \