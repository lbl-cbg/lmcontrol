#!/bin/bash
#SBATCH -A m3513
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -n 1
#SBATCH --time=2:30:00
#SBATCH --array=0  # Only one job in the array (since we're only using one name)
#SBATCH -e /pscratch/sd/n/niranjan/error/clf_error_rep/opta_1296_combis/optatune.%A_multilabels_1e-3_40e_ambr03.err
#SBATCH -o /pscratch/sd/n/niranjan/error/clf_error_rep/opta_1296_combis/optatune.%A_multilabels_1e-3_40e_ambr03.out
#SBATCH -J multilabels_1e-3_40e_ambr03. # Job name

# Assign custom name (SLURM_ARRAY_TASK_ID is replaced with the custom name)
task_name="multilabels_1e-3_40e_ambr03."

# SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-10_15_03_00}
INPUT_DIR="$SCRATCH/tar_ball/ambr_03/"

lmcontrol sup-train \
    --training \
    $INPUT_DIR/S6/S6_HT1/all_processed.npz \
    $INPUT_DIR/S6/S6_HT3/all_processed.npz \
    $INPUT_DIR/S6/S6_HT5/all_processed.npz \
    $INPUT_DIR/S6/S6_HT7/all_processed.npz \
    $INPUT_DIR/S6/S6_HT9/all_processed.npz \
    $INPUT_DIR/S6/S6_HT11/all_processed.npz \
    $INPUT_DIR/S13/S13_HT1/all_processed.npz \
    $INPUT_DIR/S13/S13_HT3/all_processed.npz \
    $INPUT_DIR/S13/S13_HT5/all_processed.npz \
    $INPUT_DIR/S13/S13_HT7/all_processed.npz \
    $INPUT_DIR/S13/S13_HT9/all_processed.npz \
    $INPUT_DIR/S13/S13_HT11/all_processed.npz \
    $INPUT_DIR/S17/S17_HT1/all_processed.npz \
    $INPUT_DIR/S17/S17_HT3/all_processed.npz \
    $INPUT_DIR/S17/S17_HT5/all_processed.npz \
    $INPUT_DIR/S17/S17_HT7/all_processed.npz \
    $INPUT_DIR/S17/S17_HT9/all_processed.npz \
    $INPUT_DIR/S17/S17_HT11/all_processed.npz \
    --val_frac 0.2 \
    --seed 42 \
    --checkpoint $SCRATCH/output/optatune/ambr03/opta_${task_name}/ \
    --epochs 40 \
    --outdir $SCRATCH/output/optatune/ambr03/opta_${task_name}/ \
    -n 9500 \
    testing0 \
    --stop_wandb \
    --time_weight 1e-3 \
    feed \
    time \

best_ckpt=$(ls $SCRATCH/output/optatune/ambr03/opta_${task_name}/ | \
    grep 'checkpoint-' | \
    sort -t '-' -k3,3 -g | \
    tail -n 1)

best_ckpt_path="$SCRATCH/output/optatune/ambr03/opta_${task_name}/$best_ckpt"


lmcontrol sup-predict \
    --prediction \
    $INPUT_DIR/S6/S6_HT2/all_processed.npz \
    $INPUT_DIR/S6/S6_HT4/all_processed.npz \
    $INPUT_DIR/S6/S6_HT6/all_processed.npz \
    $INPUT_DIR/S6/S6_HT8/all_processed.npz \
    $INPUT_DIR/S6/S6_HT10/all_processed.npz \
    $INPUT_DIR/S6/S6_HT12/all_processed.npz \
    $INPUT_DIR/S13/S13_HT2/all_processed.npz \
    $INPUT_DIR/S13/S13_HT4/all_processed.npz \
    $INPUT_DIR/S13/S13_HT6/all_processed.npz \
    $INPUT_DIR/S13/S13_HT8/all_processed.npz \
    $INPUT_DIR/S13/S13_HT10/all_processed.npz \
    $INPUT_DIR/S13/S13_HT12/all_processed.npz \
    $INPUT_DIR/S17/S17_HT2/all_processed.npz \
    $INPUT_DIR/S17/S17_HT4/all_processed.npz \
    $INPUT_DIR/S17/S17_HT6/all_processed.npz \
    $INPUT_DIR/S17/S17_HT8/all_processed.npz \
    $INPUT_DIR/S17/S17_HT10/all_processed.npz \
    $INPUT_DIR/S17/S17_HT12/all_processed.npz \
    --checkpoint $best_ckpt_path \
    -o $SCRATCH/output/ambr03/prediction_${task_name}.npz \
    -n 9500 \
    feed \
    time \
    # --save_confusion /pscratch/sd/n/niranjan/output/optatune/ambr03_misclassify/ \
    # --save_misclassified /pscratch/sd/n/niranjan/output/optatune/ambr03_misclassify/misclassified_${SLURM_ARRAY_TASK_ID} \
   
# Here time is expected to below -n in any case

cp /pscratch/sd/n/niranjan/output/ambr03/prediction_${task_name}.npz /global/cfs/cdirs/m3513/sdbr/lmcontrol/bisabolene_CN/sup/multimodal/
