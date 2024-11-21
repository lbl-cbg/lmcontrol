task_name="byol_delete"
INPUT_DIR="$SCRATCH/tar_ball/segmented_square_96"

lmcontrol prep-viz \
    /pscratch/sd/n/niranjan/output/prep-viz_${task_name}.npz \
    /pscratch/sd/n/niranjan/output/prediction_${task_name}.npz \
    --two-dim \
    --center \