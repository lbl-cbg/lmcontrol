task_name="byol_combined_200e"
INPUT_DIR="$SCRATCH/tar_ball/segmented_square_96"

lmcontrol prep-viz \
    /pscratch/sd/n/niranjan/output/prep-viz_3D_${task_name}.npz \
    /pscratch/sd/n/niranjan/output/prediction_${task_name}.npz \
    --center \