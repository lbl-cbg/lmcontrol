task_name="byol_combined_200e"
INPUT_DIR="$SCRATCH/tar_ball/segmented_square_96"

lmcontrol emb-viz \
    /pscratch/sd/n/niranjan/output/viz/ \
    --subsample 0.1 \
    --label campaign \