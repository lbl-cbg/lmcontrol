lmcontrol infer-byol \
    -c /pscratch/sd/n/niranjan/output/checkpoint/epoch=9-step=15550.ckpt \
    --prediction \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT3/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT6/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT9/all_processed.npz \
    -o /pscratch/sd/n/niranjan/output/byol_predictions/prediction.npz \
    --mask \
    time 