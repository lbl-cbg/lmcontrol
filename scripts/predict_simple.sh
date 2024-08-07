lmcontrol predict-clf \
    --prediction \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT3/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT6/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT9/all_processed.npz \
    -c /pscratch/sd/n/niranjan/output/checkpoint/model_simple.ckpt \
    -o /pscratch/sd/n/niranjan/output/prediction.npz \
    --pred-only \
    -n 2000 \
    time

#NOTE: I HAVE REMOVED :    --pred-only \






    



    #    /pscratch/sd/n/niranjan/output/checkpoint/model.ckpt \
