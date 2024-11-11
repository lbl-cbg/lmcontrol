
OPTS="--crop 96,96 --save-unseg --no-tifs"
INPUTDIR="/pscratch/sd/n/niranjan/tar_ball/ambr_03"
OUTDIR="/pscratch/sd/n/niranjan/tar_ball/ambr_03/S17"

lmcontrol crop $OPTS -m "ht=1,time=S17,feed=C:N 6,starting_media=C:N 6" $INPUTDIR/S17_Group_1-4/S17_HT1 $OUTDIR/S17_HT1
lmcontrol crop $OPTS -m "ht=2,time=S17,feed=C:N 6,starting_media=C:N 6" $INPUTDIR/S17_Group_1-4/S17_HT2 $OUTDIR/S17_HT2
lmcontrol crop $OPTS -m "ht=3,time=S17,feed=C:N 15,starting_media=C:N 6" $INPUTDIR/S17_Group_1-4/S17_HT3 $OUTDIR/S17_HT3
lmcontrol crop $OPTS -m "ht=4,time=S17,feed=C:N 15,starting_media=C:N 6" $INPUTDIR/S17_Group_1-4/S17_HT4 $OUTDIR/S17_HT4
lmcontrol crop $OPTS -m "ht=5,time=S17,feed=C:N 30,starting_media=C:N 6" $INPUTDIR/S17_Group_5-8/S17_HT5 $OUTDIR/S17_HT5
lmcontrol crop $OPTS -m "ht=6,time=S17,feed=C:N 30,starting_media=C:N 6" $INPUTDIR/S17_Group_5-8/S17_HT6 $OUTDIR/S17_HT6
lmcontrol crop $OPTS -m "ht=7,time=S17,feed=C:N 6,starting_media=C:N 30" $INPUTDIR/S17_Group_5-8/S17_HT7 $OUTDIR/S17_HT7
lmcontrol crop $OPTS -m "ht=8,time=S17,feed=C:N 6,starting_media=C:N 30" $INPUTDIR/S17_Group_5-8/S17_HT8 $OUTDIR/S17_HT8
lmcontrol crop $OPTS -m "ht=9,time=S17,feed=C:N 15,starting_media=C:N 30" $INPUTDIR/S17_Group_9-12/S17_HT9 $OUTDIR/S17_HT9
lmcontrol crop $OPTS -m "ht=10,time=S17,feed=C:N 15,starting_media=C:N 30" $INPUTDIR/S17_Group_9-12/S17_HT10 $OUTDIR/S17_HT10
lmcontrol crop $OPTS -m "ht=11,time=S17,feed=C:N 30,starting_media=C:N 30" $INPUTDIR/S17_Group_9-12/S17_HT11 $OUTDIR/S17_HT11
lmcontrol crop $OPTS -m "ht=12,time=S17,feed=C:N 30,starting_media=C:N 30" $INPUTDIR/S17_Group_9-12/S17_HT12 $OUTDIR/S17_HT12

OPTS="--crop 96,96 --save-unseg --no-tifs"
INPUTDIR="/pscratch/sd/n/niranjan/tar_ball/ambr_03"
OUTDIR="/pscratch/sd/n/niranjan/tar_ball/ambr_03/S13"

lmcontrol crop $OPTS -m "ht=1,time=S13,feed=C:N 6,starting_media=C:N 6" $INPUTDIR/S13_Group_1-4/S13_HT1 $OUTDIR/S13_HT1
lmcontrol crop $OPTS -m "ht=2,time=S13,feed=C:N 6,starting_media=C:N 6" $INPUTDIR/S13_Group_1-4/S13_HT2 $OUTDIR/S13_HT2
lmcontrol crop $OPTS -m "ht=3,time=S13,feed=C:N 15,starting_media=C:N 6" $INPUTDIR/S13_Group_1-4/S13_HT3 $OUTDIR/S13_HT3
lmcontrol crop $OPTS -m "ht=4,time=S13,feed=C:N 15,starting_media=C:N 6" $INPUTDIR/S13_Group_1-4/S13_HT4 $OUTDIR/S13_HT4
lmcontrol crop $OPTS -m "ht=5,time=S13,feed=C:N 30,starting_media=C:N 6" $INPUTDIR/S13_Group_5-8/S13_HT5 $OUTDIR/S13_HT5
lmcontrol crop $OPTS -m "ht=6,time=S13,feed=C:N 30,starting_media=C:N 6" $INPUTDIR/S13_Group_5-8/S13_HT6 $OUTDIR/S13_HT6
lmcontrol crop $OPTS -m "ht=7,time=S13,feed=C:N 6,starting_media=C:N 30" $INPUTDIR/S13_Group_5-8/S13_HT7 $OUTDIR/S13_HT7
lmcontrol crop $OPTS -m "ht=8,time=S13,feed=C:N 6,starting_media=C:N 30" $INPUTDIR/S13_Group_5-8/S13_HT8 $OUTDIR/S13_HT8
lmcontrol crop $OPTS -m "ht=9,time=S13,feed=C:N 15,starting_media=C:N 30" $INPUTDIR/S13_Group_9-12/S13_HT9 $OUTDIR/S13_HT9
lmcontrol crop $OPTS -m "ht=10,time=S13,feed=C:N 15,starting_media=C:N 30" $INPUTDIR/S13_Group_9-12/S13_HT10 $OUTDIR/S13_HT10
lmcontrol crop $OPTS -m "ht=11,time=S13,feed=C:N 30,starting_media=C:N 30" $INPUTDIR/S13_Group_9-12/S13_HT11 $OUTDIR/S13_HT11
lmcontrol crop $OPTS -m "ht=12,time=S13,feed=C:N 30,starting_media=C:N 30" $INPUTDIR/S13_Group_9-12/S13_HT12 $OUTDIR/S13_HT12

OPTS="--crop 96,96 --save-unseg --no-tifs"
INPUTDIR="/pscratch/sd/n/niranjan/tar_ball/ambr_03"
OUTDIR="/pscratch/sd/n/niranjan/tar_ball/ambr_03/S6"

lmcontrol crop $OPTS -m "ht=1,time=S6,feed=C:N 6,starting_media=C:N 6" $INPUTDIR/S6_Group_1-4/S6_HT1 $OUTDIR/S6_HT1
lmcontrol crop $OPTS -m "ht=2,time=S6,feed=C:N 6,starting_media=C:N 6" $INPUTDIR/S6_Group_1-4/S6_HT2 $OUTDIR/S6_HT2
lmcontrol crop $OPTS -m "ht=3,time=S6,feed=C:N 15,starting_media=C:N 6" $INPUTDIR/S6_Group_1-4/S6_HT3 $OUTDIR/S6_HT3
lmcontrol crop $OPTS -m "ht=4,time=S6,feed=C:N 15,starting_media=C:N 6" $INPUTDIR/S6_Group_1-4/S6_HT4 $OUTDIR/S6_HT4
lmcontrol crop $OPTS -m "ht=5,time=S6,feed=C:N 30,starting_media=C:N 6" $INPUTDIR/S6_Group_5-8/S6_HT5 $OUTDIR/S6_HT5
lmcontrol crop $OPTS -m "ht=6,time=S6,feed=C:N 30,starting_media=C:N 6" $INPUTDIR/S6_Group_5-8/S6_HT6 $OUTDIR/S6_HT6
lmcontrol crop $OPTS -m "ht=7,time=S6,feed=C:N 6,starting_media=C:N 30" $INPUTDIR/S6_Group_5-8/S6_HT7 $OUTDIR/S6_HT7
lmcontrol crop $OPTS -m "ht=8,time=S6,feed=C:N 6,starting_media=C:N 30" $INPUTDIR/S6_Group_5-8/S6_HT8 $OUTDIR/S6_HT8
lmcontrol crop $OPTS -m "ht=9,time=S6,feed=C:N 15,starting_media=C:N 30" $INPUTDIR/S6_Group_9-12/S6_HT9 $OUTDIR/S6_HT9
lmcontrol crop $OPTS -m "ht=10,time=S6,feed=C:N 15,starting_media=C:N 30" $INPUTDIR/S6_Group_9-12/S6_HT10 $OUTDIR/S6_HT10
lmcontrol crop $OPTS -m "ht=11,time=S6,feed=C:N 30,starting_media=C:N 30" $INPUTDIR/S6_Group_9-12/S6_HT11 $OUTDIR/S6_HT11
lmcontrol crop $OPTS -m "ht=12,time=S6,feed=C:N 30,starting_media=C:N 30" $INPUTDIR/S6_Group_9-12/S6_HT12 $OUTDIR/S6_HT12
