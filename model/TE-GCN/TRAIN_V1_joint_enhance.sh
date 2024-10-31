#!/bin/bash
RECORD=2104_joint_dataprocessv1
WORKDIR=work_dir/$RECORD
MODELNAME=runs/$RECORD

# CONFIG=./config/uav-cross-subjectv1/train_joint_motion.yaml
CONFIG=/media/sdd/robot/TE-GCN/config/uav-cross-subjectv1/train_joint_enhance.yaml
# CONFIG=/media/sdd/robot/TE-GCN/config/uav-cross-subjectv1/train.yaml

START_EPOCH=50
EPOCH_NUM=80
BATCH_SIZE=64
WARM_UP=5
SEED=777

python3 main.py --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 1 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --warm_up_epoch $WARM_UP --only_train_epoch $EPOCH_NUM --seed $SEED

