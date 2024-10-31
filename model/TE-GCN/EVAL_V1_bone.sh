#!/bin/bash

# RECORD=2996


# RECORD=res_testb_joint_enhance
RECORD=res_testb_bone_enhance
WORKDIR=work_dir/$RECORD
MODELNAME=runs/$RECORD

CONFIG=./config/uav-cross-subjectv1/test_bone.yaml

# WEIGHTS=runs/2101-43-3608.pt

# bone 69.45
# WEIGHTS=/media/sdd/robot/TE-GCN/runs/2103_bone-48-6272.pt

# enhance joint 70.75
# WEIGHTS=/media/sdd/robot/TE-GCN/runs/2104_dataprocessv1-47-3984.pt

# enhance bone 68.45
WEIGHTS=/media/sdd/robot/TE-GCN/runs/2105_bone_dataprocessv1-42-11008.pt


BATCH_SIZE=256
python3  main.py  --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0  --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --weights $WEIGHTS
# python3  main.py  --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 1 2 3 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --weights $WEIGHTS

# python3 -m debugpy --listen 5678 --wait-for-client main.py  --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 1 2 3 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --weights $WEIGHTS
