#!/bin/bash

FRAMES_DIR=frames
FILE_LIST=frames.lst
TRAIN_LIST=train.lst
TEST_LIST=test.lst

# download videos and generate frames
cd dataset
./downloadmulticategoryvideos.sh 100 selectedcategories.txt
./generateframesfromvideos.sh videos $FRAMES_DIR

# create file list for training and testing
find $FRAMES_DIR -name '*.png' >$FILE_LIST

python3 ../pex/dataset.py $FILE_LIST $TRAIN_LIST $TEST_LIST

# training
cd ..
mkdir ckpt
python3 -m pex.train --train_list dataset/$TRAIN_LIST \
  --test_list dataset/$TEST_LIST \
  --dataset_root dataset \
  --checkpoint_root ckpt \
  --model_name pex1

