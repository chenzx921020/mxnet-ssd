#!/bin/bash

python tools/prepare_dataset.py --dataset pascal --set train_chenzx --target /home/users/zhixuan.chen/project/mxnet-ssd/data/train.lst --root /data/zhixuan/out_dataset/json2txt.txt

python mxnet/tools/im2rec.py ./data/train.lst /data/zhixuan/out_dataset/ --shuffle 1 --label-pack 1
