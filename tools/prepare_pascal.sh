#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
#python $DIR/prepare_dataset.py --dataset pascal --year 2007,2012 --set trainval --target $DIR/../data/train.lst
#python $DIR/prepare_dataset.py --dataset pascal --year 2007 --set test --target $DIR/../data/val.lst --shuffle False
python $DIR/prepare_dataset.py --dataset pascal --set train_chenzx --target $DIR/../data/train.lst --root /data/zhixuan/out_dataset/json2txt.txt
