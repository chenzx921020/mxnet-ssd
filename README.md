# SSD: Single Shot MultiBox Object Detector

SSD is an unified framework for object detection with a single network.  

You can use the code to train/evaluate/test for your own object detection task.  

### Disclaimer
This is a re-implementation of original SSD which is based on caffe. The official  
repository is available [here](https://github.com/weiliu89/caffe/tree/ssd).  
The arXiv paper is available [here](http://arxiv.org/abs/1512.02325).  

Compared with official version in mxnet, this repo provides the process of sample generating,training and forward predicting and continuously updating  

* Basic preparation:
Before bellow steps, you need to build the directory model and save the pretrained model in it  
```
# mkdir model
# cd model
hadoop fs -get hdfs://hobot-bigdata/user/chenzhixuan/ssd_pretrained_model/vgg16-reduced-*
```

* Sample generating:
In `dataset/pascal_voc.py`, we have modified the data import ways. Firstly, read the txt data and output lst file, and secondly, compress lst file into rec file with mxnet tools `im2rec.py`  
The txt data format: img_url lefttop_x lefttop_y rightbottom_x rightbottom_y
```
sh prepare_dataset.sh
```
The `--root` is the source txt, the `--target` is the target lst, you can modify them by your convenience.  
In im2rec process, you need provide the source image path and lst file above.  

* Start training:
```
python train.py
```
* By default, this example will use `batch-size=32` and `learning_rate=0.004`.
You might need to change the parameters a bit if you have different configurations.  
Check `python train.py --help` for more training options. For example, if you have 4 GPUs, use:  
```
# note that a perfect training parameter set is yet to be discovered for multi-gpu
python train.py --gpus 0,1,2,3 --batch-size 128 --lr 0.001
```
* Attention memory: training `ssd300` in `vgg16-reduced` will comsume at least 2 gpus, and `ssd500` comsuming 24g is not provided.  


### Convert model to deploy mode
This simply removes all loss layers, and attach a layer for merging results and non-maximum suppression.  
Rename the model and symbol to `ssd_vgg16_reduced_300-symbol.json` and `ssd_vgg16_reduced_300-000x.params`. The x is up to epochs.  

```
python deploy.py --prefix ssd_vgg_reduced_300 --epoch x --num-class 2 
```
After steps above, in path `./model` will generate the two files with prefix `deploy`  
```
python demo.py --image 3304.png --prefix model/deploy_ssd_vgg16_reduced_300 --epoch x --deploy --data-shape 300
```
The data-shape must be consistent with training input size  

