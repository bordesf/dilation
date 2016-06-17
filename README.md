# Lasagne implementation of Multi-Scale Context Aggregation by Dilated Convolutions
The predict.py file can be used as the original one. It will use Lasagne instead of Caffe to perform
the segmentation. The lasagne model is in the file dilated_cnn.py.

To convert the caffe model into pkl file that you can load directly with lasagne, you can use
convert_to_pkl.py

When you get your pkl file, you can use directly predict_lasagne.py to run the demo.

predict.py use caffe to load the caffemodel and use lasagne to set up the model and make the
prediction.
predict_lasagne.py use only lasagne, so you need to convert the caffemodel into pkl before using it.

# Multi-Scale Context Aggregation by Dilated Convolutions

## Introduction

Properties of dilated convolution are discussed in our [ICLR 2016 conference paper](http://arxiv.org/abs/1511.07122). This repository contains the network definitions and the trained models. You can use this code together with vanilla Caffe to segment images using the pre-trained models. If you want to train the models yourself, use [this fork of Caffe](https://github.com/fyu/caffe-dilation).

### Citing

If you find the code or the models useful, please cite this paper:
```
@inproceedings{YuKoltun2016,
	author    = {Fisher Yu and Vladlen Koltun},
	title     = {Multi-Scale Context Aggregation by Dilated Convolutions},
	booktitle = {ICLR},
	year      = {2016},
}
```
### License

The code and models are released under the MIT License (refer to the LICENSE file for details).


## Installation
### Caffe

Install [Caffe](https://github.com/BVLC/caffe) and its Python interface. Make sure that the Caffe version is newer than commit [08c5df](https://github.com/BVLC/caffe/commit/08c5dfd53e6fd98148d6ce21e590407e38055984).

### Python

The companion Python script is used to demonstrate the network definition and trained weights.

The required Python packages are numba numpy opencv. Python release from Anaconda is recommended.

In the case of using Anaconda
```
conda install numba numpy opencv
```

## Running Demo

predict.py is the main script to test the pre-trained models on images. The basic usage is

    python predict.py <dataset name> <image path>

Given the dataset name, the script will find the pre-trained model and network definition. We currently support models trained from four datasets: pascal_voc, camvid, kitti, cityscapes. The steps of using the code is listed below:

* Clone the code from Github

    ```
    git clone git@github.com:fyu/dilation.git
    cd dilation
    ```
* Download pre-trained network

    ```
    sh pretrained/download_pascal_voc.sh
    ```
* Run pascal voc model on GPU 0

    ```
    python predict.py pascal_voc images/dog.jpg --gpu 0
    ```

## Implementation of Dilated Convolution

Besides Caffe support, dilated convolution is also implemented in other deep learning packages. For example,
* Torch: [SpatialDilatedConvolution](https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialDilatedConvolution)
* Lasagne: [DilatedConv2DLayer](http://lasagne.readthedocs.io/en/latest/modules/layers/conv.html?highlight=dilated#lasagne.layers.DilatedConv2DLayer)
