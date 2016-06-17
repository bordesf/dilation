# Convert caffe models of Multi-Scale Context Aggregation by Dilated Convolutions
# into lasagne model.
# TO generate the pkl files, you need first to dowonload the caffe models
# with the scripts in pretrained folder and then execute this file.

import argparse
import json
import caffe
import numpy as np
from os.path import dirname, exists, join
import theano.tensor as T

# Import dilated cnn lasagne model
from dilated_cnn import build_model
from lasagne.layers import DilatedConv2DLayer as DilatedConvLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import get_all_param_values
try:
   import cPickle as pickle
except:
   import pickle


class Dataset(object):
    def __init__(self, dataset_name):
        self.work_dir = dirname(__file__)
        info_path = join(self.work_dir, 'datasets', dataset_name + '.json')
        if not exists(info_path):
            raise IOError("Do not have information for dataset {}"
                          .format(dataset_name))
        with open(info_path, 'r') as fp:
            info = json.load(fp)
        self.palette = np.array(info['palette'], dtype=np.uint8)
        self.mean_pixel = np.array(info['mean'], dtype=np.float32)
        self.dilation = info['dilation']
        self.zoom = info['zoom']
        self.name = dataset_name
        self.model_name = 'dilation{}_{}'.format(self.dilation, self.name)
        self.model_path = join(self.work_dir, 'models',
                               self.model_name + '_deploy.prototxt')

    @property
    def pretrained_path(self):
        p = join(dirname(__file__), 'pretrained',
                 self.model_name + '.caffemodel')
        if not exists(p):
            download_path = join(self.work_dir, 'pretrained',
                                 'download_{}.sh'.format(self.name))
            raise IOError('Pleaes run {} to download the pretrained network '
                          'weights first'.format(download_path))
        return p


# Load parameters of caffe into the lasagne model
def load_caffe_model(net_lasagne, net_caffe):
    layers_caffe = dict(zip(list(net_caffe._layer_names), net_caffe.layers))
    for name, layer in net_lasagne.items():
        try:
            if isinstance(layer, ConvLayer) or isinstance(layer, DilatedConvLayer):
                W = layers_caffe[name].blobs[0].data
                if isinstance(layer, DilatedConvLayer):
                    W = W.transpose(1, 0, 2, 3)
                assert W.shape == layer.W.get_value().shape
                layer.W.set_value(W)
                b = layers_caffe[name].blobs[1].data
                assert b.shape == layer.b.get_value().shape
                layer.b.set_value(b)
            else:
                layer.W.set_value(layers_caffe[name].blobs[0].data)
                layer.b.set_value(layers_caffe[name].blobs[1].data)
        except AttributeError:
            continue


# Convert caffe model to lasagne
def convert(dataset_name):
    dataset = Dataset(dataset_name)

    # Create theano graph
    input_var = T.tensor4('input')
    net = build_model(input_var)

    # Load caffe model
    net_caffe = caffe.Net(dataset.model_path, dataset.pretrained_path, caffe.TEST)

    # Set the parameters from caffe into lasagne
    load_caffe_model(net, net_caffe)

    # Save the parameters
    p = join(dirname(__file__), 'pretrained', dataset.model_name + '.pkl')
    output = open(p, 'wb')
    params = get_all_param_values(net['prob'])
    pickle.dump(params, output)
    output.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?',
                        choices=['pascal_voc', 'camvid', 'kitti', 'cityscapes'])
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID to run CAFFE. '
                             'If -1 (default), CPU is used')
    args = parser.parse_args()
    if args.gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)
        print('Using GPU ', args.gpu)
    else:
        caffe.set_mode_cpu()
        print('Using CPU')
    convert(args.dataset)

if __name__ == '__main__':
    main()
