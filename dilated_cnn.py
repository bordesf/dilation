# Model fron the paper Multi-Scale Context Aggregation by Dilated Convolutions
# Original Caffe source:  https://github.com/fyu/dilation

from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import InputLayer, DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import rectify as relu
from lasagne.nonlinearities import softmax, linear
from lasagne.layers import DilatedConv2DLayer as DilatedConvLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DimshuffleLayer, FlattenLayer


def build_model(input_var=None):
    net = {}

    # Create front-end network
    net['input'] = InputLayer((None, 3, None, None), input_var)
    net['conv1_1'] = ConvLayer(net['input'], num_filters=64, filter_size=3, pad=0, flip_filters=False, nonlinearity=relu)
    net['conv1_2'] = ConvLayer(net['conv1_1'], num_filters=64, filter_size=3, pad=0, flip_filters=False, nonlinearity=relu)
    net['pool1'] = PoolLayer(net['conv1_2'], pool_size=2, stride=2, mode='max', ignore_border=False)
    net['conv2_1'] = ConvLayer(net['pool1'], num_filters=128, filter_size=3, pad=0, flip_filters=False, nonlinearity=relu)
    net['conv2_2'] = ConvLayer(net['conv2_1'], num_filters = 128, filter_size=3, pad=0, flip_filters=False, nonlinearity=relu)
    net['pool2'] = PoolLayer(net['conv2_2'], pool_size=2, stride=2, mode='max', ignore_border=False)
    net['conv3_1'] = ConvLayer(net['pool2'], num_filters=256, filter_size=3, pad=0, flip_filters=False, nonlinearity=relu)
    net['conv3_2'] = ConvLayer(net['conv3_1'], num_filters=256, filter_size=3, pad=0, flip_filters=False, nonlinearity=relu)
    net['conv3_3'] = ConvLayer(net['conv3_2'], num_filters=256, filter_size=3, pad=0, flip_filters=False, nonlinearity=relu)
    net['pool3'] = PoolLayer(net['conv3_3'], pool_size=2, stride=2, mode='max', ignore_border=False)
    net['conv4_1'] = ConvLayer(net['pool3'], num_filters=512, filter_size=3, pad=0, flip_filters=False, nonlinearity=relu)
    net['conv4_2'] = ConvLayer(net['conv4_1'], num_filters=512, filter_size=3, pad=0, flip_filters=False, nonlinearity=relu)
    net['conv4_3'] = ConvLayer(net['conv4_2'], num_filters=512, filter_size=3, pad=0, flip_filters=False, nonlinearity=relu)
    net['conv5_1'] = DilatedConvLayer(net['conv4_3'], num_filters=512, dilation=(2,2), filter_size=3, pad=0, flip_filters=False, nonlinearity=relu)
    net['conv5_2'] = DilatedConvLayer(net['conv5_1'], num_filters=512, dilation=(2,2), filter_size=3, pad=0, flip_filters=False, nonlinearity=relu)
    net['conv5_3'] = DilatedConvLayer(net['conv5_2'], num_filters=512, dilation=(2,2), filter_size=3, pad=0, flip_filters=False, nonlinearity=relu)
    net['fc6'] = DilatedConvLayer(net['conv5_3'], num_filters=4096, dilation=(4,4), filter_size=7, pad=0, flip_filters=False, nonlinearity=relu)
    net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = ConvLayer(net['drop6'], num_filters=4096, filter_size=1, pad=0, flip_filters=False, nonlinearity=relu)
    net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc-final'] = ConvLayer(net['drop7'], num_filters=21, filter_size=1, pad=0, flip_filters=False, nonlinearity=linear)

    # Create context network
    net['ct_conv1_1'] = ConvLayer(net['fc-final'], num_filters=42, filter_size=3, pad=33, flip_filters=False, nonlinearity=relu)
    net['ct_conv1_2'] = ConvLayer(net['ct_conv1_1'], num_filters=42, filter_size=3, pad=0, flip_filters=False, nonlinearity=relu)
    net['ct_conv2_1'] = DilatedConvLayer(net['ct_conv1_2'], num_filters=84, filter_size=3, dilation=(2,2), pad=0, flip_filters=False, nonlinearity=relu)
    net['ct_conv3_1'] = DilatedConvLayer(net['ct_conv2_1'], num_filters=168, filter_size=3, dilation=(4,4), pad=0, flip_filters=False, nonlinearity=relu)
    net['ct_conv4_1'] = DilatedConvLayer(net['ct_conv3_1'], num_filters=336, filter_size=3, dilation=(8,8), pad=0, flip_filters=False, nonlinearity=relu)
    net['ct_conv5_1'] = DilatedConvLayer(net['ct_conv4_1'], num_filters=672, filter_size=3, dilation=(16,16), pad=0, flip_filters=False, nonlinearity=relu)
    net['ct_fc1'] = ConvLayer(net['ct_conv5_1'], num_filters=672, filter_size=3, pad=0, flip_filters=False, nonlinearity=relu)
    net['ct_final'] = ConvLayer(net['ct_fc1'], num_filters=21, filter_size=1, pad=0, flip_filters=False, nonlinearity=linear)
    net['seg_prob'] = DimshuffleLayer(net['ct_final'], (1, 0, 2, 3))
    net['flatten'] = FlattenLayer(net['seg_prob'], outdim=2)
    net['seg_prob2'] = DimshuffleLayer(net['flatten'], (1, 0))
    net['prob'] = NonlinearityLayer(net['seg_prob2'], nonlinearity=softmax)
    return net
