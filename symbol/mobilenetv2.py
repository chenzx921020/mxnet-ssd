# Mobilenet v2
# based on mobilenet v1, inverted residual structure is in future
# author by chenzx

import mxnet as mx

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
    act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))
    return act 

def inv_unit(data, num_filter, stride, num_group, name=None, suffix=''):
    conv_pw1 = mx.sym.Convolution(data=data, num_filter=num_filter*6, kernel=(1, 1),stride=(1,1), pad=(0,0), name='%s%s_pw1' %(name, suffix))
    act1 = mx.sym.Activation(data=conv_pw1, act_type='relu',name='%s%s_relu1' %(name,suffix))
    conv_dw = mx.sym.Convolution(data=act1, num_filter=num_filter*6, num_group=num_group*6,kernel=(3, 3), stride=(stride,stride), pad=(1, 1), name='%s%s_dw' %(name, suffix))
    act2 = mx.sym.Activation(data=conv_dw, act_type='relu', name='%s%s_relu2' %(name,suffix))
    conv_pw2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1, 1),stride=(1,1), pad=(0, 0), name='%s%s_pw2' %(name,suffix))
    shortcut = mx.sym.Convolution(data=data,num_filter=num_filter,kernel=(1, 1),stride=(stride,stride),no_bias=True,name='%s%s_sc' %(name,suffix))
    return conv_pw2+shortcut

def get_symbol(num_classes, **kwargs):
    data = mx.symbol.Variable(name="data")
    conv_1 = Conv(data, num_filter=32, kernel=(3, 3),pad=(1, 1), stride=(2, 2),name="conv_1")
    conv_2 = inv_unit(conv_1, 32, 1, 32, name='conv_2')
    conv_3 = inv_unit(conv_1, 32, 2, 32, name='conv_3')
    conv_4 = inv_unit(conv_3, 64, 1, 64, name='conv_4')
    conv_5 = inv_unit(conv_4, 64, 2, 64, name='conv_5')
    conv_6 = inv_unit(conv_5, 128, 1, 128, name='conv_6')
    conv_7 = inv_unit(conv_6, 128, 2, 128, name='conv_7')
    #conv_8 = inv_unit(conv_7, 128, 1, 128, name='conv_8')
    conv_8 = Conv(conv_7, num_filter=256, kernel=(1, 1),pad=(0, 0),stride=(1, 1),name="conv_8")
    #conv_9 = inv_unit(conv_8, 128, 2, 128, name='conv_9')
    conv_9 = Conv(conv_8, num_filter=512, kernel=(3, 3),pad=(1, 1),stride=(2, 2),name="conv_9")
    ##normal convolution
    conv_10 = Conv(conv_9, num_filter=512, kernel=(1, 1),pad=(0, 0),stride=(1, 1),name="conv_10")
    pool = mx.sym.Pooling(data=conv_10, kernel=(1, 1),stride=(1, 1), pool_type="avg", name="gap",global_pool=True)
    flatten = mx.sym.Flatten(data=pool, name="flatten")
    fc = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name='fc')
    softmax = mx.symbol.SoftmaxOutput(data=fc,name='softmax')
    return softmax
