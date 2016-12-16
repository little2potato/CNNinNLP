# -*- coding: cp936 -*-
"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import logging
from conv_net_common import *


warnings.filterwarnings("ignore")   

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)


log_format = '%(asctime)s [%(levelname)s]\t%(message)s'
logging.basicConfig(format=log_format, datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)

formatter = logging.Formatter(log_format)
filelog = logging.FileHandler('log/run_kim_conv_net.log', 'a')
filelog.setLevel(logging.INFO)
filelog.setFormatter(formatter)
logging.getLogger('').addHandler(filelog)


def eval_kim_conv_net(datafile,
                   img_w=300, 
                   filter_hs=[3,4,5],
                   hidden_units=[100, 200, 4], 
                   dropout_rate=[0.5],
                   n_epochs=25, 
                   batch_size=50, 
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   activations=[Tanh],
                   sqr_norm_lim=9,
                   non_static=True,
                   shuffle_batch=True,
                   L2_reg=0.0001,
                   regluar=False):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y,z] x is the number of feature maps (per filter window), and y is the penultimate layer, z is the output layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """
    logging_argv_info(logging.info, locals())

    rng = np.random.RandomState(3435)
    train_set_x, train_set_y, test_set_x, test_set_y, word_idx_matrix  = load_data(datafile, batch_size)

    #print train_set_y.eval()
    #print test_set_y.eval()
    test_inst = test_set_x[0].get_value(borrow=True).shape[0]
    assert test_inst%batch_size == 0, 'test size must be multiple of batch_size: test_size: %d, batch_size: %d'%(test_inst, batch_size)     
    assert test_set_x[0].get_value(borrow=True).shape[0] == test_set_x[1].get_value(borrow=True).shape[0], 'test 0,1 dim'
    #test the shape of the train set and the test set
    logging.debug('train_set_x[0] shapes: %s; train_set_x[1]: %s '%(str(train_set_x[0].get_value().shape), str(train_set_x[1].get_value().shape)))
    logging.debug('test_set_x[0] shapes: %s; test_set_x[1]: %s '%(str(test_set_x[0].get_value().shape), str(test_set_x[1].get_value().shape)))

    # concatenate((s1,s2),1) two sentences
    train_set_x = theano.shared(np.concatenate(
        [train_set_x[0].get_value(),train_set_x[1].get_value()], 1))
    test_set_x = theano.shared(np.concatenate(
        [test_set_x[0].get_value(),test_set_x[1].get_value()], 1))
    print train_set_x.eval()[0:10]
    print test_set_x.eval()[0:10]

    print train_set_x.dtype
    print type(train_set_x)
    #train_set_x = theano.shared(np.concatenate(
    #    [train_set_x.get_value(), test_set_x.get_value()]))
    #train_set_y = T.concatenate([train_set_y, test_set_y])
    logging.debug('train_set_x shapes: %s; test_set_x: %s '%(str(train_set_x.get_value().shape), str(test_set_x.get_value().shape)))
    img_h = train_set_x.get_value().shape[1]
    logging.debug('img_h=%d'%img_h)
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    
    #define model architecture
    index = T.lscalar()
    x = T.matrix('x')   
    y = T.ivector('y')
    Words = theano.shared( np.array(word_idx_matrix), name = "Words")
    logging.debug('Words shapes: %s '%(str(Words.get_value().shape)))
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))])
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))                     
    conv_layers = []
    layer1_inputs = []
    L2_norms = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
        L2_norms.append(conv_layer.L2_norm)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)    
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    L2_norms.append(classifier.L2_norm)
    #define parameters of the model and update functions using adadelta
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)
    dropout_cost_with_regluar = dropout_cost + L2_reg*np.sum(L2_norms)
    if regluar:
        grad_updates = sgd_updates_adadelta(params, dropout_cost_with_regluar, lr_decay, 1e-6, sqr_norm_lim)
        #gparams = [ T.grad(dropout_cost_with_regluar, param) for param in params ]
        #grad_updates = [ (param, param - 0.1*gparam) for param, gparam in zip(params, gparams) ]
    else:
        grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
             
    #make theano functions to get train/val/test errors


    n_train_batches = train_set_x.get_value().shape[0]/batch_size
    n_test_batches = test_set_x.get_value().shape[0]/batch_size
    
    logging.debug('n_train_batches=%d'%n_train_batches)
    logging.debug('n_test_batches=%d'%n_test_batches)
    
    test_model = theano.function([index], classifier.errors(y)*batch_size,
                givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]}
    )
    
    train_loss_model = theano.function([index], classifier.errors(y),
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]})
    
    train_model = theano.function([index], [cost, L2_reg*np.sum(L2_norms)], updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]}
            )
    
    
    #start training over mini-batches
    logging.debug('... training')
    epoch = 0
    best_test_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0    
    while (epoch < n_epochs):        
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch, L2_norm = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch, L2_norm = train_model(minibatch_index)  
                set_zero(zero_vec)
        logging.debug('epoch %i/%i, train cost %f, train L2_norm %f' % (epoch, n_epochs, cost_epoch, L2_norm))
        train_losses = [train_loss_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        errors  = [test_model(i) for i in xrange(n_test_batches)]
        error = numpy.sum(errors)
        test_loss = error/float(test_inst)
        test_perf = 1- test_loss                
        logging.info('epoch %i/%i, train perf %f%%, val perf %f%%' % (epoch, n_epochs, train_perf * 100., test_perf*100.))
        if test_perf >= best_test_perf:
            best_test_perf = test_perf
    logging.info('best_test_perf=%.5f'%best_test_perf)
    return best_test_perf   
  
   
if __name__=="__main__":
   if len(sys.argv) < 2:
       print "Usage: run datafile"
       sys.exit(0)
   filepath = sys.argv[1]
    
   eval_kim_conv_net(
        datafile = filepath,
        filter_hs=[1,2,3,4,5],
        batch_size=20,
        lr_decay = 0.95,
        conv_non_linear="relu",
        activations=[Iden],
        sqr_norm_lim=9,
        non_static=True,
        L2_reg=0.00001,
        regluar=True
        )

