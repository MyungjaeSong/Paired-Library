import os, sys
from os import system
import tensorflow as tf
import numpy as np
from numpy import *
import xlsxwriter
import pyexcel as pe
from random import shuffle

##############################################################################




##############################################################################
## System Paths ##
path                 = './'
parameters           = {'0': 'ABE_Efficiency_sample.txt'} # Dictionary can be expanded for multiple test parameters

## Run Parameters ##
TEST_NUM_SET         = [0] # List can be expanded in case of multiple test parameters
best_model_path_list = ['./ABE_Efficiency_Weight']

# Model
length = 24

class Deep_xCas9(object):
    def __init__(self, filter_size, filter_num, node_1 = 80, node_2 = 60, l_rate = 0.005):
        self.inputs         = tf.placeholder(tf.float32, [None, 1, length, 4])
        self.targets        = tf.placeholder(tf.float32, [None, 1])
        self.is_training    = tf.placeholder(tf.bool)
        def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
            # setup the filter input shape for tf.nn.conv_2d
            conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                              num_filters]

            # initialise weights and bias for the filter
            weights   = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                              name=name+'_W')
            bias      = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

            # setup the convolutional layer operation
            out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='VALID')
            #out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

            # add the bias
            out_layer += bias

            # apply a ReLU non-linear activation
            out_layer = tf.layers.dropout(tf.nn.relu(out_layer), 0.3, self.is_training)

            # now perform max pooling
            #ksize     = [1, pool_shape[0], pool_shape[1], 1]
            #strides   = [1, 1, 2, 1]
            #out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, 
            #                           padding='SAME')
            return out_layer
        #def end: create_new_conv_layer

        L_filter_num = 4
        L_inputs = self.inputs
        L_pool_0 = create_new_conv_layer(L_inputs, L_filter_num, filter_num[0] * 3, [1, filter_size[0]], [1, 2], name='conv1')
        
        with tf.variable_scope('Fully_Connected_Layer1'):
            layer_node_0 = int((length-filter_size[0])/1)+1
            node_num_0   = layer_node_0*filter_num[0] * 3
            
            L_flatten_0  = tf.reshape(L_pool_0, [-1, node_num_0])
            W_fcl1       = tf.get_variable("W_fcl1", shape=[node_num_0, node_1])
            B_fcl1       = tf.get_variable("B_fcl1", shape=[node_1])
            L_fcl1_pre   = tf.nn.bias_add(tf.matmul(L_flatten_0, W_fcl1), B_fcl1)
            L_fcl1       = tf.nn.relu(L_fcl1_pre)
            L_fcl1_drop  = tf.layers.dropout(L_fcl1, 0.3, self.is_training)
        
        with tf.variable_scope('Fully_Connected_Layer2'):
            W_fcl2       = tf.get_variable("W_fcl2", shape=[node_1, node_2])
            B_fcl2       = tf.get_variable("B_fcl2", shape=[node_2])
            L_fcl2_pre   = tf.nn.bias_add(tf.matmul(L_fcl1_drop, W_fcl2), B_fcl2)
            L_fcl2       = tf.nn.relu(L_fcl2_pre)
            L_fcl2_drop  = tf.layers.dropout(L_fcl2, 0.3, self.is_training)
            
        with tf.variable_scope('Output_Layer'):
            W_out        = tf.get_variable("W_out", shape=[node_2, 1])#, initializer=tf.contrib.layers.xavier_initializer())
            B_out        = tf.get_variable("B_out", shape=[1])#, initializer=tf.contrib.layers.xavier_initializer())
            self.outputs = tf.nn.bias_add(tf.matmul(L_fcl2_drop, W_out), B_out)

        # Define loss function and optimizer
        self.obj_loss    = tf.reduce_mean(tf.square(self.targets - self.outputs))
        self.optimizer   = tf.train.AdamOptimizer(l_rate).minimize(self.obj_loss)
    #def end: def __init__
#class end: Deep_xCas9

# Test Model
def Model_Inference(sess, TEST_X, model, args, load_episode, test_data_num, testvalbook, testvalsheet, col_index=1):
    test_batch = 500
    test_spearman = 0.0
    optimizer = model.optimizer
    TEST_Z = zeros((TEST_X.shape[0], 1), dtype=float)
    
    for i in range(int(ceil(float(TEST_X.shape[0])/float(test_batch)))):
        Dict = {model.inputs: TEST_X[i*test_batch:(i+1)*test_batch], model.is_training: False}
        TEST_Z[i*test_batch:(i+1)*test_batch] = sess.run([model.outputs], feed_dict=Dict)[0]
    
    testval_row = 0
    testval_col = 2
    sheet_index = 0
    
    for test_value in (TEST_Z):
        testvalsheet[sheet_index].write(testval_row, testval_col, 100*test_value[0])
        testval_row += 1
    
    return


def preprocess_seq(data):
    print("Start preprocessing the sequence done 2d")
    length  = 24
    
    DATA_X = np.zeros((len(data),1,length,4), dtype=int)
    print(np.shape(data), len(data), length)
    for l in range(len(data)):
        for i in range(length):

            try: data[l][i]
            except: print(data[l], i, length, len(data))

            if data[l][i]in "Aa":    DATA_X[l, 0, i, 0] = 1
            elif data[l][i] in "Cc": DATA_X[l, 0, i, 1] = 1
            elif data[l][i] in "Gg": DATA_X[l, 0, i, 2] = 1
            elif data[l][i] in "Tt": DATA_X[l, 0, i, 3] = 1
            else:
                print "Non-ATGC character " + data[l]
                print i
                print data[l][i]
                sys.exit()
        #loop end: i
    #loop end: l
    print("Preprocessing the sequence done")
    return DATA_X
#def end: preprocess_seq


def getfile_inference(filenum):
    param   = parameters['%s' % filenum]
    FILE    = open(path+param, "r")
    data    = FILE.readlines()
    data_n  = len(data) - 1
    seq     = []

    for l in range(1, data_n+1):
        try:
            data_split = data[l].split()
            seq.append(data_split[1])
        except:
            print data[l]
            raise ValueError
    #loop end: l
    FILE.close()
    processed_full_seq = preprocess_seq(seq)

    return processed_full_seq, seq
#def end: getseq


if "outputs" not in os.listdir(os.getcwd()):
    os.makedirs('outputs')

#TensorFlow config
conf                                = tf.ConfigProto()
conf.gpu_options.allow_growth       = True
os.environ['CUDA_VISIBLE_DEVICES']  = '0'
best_model_cv                       = 0.0
best_model_list                     = []

testbook = xlsxwriter.Workbook('outputs/TEST_OUTPUT_fortest.xlsx')

TEST_X = []
testsheet = []
for TEST_NUM_index in range(len(TEST_NUM_SET)):
    TEST_NUM = TEST_NUM_SET[TEST_NUM_index]
    testsheet.append([testbook.add_worksheet('{}'.format(TEST_NUM))])
    tmp_X, pre_X = getfile_inference(TEST_NUM)
    TEST_X.append(tmp_X)
    test_row = 0
    for index_X in range(np.shape(pre_X)[0]):
        testsheet[-1][-1].write(test_row, 0, pre_X[index_X])
        test_row += 1
        
for best_model_path in best_model_path_list:
    for modelname in os.listdir(best_model_path):
        if "meta" in modelname:
            best_model_list.append(modelname[:-5])
#loop end: best_model_path

for index in range(len(best_model_list)):
    best_model_path = best_model_path_list[index]
    best_model      = best_model_list[index]
    valuelist       = best_model.split('-')
    fulllist        = []
    
    for value in valuelist:
        try:
            value=int(value)
        except:
            try:    value=float(value)
            except: pass
        fulllist.append(value)

    print(fulllist[2:])
    
    filter_size_1, filter_size_2, filter_size_3, filter_num_1, filter_num_2, filter_num_3, l_rate, load_episode, node_1, node_2 = fulllist[2:]
    filter_size = [filter_size_1, filter_size_2, filter_size_3]
    filter_num  = [filter_num_1, filter_num_2, filter_num_3]

    args = [filter_size, filter_num, l_rate, 0, None, node_1, node_2]
    # Loading the model with the best validation score and test
    tf.reset_default_graph()
    with tf.Session(config=conf) as sess:
        sess.run(tf.global_variables_initializer())
        model = Deep_xCas9(filter_size, filter_num, node_1, node_2, args[2])
        saver = tf.train.Saver()
        saver.restore(sess, best_model_path+"/PreTrain-Final-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(args[0][0], args[0][1], args[0][2], args[1][0], args[1][1], args[1][2], args[2], load_episode, args[5], args[6]))
        for i in range(len(TEST_NUM_SET)):
            Model_Inference(sess, TEST_X[i], model, args, load_episode, TEST_NUM_SET[i], testbook, testsheet[i])
        testbook.close()
