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
parameters           = {'0': 'ABE_Proportion_sample.txt'} # Dictionary can be expanded for multiple test parameters

## Run Parameters ##
TEST_NUM_SET         = [0] # List can be expanded in case of multiple test parameters
best_model_path_list = ['./ABE_Proportion_Weight']

# Model
length = 26
window_start = 5
window_size = 8

class Deep_xCas9(object):
    def __init__(self, filter_size, filter_num, node_1 = 80, node_2 = 60, l_rate = 0.005, window_size = 5):
        self.inputs          = tf.placeholder(tf.float32, [None, 1, length, 4])
        self.targets         = tf.placeholder(tf.float32, [None, 2**window_size-1])
        self.wow         = tf.placeholder(tf.float32, [None, 2**window_size-1])
        self.possible_labels = tf.placeholder(tf.float32, [None, 2**window_size-1])
        self.is_training     = tf.placeholder(tf.bool)

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
        
        with tf.variable_scope('Output_Layer'):
            W_out        = tf.get_variable("W_out", shape=[node_1, 2**window_size-1])#, initializer=tf.contrib.layers.xavier_initializer())
            B_out        = tf.get_variable("B_out", shape=[2**window_size-1])#, initializer=tf.contrib.layers.xavier_initializer())
            self.outputs = tf.nn.bias_add(tf.matmul(L_fcl1_drop, W_out), B_out)

        #self.possible_outputs = self.outputs
        self.possible_outputs = tf.nn.softmax(self.outputs)#tf.multiply(self.outputs, self.possible_labels))
        # Define loss function and optimizer
        self.obj_loss = tf.reduce_mean(-tf.reduce_sum(self.targets * tf.log(self.possible_outputs) - self.targets * tf.log(self.targets), reduction_indices=[1]))
        self.obj_loss1 = tf.reduce_mean(-tf.reduce_sum(self.targets * tf.log(self.wow) - self.targets * tf.log(self.targets), reduction_indices=[1]))
        self.optimizer   = tf.train.AdamOptimizer(l_rate).minimize(self.obj_loss)
    #def end: def __init__
#class end: Deep_xCas9

def Model_Inference(sess, TEST_X, TEST_Label, model, args, load_episode, test_data_num, testvalbook, testvalsheet, window_size=8):
    test_batch = 1024
    optimizer = model.optimizer
    TEST_Z = np.zeros((TEST_X.shape[0], 2**window_size - 1), dtype=float)
    
    for i in range(int(ceil(float(TEST_X.shape[0])/float(test_batch)))):
        Dict = {model.inputs: TEST_X[i*test_batch:(i+1)*test_batch], model.is_training: False}
        TEST_Z[i*test_batch:(i+1)*test_batch] = sess.run([model.possible_outputs], feed_dict=Dict)[0]
    
    testval_row = 0
    testval_col = 3
    sheet_index = 0
    
    sum = 0
    for test_index in range(len(TEST_Z)):
        sum += TEST_Z[test_index][TEST_Label[test_index]]
    TEST_Z /= sum
    
    for test_index in range(len(TEST_Z)):
        test_value = TEST_Z[test_index][TEST_Label[test_index]]
        testvalsheet[sheet_index].write(testval_row, testval_col, test_value)
        testval_row += 1
    return


# One hot encoding for DNA Sequence
def preprocess_seq(data):
    print("Start preprocessing the sequence done 2d")
    length  = 26
    
    DATA_X = np.zeros((len(data),1,length,4), dtype=int)
    print(np.shape(data), len(data), length)
    for l in range(len(data)):
        for i in range(length):

            try: data[l][i]
            except: print(data[l], i, length, len(data))

            if data[l][i] in "Aa":    DATA_X[l, 0, i, 0] = 1
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

def inference_index(orig_seq, req_seq):
    window_start = 5
    window_size = 8
    index = []
    for seq_index in range(len(list(req_seq))):
        labels_index = -1
        for ind in range(window_size): #change to 2
            if req_seq[seq_index][window_start + ind] == orig_seq[seq_index][window_start + ind]: pass
            else:
                labels_index += 2**(window_size - 1-ind)
        if labels_index < 0:
            print("WT INCLUDED IN REQUIRED SEQUENCE")
            #raise ValueError
            labels_index = 0
        index.append(labels_index)
    return index

def req_seq_produce(seq):
    window_start = 5
    window_size = 8
    req_seq = []
    full_seq = []
    for indiv_seq in seq:
        tmp_seq = [indiv_seq]
        for index in range(window_size):
            if indiv_seq[window_start+index] == 'A':
                print(index, indiv_seq[window_start:window_start+index])
                tmp = []
                for tmp_indiv_seq in tmp_seq:
                    tmp.append(tmp_indiv_seq[:window_start+index]+str("G")+tmp_indiv_seq[window_start+index+1:])
                    full_seq.append(indiv_seq)
                for sequence in tmp:
                    tmp_seq.append(sequence)
        for req_sequence in tmp_seq:
            if req_sequence != tmp_seq[0]:
                req_seq.append(req_sequence)
    return full_seq, req_seq

def getfile_inference(filenum):
    param = parameters['%s'%filenum]
    FILE    = open(path+param, "r")
    data    = FILE.readlines()
    data_n  = len(data) - 1
    seq     = []
    req_seq     = []
    
    for l in range(1, data_n+1):
        try:
            data_split = data[l].split()
            seq.append(data_split[1])
        except:
            print data[l]
            seq.append(data[l])
    #loop end: l
    FILE.close()
    full_seq, req_seq = req_seq_produce(seq)
    processed_full_seq = preprocess_seq(full_seq)
    processed_full_req_seq = inference_index(full_seq, req_seq)
    return processed_full_seq, full_seq, processed_full_req_seq, req_seq

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
TEST_Label = []
testsheet = []
for TEST_NUM_index in range(len(TEST_NUM_SET)):
    TEST_NUM = TEST_NUM_SET[TEST_NUM_index]
    testsheet.append([testbook.add_worksheet('{}'.format(TEST_NUM))])
    tmp_X, pre_X, tmp_Label, pre_Label = getfile_inference(TEST_NUM)
    TEST_X.append(tmp_X)
    TEST_Label.append(tmp_Label)
    test_row = 0
    for index_X in range(np.shape(pre_X)[0]):
        testsheet[-1][-1].write(test_row, 0, pre_X[index_X])
        testsheet[-1][-1].write(test_row, 1, pre_Label[index_X])
        test_row += 1

for best_model_path in best_model_path_list:
    for modelname in os.listdir(best_model_path):
        print(modelname)
        if "meta" in modelname:
            best_model_list.append(modelname[:-5])

print(best_model_list)
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
        model = Deep_xCas9(filter_size, filter_num, node_1, node_2, args[2], window_size)
        saver = tf.train.Saver()
        saver.restore(sess, best_model_path+"/PreTrain-Final-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(args[0][0], args[0][1], args[0][2], args[1][0], args[1][1], args[1][2], args[2], load_episode, args[5], args[6]))
        for i in range(len(TEST_NUM_SET)):
            Model_Inference(sess, TEST_X[i], TEST_Label[i], model, args, load_episode, TEST_NUM_SET[i], testbook, testsheet[i], window_size)
        testbook.close()
