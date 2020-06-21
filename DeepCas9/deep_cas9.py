import os
import sys
import argparse
import numpy as np
import tensorflow as tf


class DeepCas9(object):
    def __init__(self, filter_size, filter_num, node_1=80, node_2=60, l_rate=0.005, length=30):
        self.inputs = tf.placeholder(tf.float32, [None, 1, length, 4])
        self.targets = tf.placeholder(tf.float32, [None, 1])
        self.is_training = tf.placeholder(tf.bool)

        L_pool_0 = self.create_new_conv_layer(4, filter_num[0], [1, filter_size[0]], [1, 2], name='conv1')
        L_pool_1 = self.create_new_conv_layer(4, filter_num[1], [1, filter_size[1]], [1, 2], name='conv2')
        L_pool_2 = self.create_new_conv_layer(4, filter_num[2], [1, filter_size[2]], [1, 2], name='conv3')

        with tf.variable_scope('Fully_Connected_Layer1'):
            layer_node_0 = int((length - filter_size[0]) / 2) + 1
            node_num_0 = layer_node_0 * filter_num[0]
            layer_node_1 = int((length - filter_size[1]) / 2) + 1
            node_num_1 = layer_node_1 * filter_num[1]
            layer_node_2 = int((length - filter_size[2]) / 2) + 1
            node_num_2 = layer_node_2 * filter_num[2]
            L_flatten_0 = tf.reshape(L_pool_0, [-1, node_num_0])
            L_flatten_1 = tf.reshape(L_pool_1, [-1, node_num_1])
            L_flatten_2 = tf.reshape(L_pool_2, [-1, node_num_2])
            L_flatten = tf.concat([L_flatten_0, L_flatten_1, L_flatten_2], 1, name='concat')
            node_num = node_num_0 + node_num_1 + node_num_2
            W_fcl1 = tf.get_variable("W_fcl1", shape=[node_num, node_1])
            B_fcl1 = tf.get_variable("B_fcl1", shape=[node_1])
            L_fcl1_pre = tf.nn.bias_add(tf.matmul(L_flatten, W_fcl1), B_fcl1)
            L_fcl1 = tf.nn.relu(L_fcl1_pre)
            L_fcl1_drop = tf.layers.dropout(L_fcl1, 0.3, self.is_training)

        with tf.variable_scope('Fully_Connected_Layer2'):
            W_fcl2 = tf.get_variable("W_fcl2", shape=[node_1, node_2])
            B_fcl2 = tf.get_variable("B_fcl2", shape=[node_2])
            L_fcl2_pre = tf.nn.bias_add(tf.matmul(L_fcl1_drop, W_fcl2), B_fcl2)
            L_fcl2 = tf.nn.relu(L_fcl2_pre)
            L_fcl2_drop = tf.layers.dropout(L_fcl2, 0.3, self.is_training)

        with tf.variable_scope('Output_Layer'):
            W_out = tf.get_variable("W_out", shape=[node_2, 1])  # , initializer=tf.contrib.layers.xavier_initializer())
            B_out = tf.get_variable("B_out", shape=[1])  # , initializer=tf.contrib.layers.xavier_initializer())
            self.outputs = tf.nn.bias_add(tf.matmul(L_fcl2_drop, W_out), B_out)

        # Define loss function and optimizer
        self.obj_loss = tf.reduce_mean(tf.square(self.targets - self.outputs))
        self.optimizer = tf.train.AdamOptimizer(l_rate).minimize(self.obj_loss)

    # def end: def __init__

    def create_new_conv_layer(self, num_input_channels, num_filters, filter_shape, pool_shape, name):
        # setup the filter input shape for tf.nn.conv_2d
        conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                           num_filters]

        # initialise weights and bias for the filter
        weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                              name=name + '_W')
        bias = tf.Variable(tf.truncated_normal([num_filters]), name=name + '_b')

        # setup the convolutional layer operation
        out_layer = tf.nn.conv2d(self.inputs, weights, [1, 1, 1, 1], padding='VALID')

        # add the bias
        out_layer += bias

        # apply a ReLU non-linear activation
        out_layer = tf.layers.dropout(tf.nn.relu(out_layer), 0.3, self.is_training)

        # now perform max pooling
        ksize = [1, pool_shape[0], pool_shape[1], 1]
        strides = [1, 1, 2, 1]
        out_layer = tf.nn.avg_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
        return out_layer


# class end: DeepCas9

def model_final_test(sess, TEST_X, filter_size, filter_num, if3d, model, l_rate, load_episode, best_model_path, OUT):
    test_batch = 500
    test_spearman = 0.0
    optimizer = model.optimizer
    TEST_Z = np.zeros((TEST_X.shape[0], 1), dtype=float)

    for i in range(int(np.ceil(float(TEST_X.shape[0]) / float(test_batch)))):
        Dict = {model.inputs: TEST_X[i * test_batch:(i + 1) * test_batch], model.is_training: False}
        TEST_Z[i * test_batch:(i + 1) * test_batch] = sess.run([model.outputs], feed_dict=Dict)[0]

    OUT.write("Testing final \n {} ".format(tuple(TEST_Z.reshape([np.shape(TEST_Z)[0]]))))
    OUT.write("\n")
    return


# def end: Model_Finaltest


def preprocess_seq(data):
    print("Convert sequences into one-hot encoding matrices")
    length = 30

    data_x = np.zeros((len(data), 1, length, 4), dtype=int)
    print(np.shape(data), len(data), length)
    for l in range(len(data)):
        for i in range(length):

            try:
                data[l][i]
            except:
                print(data[l], i, length, len(data))

            if data[l][i] in "Aa":
                data_x[l, 0, i, 0] = 1
            elif data[l][i] in "Cc":
                data_x[l, 0, i, 1] = 1
            elif data[l][i] in "Gg":
                data_x[l, 0, i, 2] = 1
            elif data[l][i] in "Tt":
                data_x[l, 0, i, 3] = 1
            else:
                print "Non-ATGC character " + data[l]
                print i
                print data[l][i]
                sys.exit()
        # loop end: i
    # loop end: l
    print("Conversion done")
    return data_x


# def end: preprocess_seq


def load_sequences(path):
    FILE = open(path, "r")
    data = FILE.readlines()
    data_n = len(data) - 1
    seq = []

    for l in range(1, data_n + 1):
        try:
            data_split = data[l].split()
            seq.append(data_split[1])
        except:
            print data[l]
            seq.append(data[l])
    # loop end: l
    FILE.close()
    processed_full_seq = preprocess_seq(seq)
    return processed_full_seq, seq


# def end: getseq


def main(argv):
    parser = argparse.ArgumentParser(description='Predict indel frequency from sequence using trained DeepCas9 model')
    parser.add_argument('target_sequences_paths',
                        help='comma separated list of target dataset paths, each tsv file contains table of '
                             'Target Number <TAB> 30 bp target sequence (4 bp + 20 bp protospacer + PAM + 3 bp)')
    parser.add_argument('model_dir_paths',
                        help='comma separated list of trained model parameters folder')
    parser.add_argument('output_prefix',
                        help='prefix to output prediction files, will write output per model')

    args = parser.parse_args(argv)
    process(args)

def process(args):
    np.set_printoptions(threshold='nan')

    target_sequences_files_list = args.target_sequences_paths.split(',')
    model_dir_list = args.model_dir_paths.split(',')
    model_name_list = get_model_names(model_dir_list)

    # TensorFlow config
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    best_model_cv = 0.0

    test_x = []
    test_x_nohot = []
    for target_sequences_path in target_sequences_files_list:
        tmp_x, tmp_x_nohot = load_sequences(target_sequences_path)
        test_x.append(tmp_x)
        test_x_nohot.append(tmp_x_nohot)

    for model_path, model_name in zip(model_dir_list, model_name_list):
        l_rate, filter_num, filter_size, if3d, load_episode, node_1, node_2 = get_args_from_model_name(model_name)
        tf.reset_default_graph()

        with tf.Session(config=conf) as sess:
            sess.run(tf.global_variables_initializer())
            model = DeepCas9(filter_size, filter_num, node_1, node_2, l_rate)

            saver = tf.train.Saver()
            saver.restore(sess, model_path + '/' + model_name)

            OUT = open("{}_{}.txt".format(args.output_prefix, model_path.split('/')[-1]), "a")
            OUT.write("{}".format(model_name))
            OUT.write("\n")

            for i in range(len(target_sequences_files_list)):
                print ("TEST_NUM : {}".format(i))
                OUT.write("\n")
                OUT.write("TEST_FILE : {}".format(target_sequences_files_list[i]))
                OUT.write("\n")
                model_final_test(sess, test_x[i], filter_size, filter_num, if3d, model, l_rate, load_episode,
                                 model_path, OUT)
            # loop end: i
            OUT.write("\n")
            OUT.close()


def get_args_from_model_name(model_name):
    full_list = extract_params_from_model_name(model_name)
    print(full_list[2:])
    if full_list[2:][-3] is True:
        if3d, filter_size_1, filter_size_2, filter_size_3, filter_num_1, filter_num_2, filter_num_3, \
        l_rate, load_episode, inception, node_1, node_2 = full_list[2:]
        filter_size = [filter_size_1, filter_size_2, filter_size_3]
        filter_num = [filter_num_1, filter_num_2, filter_num_3]
    else:
        if3d, filter_size, filter_num, l_rate, load_episode, inception, node_1, node_2 = full_list[2:]
    return l_rate, filter_num, filter_size, if3d, load_episode, node_1, node_2


def extract_params_from_model_name(model_name):
    value_list = model_name.split('-')
    full_list = []
    for value in value_list:
        if value == 'True':
            value = True
        elif value == 'False':
            value = False
        else:
            try:
                value = int(value)
            except:
                try:
                    value = float(value)
                except:
                    pass
        full_list.append(value)
    return full_list


def get_model_names(model_dir_list):
    model_name_list = []
    for model_path in model_dir_list:
        for model_meta_file_name in os.listdir(model_path):
            if "meta" in model_meta_file_name:
                model_name_list.append(model_meta_file_name[:-5])
    return model_name_list


if __name__ == '__main__':
    main(sys.argv[1:])
