import os
# disable obnoxious TF warnings about instruction sets
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import math
from random import shuffle
import numpy as np
import pickle
import random
import json
import collections
import matplotlib
# in order to run matplotlib w/o a running X-server,
# use Agg backend
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import pylab
from sklearn.manifold import TSNE
import pprint
from argparse import ArgumentParser
import shutil

num_heroes = 113

# define command line arguments to be parsed
# note: default parameters may not be optimal, see readme for more info
parser = ArgumentParser()
parser.add_argument("-useL2", "--l", type=int, default=1, dest="useL2",help="var<=0 => don't use l2 regularization", metavar="1")
parser.add_argument("-beta", "--b", type=float, default=0.01, dest="beta",help="l2 loss coefficient", metavar="0.01")
parser.add_argument("-Pkeep", "--p", type=float, default=0.9, dest="Pkeep", help="dropout hyperparameter Pkeep", metavar="0.5")
parser.add_argument("-useDropout", "--d", type=int, default=-1, dest="useDropout", help="var<=0 => don't use dropout", metavar="-1")
parser.add_argument("-batchSize", "--B", type=int, default=200, dest="batchsize", help="batch size", metavar="-1")
default_sizes = [128, 84, 64, 48, 16]
for i in range(5):
    parser.add_argument("-hn"+str(i+1), type=int, default=default_sizes[i], dest="hn"+str(i+1), help="hidden neuron 3", metavar="-1")
parser.add_argument("-log", "--L", type=int, default=-1, dest="log", help="var>0 => log / save weights", metavar="-1")
parser.add_argument("-plot", "--P", type=int, default=-1, dest="plot", help="var>0 => plot accuracy", metavar="-1")
parser.add_argument("-steps", "--S", type=int, default="100000", dest="steps", help="num of steps", metavar="100000")

# parse arguments
args = parser.parse_args()
useDropout = True if args.useDropout > 0 else False
useL2 = True if args.useL2 > 0 else False
Pkeep = args.Pkeep
beta = args.beta
batch_size = args.batchsize
for i in range(5):
    exec("hidden_neurons_"+str(i+1)+" = args.hn"+str(i+1))
log = True if args.log > 0 else False
plot = True if args.plot > 0 else False
num_steps = args.steps

# print main arguments
print("Arguments: useL2: {}, beta: {}, useDropout: {}, Pkeep: {}, log: {}, plot: {}".format(useL2, beta, useDropout, Pkeep, log, plot))

# load hero name-id look up table
hero_name_data = json.load(open('hero_name_lut.txt', 'r'))
hero_name_lut = [None] * (num_heroes+1)
for hero in hero_name_data:
    hero_name_lut[hero['id']-1] = hero['name']
hero_name_lut = hero_name_lut[:23] + hero_name_lut[24:num_heroes+1]

# load trained hero embeddings
hero_embeddings = pickle.load(open("final_embeddings", 'rb'))
embedding_size = len(hero_embeddings[0])
hero_embeddings = np.array(hero_embeddings)

# id_fix function
id_fix = lambda hero_ids: [hero_id-1 if hero_id < 24 else hero_id-2 for hero_id in hero_ids]

# load math dump
match_dump = pickle.load(open('pro_dump.p', 'rb'))
num_matches = len(match_dump) * 2
print("num of matches: ", num_matches)

# as radiant and dire sides are virually equivalent
# we can mirror the data so that every match 
# [radiant-<R, dire-<D] -> b_radiant_win can be interpreted as [radiant<-D, dire<-R] -> !b_radiant_win
context = np.ndarray(shape=(num_matches, 10))
labels = np.zeros(shape=(num_matches))
for i in range(num_matches//2):
    _, _, radiant_win, heroes = match_dump[i][:]
    labels[2*i] = 1 if radiant_win else 0
    labels[2*i+1] = 0 if radiant_win else 1
    context[2*i, :] = np.array(id_fix(heroes))
    context[2*i+1, :] = np.array(id_fix(heroes[5:]+heroes[:5]))

# shuffle
def shuffle_set(x, y, n):
    l = list(range(n))
    shuffle(l)
    return x[l], y[l]

context, labels = shuffle_set(context, labels, num_matches)

# split data into validation and training sets
# from previous experience, it's been established that
# it's very unlikely to overfit validation set, thus, validation set is used for test purposes as well.
valdiation_size = 1000
num_training_matches = num_matches-valdiation_size

validation_set = context[-valdiation_size:, :]
validation_labels = labels[-valdiation_size:]
validation_set, validation_labels = shuffle_set(validation_set, validation_labels, valdiation_size)

training_set = context[:num_training_matches, :]
training_labels = labels[:num_training_matches]

# load batch
batch_offset = 0
def load_batch(batch_size):
    global batch_offset, training_set, training_labels
    if batch_offset == 0:
        training_set, training_labels = shuffle_set(training_set, training_labels, num_training_matches)
    X = training_set[batch_offset:batch_offset+batch_size, :]
    Y = training_labels[batch_offset:batch_offset+batch_size]
    batch_offset = batch_offset+batch_size
    return X, Y

# create tf Graph
graph = tf.Graph()

# create the model
# UPDATE: created methods to organize and keep track of fully connected layers in a Pythonic way
with graph.as_default():
    heroes = tf.placeholder(dtype=tf.int32, shape=[batch_size, 10])
    labels = tf.placeholder(dtype=tf.int32, shape=[batch_size])

    hero_embedding_lut = tf.Variable(hero_embeddings)

    var_dict = {"h_embd" : hero_embedding_lut}
    l2_loss_contrib_vars = {}

    def get_embeddings(_heroes):
        radiant_vector = tf.reduce_sum(tf.nn.l2_normalize(tf.nn.embedding_lookup(hero_embedding_lut, _heroes[:, :5]), dim=1), 1)
        dire_vector = tf.reduce_sum(tf.nn.l2_normalize(tf.nn.embedding_lookup(hero_embedding_lut, _heroes[:, 5:]), dim=1), 1)
        return tf.concat([radiant_vector, dire_vector], 1)
    
    def get_var(name, shape, mean=0.0, stdev=0.001):
        global var_dict

        var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(mean, stdev))

        if var not in  var_dict:
            var_dict[name] = var

        return var
    
    def get_fc_var(name, in_size, out_size, mean=0.0, stdev=0.001):
        weights = get_var(name + "_weights", [in_size, out_size], mean, stdev)
        biases = get_var(name + "_biases", [out_size], mean, stdev)
        return weights, biases

    def fc_layer(name, input, in_size, out_size, gate="relu", l2=True, dropout=False):
        global l2_loss_contib_vars

        # create/fetch variables 
        # use Xavier initialization
        weights, biases = get_fc_var(name, in_size, out_size, 0.0, 2.0/math.sqrt(in_size))
        output = tf.nn.xw_plus_b(input, weights, biases, name=name)
        
        # gate the output, forthe final layer, softmax is handled by softmax_cross_entropy_with_logits()
        if gate == "relu":
            output = tf.nn.relu(output)
        elif gate == "tanh":
            output = tf.tanh(output)
        elif gate == "sigmoid":
            output = tf.sigmoid(output)
        
        # apply dropout
        if dropout:
            output = tf.nn.dropout(output, Pkeep)

        # append weights variable to the dict of variables that contribute to l2 regularization term
        if l2:
            l2_loss_contrib_vars[name] = weights

        return output

    def residual_block(name, input, size, gate="relu", l2=True, dropout=False):
        first_layer = fc_layer(name+"/res_hid_1", input, size, size, "relu", l2, dropout)
        second_layer = fc_layer(name+"/res_hid_2", first_layer, size, size, "relu", l2, dropout)
        output = second_layer + input
           
        if gate == "relu":
            output = tf.nn.relu(output)
        elif gate == "tanh":
            output = tf.tanh(output)
        elif gate == "sigmoid":
            output = tf.sigmoid(output)

        if dropout:
            output = tf.nn.dropout(output, Pkeep)

        return output

    def model(input, l2=True, dropout=False):
        assert hidden_neurons_1 == 2*embedding_size
        # experimenting with residual layers, even though the model is not very deep
        o1 = residual_block("res_block_1_1", input, 2*embedding_size, l2=l2, dropout=dropout)      
        o2 = fc_layer("fc_1_2", o1, hidden_neurons_1, hidden_neurons_2, l2=l2, dropout=dropout)    
        o3 = residual_block("res_block_2_3", o2, hidden_neurons_2, l2=l2, dropout=dropout)         
        o4 = fc_layer("fc_2_4", o3, hidden_neurons_2, hidden_neurons_3, l2=l2, dropout=dropout)    
        o5 = residual_block("res_block_3_5", o4, hidden_neurons_3, l2=l2, dropout=dropout)         
        o6 = fc_layer("fc_3_6", o5, hidden_neurons_3, hidden_neurons_4, l2=l2, dropout=dropout)    
        o7 = residual_block("res_block_4_7", o6, hidden_neurons_4, l2=l2, dropout=dropout)         
        o8 = fc_layer("fc_4_8", o7, hidden_neurons_4, hidden_neurons_5, l2=l2, dropout=dropout)    
        o9 = fc_layer("fc_5_9", o8, hidden_neurons_5, 2, gate="linear", l2=l2, dropout=dropout)    
        return o9

    # compute output of model on training batch
    output = model(get_embeddings(heroes), l2=useL2, dropout=useDropout)

    # compute cross entropy loss and add l2 loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=labels)

    if useL2:
        l2_regularization = tf.add_n([tf.nn.l2_loss(k) for k in l2_loss_contrib_vars.values()], name="l2_regularization")
        loss = tf.reduce_mean(cross_entropy) + beta * l2_regularization
    else:
        loss = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdadeltaOptimizer(1.0).minimize(loss)

    tf.get_variable_scope().reuse_variables()
    batch_predictions = tf.argmax(model(get_embeddings(heroes), l2=useL2, dropout=False), axis=1)

    valid_set = tf.constant(validation_set, dtype=tf.int32)
    valid_labels = tf.constant(validation_labels)
    validation_predictions = tf.argmax(model(get_embeddings(valid_set), l2=False, dropout=False), axis=1)

# method to calculate acuracy of predictions
def acc(pred, truth):
    k = 0
    for i, p in enumerate(pred):
        if p == truth[i]:
            k = k + 1
    return k/i

# create log folder, as this script is intented to be used for hyperparameter search, use specific names for folders
log_folder = "pred_log/"
wfolder = "{}/weights_{}_{}".format(log_folder, beta if useL2 else "NL2", Pkeep if useDropout else "ND")

for i in range(1, 6):
    wfolder = wfolder + "_" + str(globals()["hidden_neurons_"+str(i)])

if log:
    if not os.path.isdir(wfolder):
        os.makedirs(wfolder, exist_ok=True)
    else:
        i = 1
        while os.path.isdir(wfolder + "_" + str(i)):
            i = i + 1
        wfolder = wfolder + "_" + str(i)
        os.makedirs(wfolder, exist_ok=True)

# train the model
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    
    saver = tf.train.Saver(var_list=var_dict)

    loss_hist = []
    train_acc_hist = []
    val_acc_hist = []

    average_loss = 0
    average_accuracy = 0

    for step in range(num_steps+1):

        if batch_offset+batch_size > num_training_matches:
            batch_offset = 0

        _x, _labels = load_batch(batch_size)
        feed_dict = {heroes : _x, labels : _labels}

        _, l, _pred = session.run([optimizer, loss, batch_predictions], feed_dict=feed_dict)
        
        average_loss = average_loss + l
        average_accuracy = average_accuracy + acc(_pred, _labels)

        if step == 0:
            print('initial loss: ', average_loss)

        # log
        if step % 500 == 0:
            average_loss = average_loss/500 if step>0 else average_loss
            average_accuracy = average_accuracy/500 if step>0 else average_accuracy
            val_acc = acc(validation_predictions.eval(), validation_labels)

            loss_hist.append(average_loss)
            train_acc_hist.append(average_accuracy)
            val_acc_hist.append(val_acc)

            print("avg. loss at step", step, average_loss)
            print("acc:", average_accuracy)
            print("val acc:", val_acc)

            average_loss = 0
            average_accuracy = 0

        # if the model isn't converging, stop the training and delete logs so far
        if step > 10000 and  train_acc_hist[-1]< 0.6:
            print("The model is not converging! Terminating process, not saving the weights or hyperparameters")
            if log:
                shutil.rmtree(wfolder+"")
            exit()

        # save tf graph
        if log and step % 1000 == 0 and step > 0:
            saver.save(session, "{}/pred_model".format(wfolder))

    if log:
        pickle.dump({"embd_size" : embedding_size, "hn1" : hidden_neurons_1, "hn2" : hidden_neurons_2, "hn3": hidden_neurons_3, "hn4" : hidden_neurons_4,
                     "hn5" : hidden_neurons_5, "embd_type" : embd_type},
                open("{}/nn_parameters.p".format(wfolder), "wb"))

# create polynomial fit function
def fit(x, y):
    z = np.polyfit(x, y, 5)
    f = np.poly1d(z)

    # calculate new x's and y's
    x_new = np.linspace(x[0], x[-1], 50)
    y_new = f(x_new)
    return x_new, y_new

# exit the script if plotting is disabled
if not plot:
    exit()

# plot
x = np.arange(0, num_steps+500, 500)
plt.plot(x, train_acc_hist)
plt.plot(x, val_hist)
xv, yv = fit(x, val_hist)
plt.plot(xv, yv)

plt.legend(['train acc', 'val acc', "val fit"], loc='upper left')
plt.savefig("{}/pred.png".format(w_folder), dpi=800)

plt.clf()
plt.cla()
plt.close()
