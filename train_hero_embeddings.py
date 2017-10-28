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

num_heroes = 113

# load hero name-id look up table
hero_name_data = json.load(open('hero_name_lut.txt', 'r'))
hero_name_lut = [None] * (num_heroes+1)
for hero in hero_name_data:
    hero_name_lut[hero['id']-1] = hero['name']

hero_name_lut = hero_name_lut[:23] + hero_name_lut[24:num_heroes+1]

# load 47550 pro matches
match_dump = pickle.load(open('pro_dump.p', 'rb'))
num_available_matches = len(match_dump)

def id_fix(hero_id):
    return hero_id-1 if hero_id < 24 else hero_id-2

# create batch method
batch_start_match = 0
def load_batch(batch_size):
    global batch_start_match
    
    # from every game, we can extract 5 samples, one for each hero in the winning team
    assert batch_size % 5 == 0

    num_matches = batch_size//5
    assert batch_start_match+num_matches <= num_available_matches

    if batch_start_match == 0:
        shuffle(match_dump)

    batch_heroes = []
    batch_allies = []
    batch_enemies = []
    for index in range(batch_start_match, batch_start_match+num_matches):
        # dump format: [ ...  (hero, [allies], [enemies])  ... ]
        match_id, start_time, radiant_win, heroes = match_dump[index][:]

        for hero_index in range(10):
            batch_heroes.append(id_fix(heroes[hero_index]))

        k = 0
        if not radiant_win:
            k = 5        
        for hero_index in range(k, k+5):
            allies = []
            enemies = []
            for ally_index in range(k, k+5):
                if ally_index == hero_index : continue
                allies.append(id_fix(heroes[ally_index]))
            for enemy_index in range(5-k, 10-k):
                enemies.append(id_fix(heroes[enemy_index]))
            batch_allies.append(allies)
            batch_enemies.append(enemies)

    batch_start_match = batch_start_match+num_matches

    return batch_heroes, batch_allies, batch_enemies

# create tf Graph
graph = tf.Graph()

# setting hyperparameters
batch_size = 400
embedding_size = 64
num_sampled = 64
learning_rate = 1.0
validation_set_size = 16
validation_set = [49, 53, 55, 54, 65, 64, 37, 77, 80, 82, 87, 98, 99, 100, 104, 107]

# create the model
with graph.as_default(), tf.device('/cpu:0'):
    training_batch_allies = tf.placeholder(tf.int32, shape=[batch_size, 4])
    training_batch_enemies = tf.placeholder(tf.int32, shape=[batch_size, 5])
    training_heroes = tf.placeholder(tf.int32, shape=[batch_size, 1])
    random_hero_samples = tf.constant(validation_set)

    hero_embeddings_lut = tf.Variable(tf.random_uniform([num_heroes, embedding_size], -1.0, 1.0))

    # embeddings for each sample, list of size batch_size
    ally_embeddings = tf.reduce_sum(tf.nn.embedding_lookup(hero_embeddings_lut, training_batch_allies), axis=1)
    enemy_embeddings = tf.reduce_sum(tf.nn.embedding_lookup(hero_embeddings_lut, training_batch_enemies), axis=1)

    # concatenate embeddings vertically to form input vector(rows represent unique samples)
    context_embedding = tf.concat([ally_embeddings, enemy_embeddings], 1)

    # f.c. output layer
    linear_layer_weights = tf.Variable(tf.truncated_normal(shape=[num_heroes, 2*embedding_size], stddev=2.0/math.sqrt(embedding_size)))
    linear_layer_biases = tf.Variable(tf.truncated_normal(shape=[num_heroes], stddev=2.0/math.sqrt(embedding_size)))

    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdagradOptimizer(learning_rate)

    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=linear_layer_weights, biases=linear_layer_biases, num_sampled=num_sampled,
                                                     num_classes=num_heroes, labels=training_heroes, inputs=context_embedding))
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

    # normalize embeddings:
    unit_embeddings = hero_embeddings_lut / tf.sqrt(tf.reduce_sum(tf.square(hero_embeddings_lut), 1, keep_dims=True))

    # calculate most similar heroes for heroes in validation set, to give an idea human interpretable state of embeddings
    validation_set_embeddings = tf.nn.embedding_lookup(unit_embeddings, random_hero_samples)
    validation_set_similarities = tf.matmul(validation_set_embeddings, tf.transpose(unit_embeddings))

def safe_makedir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

# parameters for training
num_steps = 400000
step_to_save = 500

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    
    average_loss = 0
    loss_hist = []

    for step in range(num_steps):

        # loop over to start if training set has been processed completely
        if batch_start_match+batch_size//5 >= num_available_matches:
            learning_rate = learning_rate*0.9
            batch_start_match = 0

        # load batch
        hero_group, allies_group, enemies_group = load_batch(batch_size)
        feed_dict = {training_heroes : np.array(hero_group).reshape(batch_size, 1), training_batch_allies : allies_group,
                     training_batch_enemies : enemies_group}

        # run optimizer
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)

        average_loss = average_loss + l

        if step == 0:
            print('initial loss: ', average_loss)

        # log
        if step % step_to_log == 0:
            average_loss = average_loss / step_to_log
            loss_hist.append(average_loss)
            print('step: ', step, ', avg. loss: ', average_loss)
            average_loss = 0

        # print 10 most similar heroes to each hero in validation set
        if step % 2000 == 0 or step == num_steps-1:
            similarities = validation_set_similarities.eval()
            for i in range(validation_set_size):
                nearest = (-similarities[i, :]).argsort()[1:9]
                valid_hero = validation_set[i]
                print('nearest to hero: ', hero_name_lut[valid_hero], ' : ', [hero_name_lut[near_index] for near_index in nearest])
    
    # evaluate trained embeddings
    final_embeddings = unit_embeddings.eval()
    # save tensorflow graph
    safe_makedir('embeddings_log')
    saver.save(session, 'embeddings_log/')

# plot the loss graph
# discard first 1000 or so steps in order to keep graph from becoming ill-scaled
start = 1000
plt.plot(np.arange(start, num_steps, 500), loss_hist[start//500:])
plt.savefig("loss_hist_only_win.png")

# save final embeddings on disk
pickle.dump(final_embeddings, open('final_embeddings', 'wb'))
