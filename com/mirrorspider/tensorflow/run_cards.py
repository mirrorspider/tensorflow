import os
import argparse
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random

from sklearn.model_selection import train_test_split

from com.mirrorspider.tensorflow.cards.draw import Pack
from com.mirrorspider.tensorflow.plotting import jitter
from com.mirrorspider.tensorflow.colours import TrueFalseCM

def display_cards(no_cards=9, dirty=False, dir=False, dirname="."):
    cards, ranks = Pack.shuffle(no_cards, dirty)
    if dir:
        np.save(dirname+"/cards.npy", cards)
        np.save(dirname+"/labels.npy", ranks)
    else:
        suits = Pack.get_suits()
        n = len(cards)
        sq = int(sqrt(n))
        plt.figure(figsize=(sq + 2,sq + 2))
        for i in range(n):
            plt.subplot(sq,sq,i+1)
            ci = cards[i]
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(ci)
            plt.xlabel(suits[ranks[i]])
        plt.show()

def setup_model(mid_layer_nodes=14):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(7,7,3)),
        keras.layers.Dense(mid_layer_nodes, activation=tf.nn.relu),
        keras.layers.Dense(4, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_image(i, predictions_array, true_label, img):
    class_names = Pack.get_suits()
    predictions_array, true_label, img = predictions_array[i], int(true_label[i]), img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
                                         color=color)
                                         
def plot_value_array(i, predictions_array, true_label, abbrev = False):
    predictions_array, true_label = predictions_array[i], int(true_label[i])
    plt.grid(False)
    sts = Pack.get_suits(abbrev)
    if abbrev:
        plt.xticks(range(4), sts)
    else:
        plt.xticks(range(4), sts, rotation=45)
    plt.yticks([])
    thisplot = plt.bar(range(4), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def plot_instance(i, predictions_array, test_labels, test_images):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions_array, test_labels, test_images)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions_array, test_labels)
    plt.subplots_adjust(bottom=0.25)
    plt.show()

def plot_nine_instances(predictions_array, test_labels, test_images):
    if len(predictions_array) > 9:
        predictions_array = predictions_array[0:9]
    num_rows = 3
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2*2*num_cols,2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows,2*num_cols,2*i+1)
        plot_image(i, predictions_array, test_labels, test_images)
        plt.subplot(num_rows,2*num_cols,2*i+2)
        plot_value_array(i, predictions_array, test_labels, True)
    #plt.subplots_adjust(bottom=0.25)
    plt.show()    

def plot_accuracy(predictions_array, test_labels, test_images):
    plt.figure()
    sts = Pack.get_suits(True)
    plt.xticks(range(4), sts)
    plt.yticks(range(4), sts)
    predictions = np.empty(predictions_array.shape[0])
    truth = np.empty(predictions_array.shape[0])
    alph = np.empty(predictions_array.shape[0])
    i=0
    for p in predictions_array:
        predictions[i] = np.argmax(p)
        if predictions[i] == test_labels[i]:
            truth[i] = 1
        else:
            truth[i] = 0
        i+=1
    tfc = TrueFalseCM().get_cmap()
    bounds = np.array([0, 1, 2])
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=2)
    jitter(test_labels, predictions, c=truth, cmap= tfc, norm=norm, alpha=0.2)
    plt.xlabel("actual value")
    plt.ylabel("prediction", rotation=90)
    plt.ylim(-0.5,3.5)
    plt.xlim(-0.5,3.5)
    plt.show()  
    
def net_demo(cards = 100, nodes = 14, dirty = False, train = False, test = False):
    # if test only don't dirty training set
    if test:
        train_images, train_labels = Pack.shuffle(cards)
    else:
        train_images, train_labels = Pack.shuffle(cards, dirty)
    
    train_images = train_images / 255.0
    
    mdl = setup_model(nodes)
    mdl.fit(train_images, train_labels, epochs=5)
    
    # if train only don't dirty test set
    if train:
        test_images, test_labels = Pack.shuffle(100)
    else:
        test_images, test_labels = Pack.shuffle(100, dirty)
    test_images = test_images / 255.0
    
    test_loss, test_acc = mdl.evaluate(test_images, test_labels)
    
    print('Test accuracy:', test_acc)
    
    predictions = mdl.predict(test_images)
    
    #plot_nine_instances(predictions, test_labels, test_images)
    plot_accuracy(predictions, test_labels, test_images)

    
def net_file_demo(nodes = 14, dirname=".", test_size=0.2, reproducible=False):

    if reproducible:
        # setup environment for reproducible results
        # also requires environment variables to be set:
        # CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0
        
        np.random.seed(42)
        random.seed(42)
        
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                      inter_op_parallelism_threads=1)
        
        tf.set_random_seed(42)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

    data_images = np.load(dirname+"/cards.npy")
    data_labels = np.load(dirname+"/labels.npy")
    
    train_images, test_images, train_labels, test_labels = train_test_split(
                                                                data_images, data_labels, test_size=test_size)
    train_images = train_images/255.0
    tst_images = test_images/255.0
    
    mdl = setup_model(nodes)
    mdl.fit(train_images, train_labels, epochs=5)
    test_loss, test_acc = mdl.evaluate(test_images, test_labels)
    
    print('Test accuracy:', test_acc)
    
    predictions = mdl.predict(test_images)
    
    plot_nine_instances(predictions, test_labels, test_images)    


    
if __name__ == "__main__":
    s = ['\nFor reproducible results the following environment variables must be set:',
          '\nCUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0']
    s = ''.join(s)
    prsr = argparse.ArgumentParser(description='Neural net demonstration.'  
                                               + '\nIf no arguments are supplied, a default net demo will be run.',
                                   epilog=s)
    grp = prsr.add_mutually_exclusive_group()
    grp.add_argument('--cards', nargs='?', type=int, default=argparse.SUPPRESS, const=9, metavar='deck_size')
    grp.add_argument('--net', nargs=2, type=int, default=argparse.SUPPRESS, metavar=('deck_size', 'no_nodes'))
    grp.add_argument('--net_dir', nargs=2, default=argparse.SUPPRESS, metavar=('no_nodes', 'test_proportion'))
    prsr.add_argument('--dirty', default=argparse.SUPPRESS, action='store_true')
    grp2 = prsr.add_mutually_exclusive_group()
    grp2.add_argument('--test_only', default=argparse.SUPPRESS, action='store_true')
    grp2.add_argument('--train_only', default=argparse.SUPPRESS, action='store_true')
    prsr.add_argument('--dir', nargs=1, type=str, metavar='directory', default=argparse.SUPPRESS)
    prsr.add_argument('--rep', action='store_true', default=argparse.SUPPRESS)

    args = prsr.parse_args()
    f = ("dir" in args)
    fn = "."
    if f:
        fn = args.dir[0]
    d = ("dirty" in args)
    test = ("test_only" in args)
    train = ("train_only" in args)
    
    rep = ("rep" in args)
    
    if "cards" in args:
        display_cards(args.cards, d, f, fn)
    elif "net" in args:
        net_demo(cards=args.net[0], nodes=args.net[1], dirty=d, train=train, test=test)
    elif "net_dir" in args:
        net_file_demo(nodes=int(args.net_dir[0]), dirname=fn, test_size=float(args.net_dir[1]), reproducible=rep)
    else:
        net_demo()

    

