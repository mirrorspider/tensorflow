import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from random import randint
from pathlib import Path

print(tf.__version__)

class FirstNetwork:

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def load_data(self):
        self.fashion_mnist = keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.fashion_mnist.load_data()
        
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0
     
    def build_model(self):
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.sparse_categorical_crossentropy,
                           metrics=['accuracy'])
    
    def train_model(self):
        self.model.fit(self.train_images, self.train_labels, epochs=5)
        self.model.save('./my_model.h5')
        
    def get_model(self):
        h = Path("./my_model.h5")
        if h.is_file():
            self.model = keras.models.load_model("./my_model.h5")
        else:
            self.build_model()
            self.train_model()
        
    def evaluate_model(self):
        self.test_loss, self.test_acc = self.model.evaluate(self.test_images, self.test_labels)
        return self.test_acc
    
    def get_predictions(self):
        self.predictions = self.model.predict(self.test_images)
    
    def plot_image(self, i, predictions_array, true_label, img):
        class_names = FirstNetwork.class_names
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        
        plt.imshow(img, cmap=plt.cm.binary)
        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'
        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100*np.max(predictions_array),
                                             class_names[true_label]),
                                             color=color)
                                             
    def plot_value_array(self, i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)
        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')
        
        
    def plot_instance(self, i=0):
        i = int(randint(0, len(self.predictions) - 1))
        plt.figure(figsize=(6,3))
        plt.subplot(1, 2, 1)
        self.plot_image(i, self.predictions, self.test_labels, self.test_images)
        plt.subplot(1, 2, 2)
        self.plot_value_array(i, self.predictions, self.test_labels)
        plt.show()
        
if __name__ == "__main__":
    n = FirstNetwork()
    n.load_data()
    n.get_model()
    acc = n.evaluate_model()
    print('Test accuracy:', acc)
    n.get_predictions()
    n.plot_instance()
    
