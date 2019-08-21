import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import begin
import os


@begin.start
def main(train=0, epoch=5):
    """
    :param train: int 0 or 1
        Set to 1 if you want to force training or not
        Note that there will be one if no model found
    :param epoch: int
        Defines the number of iterations over the entire input data
        Note that epoch is only used in case of a training...
    :return:
    """

    # Load the standard dataset 'mnist' dataset => composed by 60 000 image of hand-written digits from 0 to 9
    # Unpacked as 28 x 28 int arrays, for each pixel from 0 to 255
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Makes the model way more effective to normalize the data from 0 to 1 instead of 0 to 255
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    if int(train) or not os.path.isfile('Awesome first try'):
        # Model type
        model = tf.keras.models.Sequential()

        # 1st layer: to flatten our data and additionally the 1st layer can specify the input
        # Our input shape is an array of 28 by 28 that we'll be no longer needed to provide to the other layers
        model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

        # 2nd layer: Define the numbers of "neurones" we'll be using and additionally the function we'd like to apply
        # In this case this is the Rectified Linear Unit
        model.add(tf.keras.layers.Dense(64, activation='relu'))


        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        # optimizer='adam': Configure a model for mean-squared error regression.
        #
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=epoch)
        model.save('Awesome first try')

    new_model = tf.keras.models.load_model('Awesome first try')
    predictions = new_model.predict(x_test)
    val_loss, val_acc = new_model.evaluate(x_test, y_test)

    random_num = random.randint(0, 10000)
    plt.title(f'Random data chosen: {random_num}/{len(x_test)}\n'
              f'There\'s {(val_acc*100):.2f}% that\'s a: {np.argmax(predictions[random_num])}\n')
    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.imshow(x_test[random_num])
    plt.show()

# Docs (relu) : https://stackoverflow.com/questions/43504248/what-does-relu-stand-for-in-tf-nn-relu
#               https://www.geeksforgeeks.org/python-tensorflow-nn-relu-and-nn-leaky_relu/
# Docs (tensorflow) : https://www.tensorflow.org/guide/keras
# Docs (sentdex) : https://youtu.be/wQ8BIBpya2k
