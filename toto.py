import sys
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QApplication, QLabel, QVBoxLayout, QPushButton
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


def training_model(train=0, epoch=5):
    """
    :param train: int 0 or 1
        Set to 1 if you want to force training or not
        Note that there will be one if no model found
    :param epoch: int
        Defines the number of iterations over the entire input data
        Note that epoch is only used in case of a training...
    """


class MainWindow(QDialog):

    def __init__(self):
        super().__init__()

        self.label = QLabel()
        self.button = QPushButton()
        self.button.setText('Save')
        self.button.clicked.connect(self.saving_file)
        self.canvas = QPixmap(400, 300)
        self.canvas.fill(Qt.white)
        self.label.setPixmap(self.canvas)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        self.setLayout(layout)

        self.last_x, self.last_y = None, None

    def saving_file(self):
        pixmap = self.label.pixmap()
        png_path = os.path.join(os.getcwd(), "my_number.jpg")
        print(png_path)
        print(pixmap.save(png_path, "jpg"))
        # png_array = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        # new_array = cv2.resize(png_array, (28, 28))
        # plt.imshow(new_array, cmap="gray")
        # plt.show()

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # Normalizing the RGB codes by dividing it to the max RGB value.
        x_train /= 255
        x_test /= 255

        if not os.path.exists(os.path.join(os.getcwd(), "Awesome first try")):
            # -------------------------- CREATE MODEL ------------------------------

            model = Sequential()
            model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            # Flattening the 2D arrays for fully connected layers
            model.add(Flatten())
            model.add(Dense(128, activation=tf.nn.relu))
            model.add(Dropout(0.2))
            model.add(Dense(10,activation=tf.nn.softmax))

            # ----------------------------------------------------------------------

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            model.fit(x=x_train,y=y_train, epochs=1)
            model.save('Awesome first try')

            # ----------------------------------------------------------------------

        model = tf.keras.models.load_model("Awesome first try")
        file = png_path
        val_loss, val_acc = model.evaluate(x_test, y_test)

        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))
        image = image.astype('float32')
        image = image.reshape(1, 28, 28, 1)
        image = 255 - image
        image /= 255

        pred = model.predict(image, batch_size=1)

        plt.title(f'There\'s {(val_acc*100):.2f}% that\'s a: {pred.argmax()}\n')
        plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.imshow(image.reshape(28, 28), cmap='Greys')
        plt.show()
        self.close()

    def mouseMoveEvent(self, e):
        # First event
        if self.last_x is None:
            self.last_x = e.x()
            self.last_y = e.y()
            # Ignore the first time
            return

        painter = QPainter(self.label.pixmap())
        p = painter.pen()
        p.setWidth(20)
        painter.setPen(p)
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        # Update the origin for next time
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()

