import sys
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
from PyQt5.QtGui import QPixmap, QPainter, QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QApplication, QLabel, QGridLayout, QPushButton
# from tf.keras.models import Sequential
# from tf.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


class MainWindow(QDialog):

    def __init__(self):
        super().__init__()
        self.label = QLabel()
        self.canvas = QPixmap(400, 300)
        self.last_x, self.last_y = None, None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('My_Mnist')
        self.setWindowIcon(QIcon(os.path.join(os.getcwd(), 'logo')))
        button_1 = QPushButton()
        button_2 = QPushButton()
        button_3 = QPushButton()
        button_1.setText('&Run')
        button_2.setText('&Refresh')
        button_3.setText('&Exit')
        self.canvas.fill(Qt.white)
        self.label.setPixmap(self.canvas)
        button_1.clicked.connect(self.run_prediction)
        button_2.clicked.connect(self.refreshing)
        button_3.clicked.connect(self.close)
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label, 0, 0, 0, 0)
        layout.addWidget(button_1, 6, 0, 1, 1)
        layout.addWidget(button_2, 6, 1, 1, 1)
        layout.addWidget(button_3, 6, 2, 1, 1)
        self.setLayout(layout)

    def run_prediction(self):
        pixmap = self.label.pixmap()
        png_path = os.path.join(os.getcwd(), "my_number.jpg")
        pixmap.save(png_path, "jpg")

        # Load the standard dataset 'mnist' => composed by 60 000 image of hand-written digits from 0 to 9
        # Unpacked as 28 x 28 int arrays, for each pixel from 0 to 255
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Reshape all the arrays to the "standard"
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

        # Define type
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # Normalizing the RGB codes by dividing it to the max RGB value.
        x_train /= 255
        x_test /= 255

        # Load model
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

    def refreshing(self):
        self.label.pixmap().fill(Qt.white)
        self.update()

    def mouseMoveEvent(self, e):
        # First event
        if self.last_x is None:
            self.last_x = e.x()
            self.last_y = e.y()
            # Ignore the first time
            return

        painter = QPainter(self.label.pixmap())
        p = painter.pen()
        p.setWidth(30)
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


def train_model(epochs):
    # Load the standard dataset 'mnist' => composed by 60 000 image of hand-written digits from 0 to 9
    # Unpacked as 28 x 28 int arrays, for each pixel from 0 to 255
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    # Reshape all the arrays to the "standard"
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

    # Define type
    x_train = x_train.astype('float32')

    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    # -------------------------- CREATE MODEL ------------------------------

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(28, kernel_size=(3, 3), input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Flattening the 2D arrays for fully connected layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=epochs)
    model.save('Awesome first try')

    # ----------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='py My_Mnist')
    parser.add_argument('epochs', type=int, default=0, choices=range(2), nargs='?',
                        help='Number of epochs applied')
    args = parser.parse_args()
    if not os.path.exists(os.path.join(os.getcwd(), "Awesome first try")) or args.epochs:
        train_model(args.epochs)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()

