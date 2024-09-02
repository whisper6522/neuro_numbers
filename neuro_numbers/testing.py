from keras.models import load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Function to display an image and prediction
def plot_image(image, label, prediction):
    plt.imshow(image, cmap='gray')
    plt.title(f'Actual: {label}, Predicted: {prediction}')
    plt.show()


model = load_model('my_model.keras')

# Download EMNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Bring the data to the range from 0 to 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert labels to categorical ones
y_train = tf.keras.utils.to_categorical(y_train, 26)
y_test = tf.keras.utils.to_categorical(y_test, 26)

# Example: output multiple images and their predictions
for i in range(5):
    index = np.random.randint(0, len(x_test))
    image = x_test[index].reshape(28, 28)
    actual_label = np.argmax(y_test[index])
    predicted_label = np.argmax(model.predict(x_test[index:index + 1]))
    plot_image(image, actual_label, predicted_label)
