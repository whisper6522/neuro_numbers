import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np


# Function to display an image and prediction
def plot_image(image, label, prediction):
    plt.imshow(image, cmap='gray')
    plt.title(f'Actual: {label}, Predicted: {prediction}')
    plt.show()


# Download EMNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Bring the data to the range from 0 to 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert labels to categorical ones
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# "Pull" the result into a flat vector
model.add(Flatten())

# Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))  # 26 output neurons

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()  # Display brief information about the model

history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=10,
                    validation_data=(x_test, y_test),
                    verbose=1)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Example: output multiple images and their predictions
for i in range(5):
    index = np.random.randint(0, len(x_test))
    image = x_test[index].reshape(28, 28)
    actual_label = np.argmax(y_test[index])
    predicted_label = np.argmax(model.predict(x_test[index:index + 1]))
    plot_image(image, actual_label, predicted_label)

# Save model
model.save('my_model.keras')