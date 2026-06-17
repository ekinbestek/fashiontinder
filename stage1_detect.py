"""
=============================================================
 FASHIONTINDER - STAGE 1: Clothing Detection
=============================================================

GOAL:
  Take a clothing image and predict its type
  (T-shirt, trouser, dress, shoe...).

DATASET:
  Fashion-MNIST -> 70,000 small (28x28) grayscale clothing images, 10 classes.
  It comes built into TensorFlow, so there is no manual download needed.

HOW TO RUN:
  In the terminal (with venv active):  python stage1_detect.py
"""

# ---------------------------------------------------------------
# 1) LIBRARIES
# ---------------------------------------------------------------
# tensorflow -> the neural network
# numpy      -> number crunching
# matplotlib -> showing images / results
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# 2) LOAD THE DATA
# ---------------------------------------------------------------
# Fashion-MNIST is split in two:
#   - train: images the model learns from
#   - test:  images the model has never seen (to check honestly)
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Names of the 10 categories (labels arrive as numbers 0-9).
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# ---------------------------------------------------------------
# 3) PREPARE THE DATA
# ---------------------------------------------------------------
# Each pixel is a value 0-255. Neural networks learn more easily
# with small 0-1 values, so we divide by 255. This is "normalization".
train_images = train_images / 255.0
test_images = test_images / 255.0


# ---------------------------------------------------------------
# 4) BUILD THE MODEL
# ---------------------------------------------------------------
# We build the model layer by layer:
#
#   Flatten  -> turns the 28x28 square into one row of 784 numbers
#   Dense128 -> a thinking layer of 128 neurons (relu zeroes out negatives)
#   Dense10  -> 10 outputs, one score per category
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Tell the model HOW to learn:
#   optimizer -> the learning method (adam = a good, standard choice)
#   loss      -> the error measure (how wrong it currently is)
#   metrics   -> the success measure we track (accuracy)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


# ---------------------------------------------------------------
# 5) TRAIN THE MODEL
# ---------------------------------------------------------------
# epochs = how many times the model goes through all training images.
# 5 is a quick start; increase it later for higher accuracy.
print("\n>>> Training the model...\n")
model.fit(train_images, train_labels, epochs=5)


# ---------------------------------------------------------------
# 6) TEST THE MODEL
# ---------------------------------------------------------------
# We measure honest accuracy on images it has never seen.
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\n>>> Test accuracy: {test_acc:.2%}\n")


# ---------------------------------------------------------------
# 7) PREDICT ONE IMAGE + SHOW IT
# ---------------------------------------------------------------
# The raw output is 10 scores; Softmax turns them into probabilities.
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# Pick one image from the test set (change the index to try others).
index = 0
img = test_images[index]

# The model expects a batch (list) of images, so we wrap it in one.
img_batch = np.expand_dims(img, 0)

predictions = probability_model.predict(img_batch)
predicted_label = np.argmax(predictions[0])   # category with the highest score
confidence = np.max(predictions[0])            # probability of that category

print(f">>> Prediction : {class_names[predicted_label]}")
print(f">>> Confidence : {confidence:.2%}")
print(f">>> True answer: {class_names[test_labels[index]]}")

# Show the image on screen with the prediction in the title.
plt.figure()
plt.imshow(img, cmap='gray')
plt.title(f"Prediction: {class_names[predicted_label]} ({confidence:.0%})")
plt.axis('off')
plt.show()