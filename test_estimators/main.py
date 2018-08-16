import cv2
import tensorflow as tf
import numpy as np
import random

NB_TRAIN_IMG = 20
NB_TEST_IMG = 10


def get_mnist_dataset():
    """Build mnist dataset for inceptionV3's input shape requirement:
    []
    return:
    x_train, y_train, x_test, y_test
    """
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test


def resize_data(data):
    data_upscaled = np.zeros((data.shape[0], 140, 140, 3))
    for i, img in enumerate(data):
        large_img = cv2.resize(img, dsize=(140, 140), interpolation=cv2.INTER_CUBIC)
        data_upscaled[i] = large_img
    return data_upscaled

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# Instantiate a Keras inception v3 model.
keras_inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights=None)
# Compile model with the optimizer, loss, and metrics you'd like to train with.
keras_inception_v3.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                          loss='categorical_crossentropy',
                          metric='accuracy')
# Create an Estimator from the compiled Keras model. Note the initial model
# state of the keras model is preserved in the created Estimator.
est_inception_v3 = tf.keras.estimator.model_to_estimator(keras_model=keras_inception_v3)

# Treat the derived Estimator as you would with any other Estimator.
# First, recover the input name(s) of Keras model, so we can use them as the
# feature column name(s) of the Estimator input function:
keras_inception_v3.input_names  # print out: ['input_1']
# Once we have the input name(s), we can create the input function, for example,
# for input(s) in the format of numpy ndarray:

## DATA
ind_train = random.sample(list(range(x_train.shape[0])), NB_TRAIN_IMG)
train_data = resize_data(x_train)[ind_train]
train_labels = y_train[ind_train]

ind_test = random.sample(list(range(x_test.shape[0])), NB_TEST_IMG)
test_data = resize_data(x_test)[ind_test]
test_labels = y_test[ind_test]


## TRAINING` GRAPH
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input_1": train_data},
    y=train_labels,
    num_epochs=1,
    shuffle=True)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input_1": test_data},
    y=test_labels,
    num_epochs=1,
    shuffle=False)


## TRAIN
print(est_inception_v3.train(input_fn=train_input_fn, steps=100))
print(est_inception_v3.evaluate(input_fn=test_input_fn))
