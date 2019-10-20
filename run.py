from deepaugment.deepaugment import DeepAugment
from tensorflow.python.client import device_lib
import time
from keras.datasets import cifar10
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

print(device_lib.list_local_devices())

# my configuration
my_config = {
    "model": "basiccnn",
    "method": "bayesian_optimization",
    "train_set_size": 2000,
    "opt_samples": 3,
    "opt_last_n_epochs": 3,
    "opt_initial_points": 10,
    "child_epochs": 10,
    "child_first_train_epochs": 0,
    "child_batch_size": 64
}

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# X_train.shape -> (N, M, M, 3)
# y_train.shape -> (N)
mytime = time.time()
deepaug = DeepAugment(images=x_train, labels=y_train, config=my_config)
best_policies = deepaug.optimize(300)
myseconds = time.time() - mytime
