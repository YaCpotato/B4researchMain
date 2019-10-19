from deepaugment.deepaugment import DeepAugment

from keras.datasets import cifar10

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
deepaug = DeepAugment(images=x_train, labels=y_train, config=my_config)

best_policies = deepaug.optimize(300)
