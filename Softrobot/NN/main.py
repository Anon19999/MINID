import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from neural_network import dataset, loss, model, settings, test, train, utils

import config

utils.set_seed(seed=settings.SEED)


def train_model():
    forward_network = model.create_linear_sequential_model(layers_size=config.MODEL_LAYERS)
    forward_network.apply(fn=utils.custom_init_weights_and_biases)
    forward_network = forward_network.to(settings.DEVICE)

    Design_data = np.empty((0,40))
    Prformance_data = np.empty((0,103,2))
    for data_n in range(5):
        path_name = 'data/soft-robot/input_%d.npy' %(data_n+1)
        Design_data = np.append(Design_data, np.load(path_name), axis=0)
        path_name = 'data/soft-robot/output_%d.npy' % (data_n + 1)
        Prformance_data = np.append(Prformance_data, np.load(path_name), axis=0)

    x_train_tensor = torch.from_numpy(Design_data).float()
    y_train_tensor = torch.from_numpy(Prformance_data).float()
    y_train_tensor = torch.reshape(y_train_tensor, (y_train_tensor.size(0), -1))

    training_tensor_datasets, testing_tensor_datasets = dataset.get_training_testing_tensor_datasets(
        tensor_dataset=dataset.create_tensor_dataset(x_train_tensor, y_train_tensor),
        train_split_ratio=0.8
    )
    training_data_loader = dataset.create_data_loader(tensor_dataset=training_tensor_datasets, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)
    testing_data_loader = dataset.create_data_loader(tensor_dataset=testing_tensor_datasets, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)

    optimizer = optim.Adam(forward_network.parameters(), lr=config.LEARNING_RATE)
    criterion = loss.custom_sum_mse

    train.train_forward_network(
        model=forward_network,
        criterion=criterion,
        optimizer=optimizer,
        learning_rate=config.LEARNING_RATE,
        learning_rate_decay=config.LEARNING_RATE_DECAY,
        train_dataset_loader=training_data_loader,
        test_dataset_loader=testing_data_loader,
        number_of_epochs=config.NUMBER_OF_EPOCHS
    )

    test.test_forward_network(
        model=forward_network,
        criterion=criterion,
        test_dataset_loader=testing_data_loader
    )

    utils.save_model(model=forward_network, path=f'{config.MODEL_NAME}.ckpt')


def save_model_weights_and_biases():
    forward_network = model.create_linear_sequential_model(layers_size=config.MODEL_LAYERS)
    forward_network = utils.load_model(model=forward_network, path=f'{config.MODEL_NAME}.ckpt')
    utils.save_model_weights_and_biases(model=forward_network, path=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'MILP', 'Data', 'soft_robot_w_b_128_128.mat'), key='w_numpy')


if os.path.exists(f'{config.MODEL_NAME}.ckpt'):
    save_model_weights_and_biases()
else:
    train_model()
    save_model_weights_and_biases()
