import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

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

    X = dataset.load_tensor_from_mat_file(path='values_Si_Tio2_Au_Ag_H2O.mat', key='values')
    X = torch.div(X, 70)
    Y = dataset.load_tensor_from_mat_file(path='spects_Si_Tio2_Au_Ag_H2O.mat', key='myspects')
    Y = torch.transpose(Y, 0, 1)

    training_tensor_datasets, testing_tensor_datasets = dataset.get_training_testing_tensor_datasets(
        tensor_dataset=dataset.create_tensor_dataset(X, Y),
        train_split_ratio=0.9
    )
    training_data_loader = dataset.create_data_loader(tensor_dataset=training_tensor_datasets, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)
    testing_data_loader = dataset.create_data_loader(tensor_dataset=testing_tensor_datasets, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)

    optimizer = optim.Adam(forward_network.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()

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
    utils.save_model_weights_and_biases(model=forward_network, path=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'MILP', 'Data', 'photonic_net_3_100_50_100_w_b.mat'), key='w_numpy')


if os.path.exists(f'{config.MODEL_NAME}.ckpt'):
    save_model_weights_and_biases()
else:
    train_model()
    save_model_weights_and_biases()
