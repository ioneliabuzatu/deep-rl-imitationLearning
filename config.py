import torch

import experiment_buddy

beta = 0.7
learning_rate = 0.001
weight_decay = 0.1
n_epochs = 1
n_dagger_iterations = 1
batch_size = 32

experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy(
    "mila",
    sweep_yaml="",
    proc_num=1,
    wandb_kwargs={"entity": "ionelia"}
)
