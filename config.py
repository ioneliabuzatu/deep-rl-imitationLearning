import torch

import experiment_buddy

learning_rate = 0.001
weight_decay = 0.1
batchsize = 32
n_epochs = 1

experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy(
    "",
    sweep_yaml="",
    proc_num=1,
    wandb_kwargs={"entity": "ionelia"}
)
