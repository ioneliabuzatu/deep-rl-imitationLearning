import numpy as np
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm

from utils import Agent
from utils import DemonstrationDataset
from utils import run_episode
import config


class AgentNetwork(nn.Module):
    def __init__(self,
                 n_units_out,
                 hidden_size=512,
                 num_feature_maps=32,
                 kernel_size=6,
                 activation=nn.LeakyReLU()
                 ):
        super(AgentNetwork, self).__init__()

        num_inputs, input_width, _ = (1, 96, 96)

        self.activation = activation

        self.cnn_layer_1 = nn.Conv2d(num_inputs, num_feature_maps, kernel_size, stride=1, padding=0)
        self.cnn_layer_2 = nn.Conv2d(num_feature_maps, num_feature_maps * 2, int(kernel_size / 2), stride=2, padding=0)
        self.cnn_layer_3 = nn.Conv2d(num_feature_maps * 2, num_feature_maps, int(kernel_size / 2), stride=1, padding=0)

        feature_map_for_linear_layer = self.calculate_next_feature_map_size(
            [self.cnn_layer_1, self.cnn_layer_2, self.cnn_layer_3], input_width
        )
        assert feature_map_for_linear_layer >= 1, f"Ouch! the layer 3 feature is of size {feature_map_for_linear_layer}"

        self.linear_layer_1 = nn.Linear(num_feature_maps * feature_map_for_linear_layer ** 2, hidden_size)
        # self.linear_layer_3 = nn.Linear(hidden_size, hidden_size)

        self.output_layer = nn.Linear(hidden_size, n_units_out)

    def forward(self, x):
        x = self.activation(self.cnn_layer_1(x))
        x = self.activation(self.cnn_layer_2(x))
        x = self.activation(self.cnn_layer_3(x))
        x = x.view(-1, x.shape[1:].numel())
        x = self.activation(self.linear_layer_1(x))

        return self.output_layer(x)

    def calculate_next_feature_map_size(self, layers: list, feature_map_size: int):
        """ Initially the feature_map_size is the width of the input """
        for layer in layers:
            padding = layer.padding[0]
            stride = layer.stride[0]
            kernel = layer.kernel_size[0]
            feature_map_size = (feature_map_size - kernel + 2 * padding) / stride + 1
        return int(feature_map_size)


param_file = "./logdir_dagger/2021-04-18T13-20-20/params.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))
train_set = DemonstrationDataset("./data/train.npz", img_stack=1)
net = AgentNetwork(n_units_out=len(train_set.action_mapping))
net = net.to(device)
train_agent = Agent(net, train_set.action_mapping, device, img_stack=1)
train_agent.load_param(param_file)
n_eval_episodes = 10
scores = []
for i in tqdm(range(n_eval_episodes), desc="Episode"):
    scores.append(run_episode(train_agent, show_progress=False, record_video=False))
    wandb.log({"Ten runs evaluation": scores[-1]}, step=i)
    print("... mean Score: %.2f (Std: %.2f)" % (np.mean(scores), np.std(scores)))
print("Final Mean Score: %.2f (Std: %.2f)" % (np.mean(scores), np.std(scores)))
