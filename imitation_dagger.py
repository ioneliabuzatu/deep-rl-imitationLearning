import os

import numpy as np
import onnx
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from onnx2pytorch import ConvertModel
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import wandb
from utils import Agent
from utils import DemonstrationDataset
from utils import Env
from utils import Logger
from utils import plot_metrics
from utils import run_episode
from utils import save_as_onnx
from utils import train
from utils import val

wandb.run = config.tensorboard.run

sns.set()


if os.path.exists("./data/val.npz"):
    expert_onnx_filepath = "./data/expert.onnx"
    train_npz_filepath = "./data/train.npz"
    val_npz_filepath = "./data/val.npz"
else:
    expert_onnx_filepath = "/home/mila/g/golemofl/data/expert.onnx"
    train_npz_filepath = "/home/mila/g/golemofl/data/train.npz"
    val_npz_filepath = "/home/mila/g/golemofl/data/val.npz"


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

        self.linear_layer_1 = nn.Linear(num_feature_maps * feature_map_for_linear_layer ** 2, n_units_out)

        self.output_layer = nn.Linear(hidden_size, n_units_out)

    def forward(self, x):
        x = self.activation(self.cnn_layer_1(x))
        x = self.activation(self.cnn_layer_2(x))
        x = self.activation(self.cnn_layer_3(x))
        x = x.view(-1, x.shape[1:].numel())
        # x = self.activation(self.linear_layer_1(x))
        return self.linear_layer_1(x)
        # return self.output_layer(x)

    @staticmethod
    def calculate_next_feature_map_size(layers: list, feature_map_size: int):
        """ Initially the feature_map_size is the width of the input """
        for layer in layers:
            padding = layer.padding[0]
            stride = layer.stride[0]
            kernel = layer.kernel_size[0]
            feature_map_size = (feature_map_size - kernel + 2 * padding) / stride + 1
        return int(feature_map_size)


def dagger(current_policy, expert_policy, beta=1.):
    # Set up environment and result lists
    env = Env(img_stack=1, record_video=False)
    state = env.reset()
    # the expert agent was trained using the last 4 frames as input so we need to account for this
    state_log = [state.squeeze()] * 4
    frame_log = []
    action_log = []

    # Use this method to prepare the state for both policies
    def prepare_state_for_policy(policy, state_log):
        return np.array(state_log[-policy.img_stack:])

    #### YOUR CODE HERE ####
    # Implement DAgger algorithm here
    # 1) Choose a policy according to the DAgger algorithm (use beta)
    # 2) Sample trajectory with this policy (here, we create one episode)
    #     -> call "policy.select_action(state)" to predict the action for the current state
    #     -> call "prepare_state_for_policy(policy, state_log)" to get the current state
    #        in the correct format regardless of the chosen policy
    # 3) Label the states of this trajectory with your expert
    #     -> the expert policy always expects 4 frames, pass the state as "np.array(state_log)"

    # 1: Choose policy

    def choose_policy():
        prob = np.random.uniform()
        if prob <= beta:
            return expert_policy
        else:
            return current_policy

    policy = choose_policy()

    done_or_die = False
    while not done_or_die:
        # 2: Sample trajectory:
        #   -> select action
        #   -> perform action in the environment
        #   -> pass "raw_state=True" to env.step() so you can record the 
        #      original frames without pre-processing (which we need to aggregate datasets later)

        #### YOUR CODE HERE ####

        state = prepare_state_for_policy(policy, state_log)
        current_action, action_idx, a_logp = policy.select_action(state)
        state, r, done, die, frame = env.step(current_action, raw_state=True)

        # Always keep the last four frames in the log as we need them for the expert
        state_log.pop(0)
        state_log.append(state.squeeze())

        # 3: label the current state with the expert policy

        #### YOUR CODE HERE ####

        raw_state = np.array(state_log)
        expert_action, action_idx, a_logp = expert_policy.select_action(raw_state)

        # Keep a record of states and actions so we can use them for training our agent
        frame_log.append(frame)
        action_log.append(expert_action)

        # Check when you're done
        if done or die:
            done_or_die = True

    env.close()
    return np.array(frame_log), np.array(action_log)


beta = config.beta
learning_rate = config.learning_rate
weight_decay = config.weight_decay
n_epochs = config.n_epochs
n_dagger_iterations = config.n_dagger_iterations
batch_size = config.batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))

loss_func = nn.CrossEntropyLoss().to(device)

expert_net = ConvertModel(onnx.load(expert_onnx_filepath)).to(device)

# Freeze expert weights
for p in expert_net.parameters():
    p.requires_grad = False

# Specify the google drive mount here if you want to store logs and weights there (and set it up earlier)
logger = Logger("logdir_dagger")
print("Saving state to {}".format(logger.basepath))

train_set = DemonstrationDataset(train_npz_filepath, img_stack=1)
val_set = DemonstrationDataset(val_npz_filepath, img_stack=1)
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=True, drop_last=False,
                          pin_memory=True)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=2, shuffle=False, drop_last=False, pin_memory=True)

net = AgentNetwork(n_units_out=len(train_set.action_mapping)).to(device)
num_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Trainable Parameters: {}".format(num_trainable_params))

optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

# for i_ep in range(n_epochs):
#     if i_ep > 0:
#         plot_metrics(logger)
#     sample_frame = train(net, train_loader, loss_func, optimizer, logger, i_ep + 1, device)
#     val_loss, val_acc = val(net, val_loader, loss_func, logger, i_ep + 1, device)

train_agent = Agent(net, train_set.action_mapping, device, img_stack=1)
# train_agent.load_param("logdir_dagger/2021-04-18T13-20-20/params.pkl")
expert_agent = Agent(expert_net, train_set.action_mapping, device, img_stack=4)

for i_ep in range(n_epochs):
    print("Saving state to {}".format(logger.basepath))
    for _ in tqdm(range(n_dagger_iterations), desc="Generating expert samples"):
        frames, actions = dagger(train_agent, expert_agent, beta=beta)
        train_set.append(frames, actions)

    if i_ep > 0:
        plot_metrics(logger)

    sample_frame = train(net, train_loader, loss_func, optimizer, logger, i_ep + 1, device)

    val_loss, val_acc = val(net, val_loader, loss_func, logger, i_ep + 1, device)

    wandb.log({"validation loss": val_loss}, step=i_ep)
    wandb.log({"validation accuracy": val_acc}, step=i_ep)

    # store logs
    logger.dump()
    if i_ep % 2 == 0:
        torch.save(net.state_dict(), logger.param_file)
        save_as_onnx(net, sample_frame, logger.onnx_file)
        wandb.save(logger.param_file)
        wandb.save(logger.onnx_file)
    # if i_ep % 1 == 0:
    #     score_epoch = run_episode(train_agent, show_progress=False, record_video=False)
    #     print(f"score spoch {i_ep} : {score_epoch}")

# store the dagger agent
print(f"Saving the onnx model: {logger.onnx_file}")
save_as_onnx(net, sample_frame, logger.onnx_file)
wandb.save(logger.onnx_file)

print("Saved state to {}".format(logger.basepath))
print("[%03d] Validation Loss: %.4f Accuracy: %.4f" % (i_ep + 1, val_loss, val_acc))
plot_metrics(logger)

scores = []
for i in tqdm(range(50), desc="Episode"):
    scores.append(run_episode(train_agent, show_progress=False, record_video=False))
    wandb.log({"Mean_Score_Evaluation": np.mean(scores)}, step=i)
print("Final Mean Score and standard deviation: %.2f (Std: %.2f)" % (np.mean(scores), np.std(scores)))
