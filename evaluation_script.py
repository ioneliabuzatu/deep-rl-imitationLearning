# PyTorch imports
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import onnx
from onnx2pytorch import ConvertModel

# Environment import and set logger level to display error only
import gym
from gym import logger as gymlogger

from pyvirtualdisplay import Display

import argparse
import os

gymlogger.set_level(40)  # error only
pydisplay = Display(visible=0, size=(640, 480))
pydisplay.start()

# Seed random number generators
if os.path.exists("seed.rnd"):
    with open("seed.rnd", "r") as f:
        seed = int(f.readline().strip())
    np.random.seed(seed)
    torch.manual_seed(seed)
else:
    seed = None

action_mapping = [
    (0, 0, 0),  # no action
    (0, 0.5, 0),  # half throttle
    (0, 1, 0),  # full trottle
    (0, 0, 0.5),  # half break
    (0, 0, 1),  # full break
    # steering left with throttle/break control
    (-0.5, 0, 0),  # half left
    (-1, 0, 0),  # full left
    (-0.5, 0.5, 0),  # half left
    (-1, 0.5, 0),  # full left
    (-0.5, 1, 0),  # half left
    (-1, 1, 0),  # full left
    (-0.5, 0, 0.5),  # half left
    (-1, 0, 0.5),  # full left
    (-0.5, 0, 1),  # half left
    (-1, 0, 1),  # full left
    # steering right with throttle/break control
    (0.5, 0, 0),  # half right
    (1, 0, 0),  # full right
    (0.5, 0.5, 0),  # half right
    (1, 0.5, 0),  # full right
    (0.5, 1, 0),  # half right
    (1, 1, 0),  # full right
    (0.5, 0, 0.5),  # half right
    (1, 0, 0.5),  # full right
    (0.5, 0, 1),  # half right
    (1, 0, 1)  # full right
]


# Convert RBG image to grayscale and normalize by data statistics
def rgb2gray(rgb, norm=True):
    # rgb image -> gray [0, 1]
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    if norm:
        # normalize
        gray = gray / 128. - 1.
    return gray

# Hide the heads up display at the bottom of the screen for evaluation
def hide_hud(img):
    img[84:] = 0
    return img

class Env():
    """
    Environment wrapper for CarRacing
    """

    def __init__(self, img_stack, seed=None):

        self.gym_env = gym.make('CarRacing-v0')
        self.env = self.gym_env
        self.action_space = self.env.action_space
        self.img_stack = img_stack
        if seed is not None:
            self.env.seed(seed)


    def reset(self, raw_state=False):
        self.env = self.gym_env
        self.rewards = []
        img_rgb = self.env.reset()
        img_gray = hide_hud(rgb2gray(img_rgb))
        self.stack = [img_gray] * self.img_stack
        if raw_state:
            return np.array(self.stack), np.array(img_rgb)
        else:
            return np.array(self.stack)

    def step(self, action, raw_state=False):
        # for i in range(self.img_stack):
        img_rgb, reward, done, _ = self.env.step(action)
        # accumulate reward
        self.rewards.append(reward)
        # if no reward recently, end the episode
        die = True if np.mean(self.rewards[-np.minimum(100, len(self.rewards)):]) <= -1 else False
        if done or die:
            self.env.close()
        img_gray = hide_hud(rgb2gray(img_rgb))
        # add to frame stack
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.img_stack
        # --
        if raw_state:
            return np.array(self.stack), np.sum(self.rewards[-1]), done, die, img_rgb
        else:
            return np.array(self.stack), np.sum(self.rewards[-1]), done, die

    def close(self):
        self.env.close()



class Agent():
    """
    Agent for training
    """

    def __init__(self, net):
        self.net = net

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.net(state)
            if type(action_probs) is tuple:
                action_probs = action_probs[0]
        action, action_idx, a_logp = Agent.sample_action(action_probs)
        a_logp = a_logp.item()

        return action, action_idx, a_logp

    @staticmethod
    def sample_action(probs):
        m = Categorical(logits=probs.to("cpu"))
        action_idx = m.sample()
        a_logp = m.log_prob(action_idx)
        action = action_mapping[action_idx.squeeze().cpu().numpy()]
        return action, action_idx, a_logp


def run_episode(agent, img_stack, seed=None):
    env = Env(img_stack=img_stack, seed=seed)
    state = env.reset()
    score = 0
    done_or_die = False
    while not done_or_die:
        action, action_idx, a_logp = agent.select_action(state)
        state, reward, done, die = env.step(action)
        score += reward

        if done or die:
            done_or_die = True
    env.close()

    return score


if __name__ == "__main__":
    N_EPISODES = 50
    IMG_STACK = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str, default="./logdir_dagger/2021-04-18T13-20-20/params.pkl")
    args = parser.parse_args()
    model_file = args.submission

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Network
    net = ConvertModel(onnx.load(model_file))
    net = net.to(device)
    net.eval()
    agent = Agent(net)

    scores = []
    for i in range(N_EPISODES):
        if seed is not None:
            seed = np.random.randint(1e7)
        scores.append(run_episode(agent, IMG_STACK, seed=seed))

    print(np.mean(scores))

