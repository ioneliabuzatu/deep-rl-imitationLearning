import os

import gym
import numpy as np
import torch
from gym.wrappers import Monitor
from torch.distributions import Categorical
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from tqdm import tqdm


def disable_view_window():
    from gym.envs.classic_control import rendering
    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, **kwargs)
        self.window.set_visible(visible=False)

    rendering.Viewer.__init__ = constructor


def run_episode(agent, show_progress=True, record_video=True):
    env = Env(img_stack=1, record_video=record_video)
    disable_view_window()
    state = env.reset()
    score = 0
    done_or_die = False
    if show_progress:
        progress = tqdm(desc="Score: 0")
    while not done_or_die:
        action, action_idx, a_logp = agent.select_action(state)
        state, reward, done, die = env.step(action)
        score += reward
        if show_progress:
            progress.update()
            progress.set_description("Score: {:.2f}".format(score))
        if done or die:
            done_or_die = True
    env.close()
    if show_progress:
        progress.close()
    if record_video:
        show_video()
    return score


class Agent():
    """
    Agent for training
    """

    def __init__(self, net, action_mapping, device, img_stack=1):
        self.net = net
        self.action_mapping = action_mapping
        self.device = device
        self.img_stack = img_stack

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.net(state)
            if type(action_probs) in (tuple, list):
                action_probs = action_probs[0]
        action, action_idx, a_logp = self.sample_action(action_probs)
        a_logp = a_logp.item()

        return action, action_idx, a_logp

    def sample_action(self, probs):
        m = Categorical(logits=probs.to("cpu"))
        action_idx = m.sample()
        a_logp = m.log_prob(action_idx)
        action = self.action_mapping[int(action_idx.squeeze().cpu().numpy())]
        return action, action_idx, a_logp

    def load_param(self, param_file):
        self.net.load_state_dict(torch.load(param_file))


def rgb2gray(rgb, norm=True):
    # rgb image -> gray [0, 1]
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    if norm:
        # normalize
        gray = gray / 128. - 1.
    return gray


def hide_hud(img):
    img[84:] = 0
    return img


class DemonstrationDataset(Dataset):
    def __init__(self, data_file, img_stack=1, show_hud=True):
        assert (os.path.exists(data_file))
        self.data = np.load(data_file)
        self.frames = self.data["frames"]
        self.actions = self.data["actions"]
        self.img_stack = img_stack

        if show_hud:
            self.transforms = Compose([rgb2gray])
        else:
            self.transforms = Compose([hide_hud, rgb2gray])

        # Action space (map from continuous actions for steering, throttle and break to 25 action combinations)
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

        self.action_mapping = {i: x for i, x in enumerate(action_mapping)}
        self.act_to_idx = {x: i for i, x in enumerate(action_mapping)}

    def __len__(self):
        return self.frames.shape[0] - self.img_stack

    def __getitem__(self, idx):
        frames, action = self.frames[idx:idx + self.img_stack], self.actions[
            idx + self.img_stack - 1]
        transformed = []
        for i in range(len(frames)):
            transformed.append(self.transforms(frames[i]))
        transformed = np.stack(transformed, axis=0)
        return transformed, self.act_to_idx[tuple(action)]

    def append(self, frame, action):
        self.frames = np.append(self.frames, frame, axis=0)
        self.actions = np.append(self.actions, action, axis=0)


def wrap_env(env):
    env = Monitor(env, './video', force=True)
    return env


class Env():
    """
    Environment wrapper for CarRacing
    """

    def __init__(self, img_stack=1, show_hud=True, record_video=True):
        from gym import wrappers
        self.record_video = record_video
        self.gym_env = gym.make('CarRacing-v0')
        self.gym_env = wrappers.Monitor(self.gym_env, "./monitor_car", video_callable=False, force=True)
        self.env = self.wrap_env(self.gym_env)
        self.action_space = self.env.action_space
        self.img_stack = img_stack
        self.show_hud = show_hud

    def reset(self, raw_state=False):
        self.env = self.wrap_env(self.gym_env)
        # self.gym_env = wrappers.Monitor(self.gym_env, "./monitor_car", video_callable=False, force=True)
        self.rewards = []
        img_rgb = self.env.reset()
        img_gray = rgb2gray(img_rgb)
        if not self.show_hud:
            img_gray = hide_hud(img_gray)
        self.stack = [img_gray] * self.img_stack
        if raw_state:
            return np.array(self.stack), np.array(img_rgb)
        else:
            return np.array(self.stack)

    def step(self, action, raw_state=False):
        img_rgb, reward, done, _ = self.env.step(action)
        # accumulate reward
        self.rewards.append(reward)
        # if no reward recently, end the episode
        die = True if np.mean(self.rewards[-np.minimum(100, len(self.rewards)):]) <= -1 else False
        if done or die:
            self.env.close()
        img_gray = rgb2gray(img_rgb)
        if not self.show_hud:
            img_gray = hide_hud(img_gray)
        # add to frame stack
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.img_stack
        # --
        if raw_state:
            return np.array(self.stack), np.sum(self.rewards[-1]), done, die, img_rgb
        else:
            return np.array(self.stack), np.sum(self.rewards[-1]), done, die

    def render(self, *arg):
        return self.env.render(*arg)

    def close(self):
        self.env.close()

    def wrap_env(self, env):
        if self.record_video:
            env = wrap_env(env)
        return env


def disable_view_window():
    from gym.envs.classic_control import rendering
    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, **kwargs)
        self.window.set_visible(visible=False)

    rendering.Viewer.__init__ = constructor
