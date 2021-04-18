import base64
import glob
import io
import os
from time import time, strftime

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import display
from IPython.display import HTML
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
        self.record_video = record_video
        self.gym_env = gym.make('CarRacing-v0')
        self.env = self.wrap_env(self.gym_env)
        self.action_space = self.env.action_space
        self.img_stack = img_stack
        self.show_hud = show_hud

    def reset(self, raw_state=False):
        self.env = self.wrap_env(self.gym_env)
        self.rewards = []
        disable_view_window()
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
        # if done or die:
        #     self.env.close()
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
        pass
        # return self.env.render(*arg)

    def close(self):
        self.env.close()

    def wrap_env(self, env):
        if self.record_video:
            env = wrap_env(env)
        return env


def plot_metrics(logger):
    train_loss = logger.get_values("training_loss")
    train_entropy = logger.get_values("training_entropy")
    val_loss = logger.get_values("validation_loss")
    val_acc = logger.get_values("validation_accuracy")

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, label="train")
    ax2 = fig.add_subplot(131, label="val", frame_on=False)
    ax4 = fig.add_subplot(132, label="entropy")
    ax3 = fig.add_subplot(133, label="acc")

    ax1.plot(train_loss, color="C0")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Update (Training)", color="C0")
    ax1.xaxis.grid(False)
    ax1.set_ylim((0, 4))

    ax2.plot(val_loss, color="C1")
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.set_xlabel('Epoch (Validation)', color="C1")
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.grid(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_ylim((0, 4))

    ax4.plot(train_entropy, color="C3")
    ax4.set_xlabel('Update (Training)', color="black")
    ax4.set_ylabel("Entropy", color="C3")
    ax4.tick_params(axis='x', colors="black")
    ax4.tick_params(axis='y', colors="black")
    ax4.xaxis.grid(False)

    ax3.plot(val_acc, color="C2")
    ax3.set_xlabel("Epoch (Validation)", color="black")
    ax3.set_ylabel("Accuracy", color="C2")
    ax3.tick_params(axis='x', colors="black")
    ax3.tick_params(axis='y', colors="black")
    ax3.xaxis.grid(False)
    ax3.set_ylim((0, 1))

    fig.tight_layout(pad=2.0)
    plt.show()


class Logger():
    def __init__(self, logdir, params=None):
        self.basepath = os.path.join(logdir, strftime("%Y-%m-%dT%H-%M-%S"))
        os.makedirs(self.basepath, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        if params is not None and os.path.exists(params):
            shutil.copyfile(params, os.path.join(self.basepath, "params.pkl"))
        self.log_dict = {}
        self.dump_idx = {}

    @property
    def param_file(self):
        return os.path.join(self.basepath, "params.pkl")

    @property
    def onnx_file(self):
        return os.path.join(self.basepath, "model.onnx")

    @property
    def log_dir(self):
        return os.path.join(self.basepath, "logs")

    def log(self, name, value):
        if name not in self.log_dict:
            self.log_dict[name] = []
            self.dump_idx[name] = -1
        self.log_dict[name].append((len(self.log_dict[name]), time(), value))

    def get_values(self, name):
        if name in self.log_dict:
            return [x[2] for x in self.log_dict[name]]
        return None

    def dump(self):
        for name, rows in self.log_dict.items():
            with open(os.path.join(self.log_dir, name + ".log"), "a") as f:
                for i, row in enumerate(rows):
                    if i > self.dump_idx[name]:
                        f.write(",".join([str(x) for x in row]) + "\n")
                        self.dump_idx[name] = i


def print_action(dataset, action):
    action = dataset.action_mapping[action]
    print("Left %.1f" % action[0] if action[0] < 0 else "Right %.1f" %
                                                        action[0] if action[0] > 0 else "Straight")
    print("Throttle %.1f" % action[1])
    print("Break %.1f" % action[2])


def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display.display(HTML(data='''<video alt="test" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")


def train(net, loader, loss_func, optimizer, logger, epoch, device):
    net.train()
    running_loss = None
    alpha = 0.3
    with tqdm(loader, desc="[%03d] Loss: %.4f" % (epoch, 0.)) as pbar:
        for frame, action in pbar:
            frame = frame.float().to(device)
            action = action.to(device)
            # prediction
            prediction = net(frame)
            # loss
            loss = loss_func(prediction, action)
            # entropy
            with torch.no_grad():
                probs = torch.softmax(prediction, dim=-1)
                entropy = torch.mean(-torch.sum(probs * torch.log(probs), dim=-1))
                # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            logger.log("training_loss", loss.item())
            logger.log("training_entropy", entropy.item())
            # update progress
            running_loss = loss.item() if running_loss is None else loss.item() * alpha + (1 - alpha) * running_loss
            pbar.set_description("[%03d] Loss: %.4f" % (epoch, running_loss))
    return frame  # serves as sample input for saving the model in ONNX format


def val(net, loader, loss_func, logger, epoch, device):
    bs = loader.batch_size
    net.eval()
    predictions = np.empty((len(loader.dataset, )), dtype=np.float32)
    targets = np.empty((len(loader.dataset, )), dtype=np.float32)
    loss_ = []
    for i, (frame, action) in enumerate(tqdm(loader, desc="[%03d] Validation" % epoch)):
        with torch.no_grad():
            frame = frame.float().to(device)
            action = action.to(device)
            # prediction
            prediction = net(frame)
            loss_.append(loss_func(prediction, action).cpu().item())
            # collect predictions and targets
            prediction = torch.argmax(prediction.cpu(), dim=-1)
            predictions[i * bs:i * bs + len(prediction)] = prediction.cpu().numpy()
            targets[i * bs:i * bs + len(prediction)] = action.cpu().numpy()
    # loss
    accuracy = np.mean(targets == predictions)
    # log
    logger.log("validation_loss", np.mean(loss_))
    logger.log("validation_accuracy", accuracy)
    # --
    return np.mean(loss_), accuracy


def save_as_onnx(torch_model, sample_input, model_path):
    torch.onnx.export(torch_model,  # model being run
                      sample_input,  # model input (or a tuple for multiple inputs)
                      f=model_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=13,
                      # the ONNX version to export the model to - see https://github.com/microsoft/onnxruntime/blob/master/docs/Versioning.md
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      )
