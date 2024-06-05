import os
import sys
import torch
import random
import logging
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.spaces import MultiDiscrete, Discrete, Box

REF_MAX_SCORE = {
    'HalfCheetah': 12135.0,
    'Walker2d': 4592.3,
    'Hopper': 3234.3,
}

REF_MIN_SCORE = {
    'HalfCheetah': -280.178953,
    'Walker2d': 1.629008,
    'Hopper': -20.272305,
}


def config_logging(log_file="main.log"):
    date_format = '%Y-%m-%d %H:%M:%S'
    log_format = '%(asctime)s: [%(levelname)s]: %(message)s'
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # Set up the FileHandler for logging to a file
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler])


def get_log_dict(manager=None, task_num=3,num_seeds=0):
    log_keys = ['rounds', 'eval_returns', 'd4rl_score', 'server_loss', 'client_loss','rtg_target']

    if manager is None:
        return {key: [[] for _ in range(task_num)]  for key in log_keys}
    else:
        return manager.dict({key: manager.list([[]] * num_seeds) for key in log_keys})


def set_seed_everywhere(env: gym.Env, seed=0):
    env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_space_shape(space, is_vector_env=False):
    if isinstance(space, Discrete):
        return space.n
    elif isinstance(space, MultiDiscrete):
        return space.nvec[0]
    elif isinstance(space, Box):
        space_shape = space.shape[1:] if is_vector_env else space.shape
        if len(space_shape) == 1:
            return space_shape[0]
        else:
            return space_shape  # image observation
    else:
        raise ValueError(f"Space not supported: {space}")


def get_d4rl_normalized_score(env_name, score):
    assert env_name in REF_MAX_SCORE, f'no reference score for {env_name} to calculate d4rl score'
    return (score - REF_MIN_SCORE[env_name]) / (REF_MAX_SCORE[env_name] - REF_MIN_SCORE[env_name]) * 100


def write_to_dict(log_dict, key, value, using_mp,env_index=0):
    if using_mp:
        log_dict[key].append(value)
    else:
        log_dict[key][env_index][-1].append(value)


def sync_and_visualize(log_dict, local_log_dict, barrier, idx, step,env_index, title, using_mp):
    if using_mp:
        for key in local_log_dict.keys():
            log_dict[key][idx] = local_log_dict[key]
        barrier.wait()
        if idx == 0:
            visualize(step, title, log_dict,env_index)
    else:
        visualize(step, title, log_dict,env_index)


def moving_average(a, n):
    if len(a) <= n:
        return a
    ret = np.cumsum(a, dtype=float, axis=-1)
    ret[n:] = ret[n:] - ret[:-n]
    return (ret[n - 1:] / n).tolist()


def pad_and_get_mask(lists):
    """
    Pad a list of lists with zeros and return a mask of the same shape.
    """
    lens = [len(l) for l in lists]
    max_len = max(lens)
    arr = np.zeros((len(lists), max_len), float)
    mask = np.arange(max_len) < np.array(lens)[:, None]
    arr[mask] = np.concatenate(lists)
    return np.ma.array(arr, mask=~mask)


def plot_scores(scores, steps=None, window=100, label=None, color=None):
    avg_scores = [moving_average(score, window) for score in scores]
    if steps is not None:
        for i in range(len(scores)):
            avg_scores[i] = np.interp(np.arange(steps[i][-1]), [0] + steps[i][-len(avg_scores[i]):], [0.0] + avg_scores[i])
    if len(scores) > 1:
        avg_scores = pad_and_get_mask(avg_scores)
        scores = avg_scores.mean(axis=0)
        scores_l = avg_scores.mean(axis=0) - avg_scores.std(axis=0)
        scores_h = avg_scores.mean(axis=0) + avg_scores.std(axis=0)
        idx = list(range(len(scores)))
        plt.fill_between(idx, scores_l, scores_h, where=scores_h > scores_l, interpolate=True, alpha=0.25, color=color)
    else:
        scores = avg_scores[0]
    plot, = plt.plot(scores, label=label, color=color)
    return plot


def visualize(step, title, log_dict,env_index):
    eval_window, loss_window = 10, 200
    plt.figure(figsize=(6, 6))

    # plot train and eval returns
    lines = []
    plt.title('round %s. score: %s' % (step, np.max(log_dict['d4rl_score'][env_index][-eval_window:])))
    plt.xlabel('round')

    lines.append(plot_scores(log_dict['eval_returns'][env_index], log_dict['rounds'][env_index], window=1, label='eval return', color='C1'))
    plt.ylabel('scores')
    plt.twinx()
    lines.append(plot_scores(log_dict['d4rl_score'][env_index], log_dict['rounds'][env_index], window=1, label='d4rl score'))
    plt.ylim(0, 110)
    plt.ylabel('d4rl score')
    plt.legend(lines, [line.get_label() for line in lines])
    plt.xlabel('round')

    # plot td losses

    plt.savefig(f'{title}.png')
    plt.close()

def visualize_loss(step, title, log_dict,):
    loss_window = 200
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.title(' metrics')
    plot_scores(log_dict['client_loss'][0], window=loss_window, label='client_loss', color='C0')

    plt.xlabel('round')
    plt.ylabel('client loss')
    plt.legend()
    plt.suptitle("client loss", fontsize=16)
    plt.subplot(1, 2, 2)
    plt.title(' metrics')
    plot_scores(log_dict['server_loss'][0], window=loss_window, label='server_loss', color='C3')
    plt.xlabel('round')
    plt.ylabel('server loss')
    plt.legend()

    plt.suptitle(title, fontsize=16)
    plt.savefig(f'loss.png')
    plt.close()

def freeze_model(model, to_freeze):

    for (name, param) in model.named_parameters():
        if to_freeze in name:
            param.requires_grad = False
        else:
            pass

def unfreeze_model(model, to_freeze):

    for (name, param) in model.named_parameters():
        if to_freeze in name:
            param.requires_grad = True
        else:
            pass


def generate_sum_parameters(sum_parameters,local_parameters,global_parameters):
    if sum_parameters is None:
        sum_parameters = {}
        for key, var in local_parameters.items():
            if "block" not in key:
                sum_parameters[key] = var.clone()
            else:
                global_parameters[key] = var.clone()
    else:
        for key in sum_parameters:
            if "block" not in key:
                sum_parameters[key] = sum_parameters[key] + local_parameters[key]
            else:
                global_parameters[key] = local_parameters[key]
    return sum_parameters

