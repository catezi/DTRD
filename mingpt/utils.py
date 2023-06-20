import torch
import random
import datetime
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


global_game_info = {
    'Breakout': {
        'action_dim': 4,
        'reward_range': {
            'min_reward': 0.0,
            'max_reward': 1.0,
        },
        'reward_vector': [0., 1.],
        'reward_category_num': 2,
        'target_reward': 90.0,
    },
    'Seaquest': {
        'action_dim': 18,
        'reward_range': {
            'min_reward': 0.0,
            'max_reward': 1.0,
        },
        'reward_vector': [0., 1.],
        'reward_category_num': 2,
        'target_reward': 290.0,
    },
    'Qbert': {
        'action_dim': 6,
        'reward_range': {
            'min_reward': 0.0,
            'max_reward': 1.0,
        },
        'reward_vector': [0., 1.],
        'reward_category_num': 2,
        'target_reward': 662.0,
    },
    'Pong': {
        'action_dim': 6,
        'reward_range': {
            'min_reward': -1.0,
            'max_reward': 1.0,
        },
        'reward_vector': [-1., 0., 1.],
        'reward_category_num': 3,
        'target_reward': 20.0,
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_config(args):
    print('initial learning rate', args.learning_rate if hasattr(args, 'learning_rate') else None)
    print('initial redistribute learning rate', args.redistribute_learning_rate if hasattr(args, 'redistribute_learning_rate') else None)
    print('dropout rate', args.drop_out if hasattr(args, 'drop_out') else None)
    print('trajectory_lambda', args.trajectory_lamb if hasattr(args, 'trajectory_lamb') else None)
    print('redistribute_step_size', args.redistribute_step_size if hasattr(args, 'redistribute_step_size') else None)
    print('redistribute_gamma', args.redistribute_gamma if hasattr(args, 'redistribute_gamma') else None)
    print('n_layer', args.n_layer if hasattr(args, 'n_layer') else None)
    print('n_head', args.n_head if hasattr(args, 'n_head') else None)
    print('n_embd', args.n_embd if hasattr(args, 'n_embd') else None)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None, actions=None, rtgs=None, timesteps=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    x: now state
    """
    block_size = model.get_block_size()
    # 测试的时候，需要关闭Normalization 和 Dropout
    model.eval()
    for k in range(steps):
        # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        # 序列长度不能超过block_size
        x_cond = x if x.size(1) <= block_size // 3 else x[:, -block_size // 3:]
        if actions is not None:
            actions = actions if actions.size(1) <= block_size // 3 else actions[:, -block_size // 3:]
        rtgs = rtgs if rtgs.size(1) <= block_size // 3 else rtgs[:, -block_size // 3:]
        logits, _ = model(x_cond, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        # x = torch.cat((x, ix), dim=1)
        x = ix

    return x


class TimeManager:
    def __init__(self, debug):
        self.now_time = datetime.datetime.now()
        self.debug = debug

    def init_now_time(self):
        self.now_time = datetime.datetime.now()

    def print_time_interval(self, info):
        now_time = datetime.datetime.now()
        if self.debug:
            print(info, (now_time - self.now_time).microseconds)
        self.now_time = now_time
