"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import cv2
import math
import random
import logging
import atari_py
import numpy as np
import torch.distributed as dist

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm
from PIL import Image
from collections import deque
from reward_redistribute import GradientApproximate

logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class ModelTrainer:

    def __init__(self, model, redistribute, device, train_dataset, val_dataset,
                 train_trajectory_dataset, val_trajectory_dataset, config):
        self.config = config
        # set policy model and redistribute mode
        self.model = model
        self.new_model = model.copy_model().to(device)
        self.redistribute = redistribute
        self.device = device
        # set data set
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_trajectory_dataset = train_trajectory_dataset
        self.val_trajectory_dataset = val_trajectory_dataset
        # set optimizer
        self.now_learning_rate = config.learning_rate
        self.tokens = 0  # counter used for learning rate decay
        # policy network optim
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = model.configure_optimizers(config)
        # reward redistribute network optim and scheduler
        self.redistribute_optimizer = torch.optim.Adam(self.redistribute.parameters(), lr=config.redistribute_learning_rate)
        self.redistribute_scheduler = torch.optim.lr_scheduler.StepLR(
            self.redistribute_optimizer, step_size=config.redistribute_step_size, gamma=config.redistribute_gamma)
        # calculate gradient approximate
        self.gradient_appro = GradientApproximate(
            device=self.device, new_model=self.new_model, eps=0.01, weight_decay=config.weight_decay,
        )

    def save_all_checkpoint(self, epoch):
        # save policy network parameters
        if not os.path.isdir(self.config.save_model_path):
            os.makedirs(self.config.save_model_path)
        save_file_name = self.config.save_model_path + '/epoch' + str(epoch) + '_' + self.config.save_model_name
        # save all model and all optim together
        torch.save({
            'epoch': epoch,
            'policy_network': self.model.state_dict(),
            'policy_optimizer': self.optimizer.state_dict(),
            'redistribute_network': self.redistribute.state_dict(),
            'redistribute_optimizer': self.redistribute_optimizer.state_dict(),
            'max_timestep': self.config.max_timestep,
            'discrete_redistribute': self.config.discrete_redistribute,
        }, save_file_name)
        logger.info("saving %s", save_file_name)

    def train(self):
        model, redistribute,  config = self.model, self.redistribute, self.config
        train_trajectory_dataset, val_trajectory_dataset = self.train_trajectory_dataset, self.val_trajectory_dataset

        ########################################## start for run epoch
        def run_epoch(epoch):
            # training model
            model.train()
            train_loader = DataLoader(self.train_dataset, shuffle=False, pin_memory=True,
                                      batch_size=config.batch_size, num_workers=config.num_workers)
            val_loader = DataLoader(self.val_dataset, shuffle=False, pin_memory=True,
                                    batch_size=config.batch_size, num_workers=config.num_workers)
            train_losses = []
            val_losses = []
            train_acc_rates = []
            val_acc_rates = []
            # if quick update tqdm for test, frequency us 0.1
            # if is main gpu use tqdm else use simple iter
            pbar = tqdm(enumerate(zip(train_loader, val_loader)), total=min(len(train_loader), len(val_loader)))
            # bio-level train and get loss
            for it, ((x_t, y_t, r_t, t_t, traj_t), (x_v, y_v, r_v, t_v, traj_v)) in pbar:

                # place train data on the correct device
                x_t = x_t.to(self.device)
                y_t = y_t.to(self.device)
                r_t = r_t.to(self.device)
                t_t = t_t.to(self.device)

                # place val data on the correct device
                x_v = x_v.to(self.device)
                y_v = y_v.to(self.device)
                r_v = r_v.to(self.device)
                t_v = t_v.to(self.device)

                # redistribute_reward: (batch_size, context_len, 1)
                redistribute_reward_t, _ = train_trajectory_dataset.get_redistribute_rtgs_local(
                    states=x_t, actions=y_t, timesteps=t_t, indexes=traj_t, redistribute_network=redistribute,
                    device=self.device, calculate_sum_square=False,
                )

                # apply redistribute train rtgs to sparse rtgs
                r_t_m = r_t - redistribute_reward_t.clone().detach()
                r_t = r_t - redistribute_reward_t

                redistribute_reward_v, redistribute_sum_square = val_trajectory_dataset.get_redistribute_rtgs_local(
                    states=x_v, actions=y_v, timesteps=t_v, indexes=traj_v, redistribute_network=redistribute,
                    device=self.device, calculate_sum_square=True,
                )

                # apply redistribute val rtgs to  sparse rtgs
                r_v_m = r_v - redistribute_reward_v.clone().detach()
                r_v = r_v - redistribute_reward_v

                # Upper-level optimization
                eta = self.now_learning_rate
                val_loss, val_logits = self.gradient_appro.redistribute_step(
                    model, redistribute, redistribute_sum_square, self.optimizer, x_t, y_t, r_t, t_t, x_v, y_v, r_v, t_v, eta)

                # calculate train acc_rate
                max_prob_action = val_logits.argmax(dim=-1)
                val_acc_rates.append(torch.sum(max_prob_action.unsqueeze(-1) - y_v == 0).item() /
                                     (max_prob_action.unsqueeze(-1) - y_v).numel())
                val_losses.append(val_loss.item())
                torch.nn.utils.clip_grad_norm_(redistribute.parameters(), config.grad_norm_clip)
                self.redistribute_optimizer.step()
                self.redistribute_scheduler.step(None)

                # Lower-level optimization
                # forward the model
                with torch.set_grad_enabled(True):
                    # logits, loss = model(x, y, r)
                    logits, loss = model(x_t, y_t, y_t, r_t_m, t_t)
                    # calculate train acc_rate
                    max_prob_action = logits.argmax(dim=-1)
                    train_acc_rates.append(torch.sum(max_prob_action.unsqueeze(-1) - y_t == 0).item() /
                                           (max_prob_action.unsqueeze(-1) - y_t).numel())
                    loss = loss.mean()
                    train_losses.append(loss.item())
                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()
                # decay the learning rate based on our progress
                if config.lr_decay:
                    self.tokens += (y_t >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                    # if retrain model, no need to warm up
                    if self.tokens < config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(self.tokens - config.warmup_tokens) / float(
                            max(1, config.final_tokens - config.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    self.now_learning_rate = lr
                else:
                    lr = config.learning_rate
                    self.now_learning_rate = lr

                # forward the model
                with torch.set_grad_enabled(True):
                    # logits, loss = model(x, y, r)
                    logits, loss = model(x_v, y_v, y_v, r_v_m, t_v)
                    # calculate acc_rate
                    max_prob_action = logits.argmax(dim=-1)
                    train_acc_rates.append(torch.sum(max_prob_action.unsqueeze(-1) - y_v == 0).item() / (max_prob_action.unsqueeze(-1) - y_v).numel())
                    loss = loss.mean()
                    train_losses.append(loss.item())
                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()

                # decay the learning rate based on our progress
                if config.lr_decay:
                    self.tokens += (y_v >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                    # if retrain model, no need to warm up
                    if self.tokens < config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(self.tokens - config.warmup_tokens) / float(
                            max(1, config.final_tokens - config.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    self.now_learning_rate = lr
                else:
                    lr = config.learning_rate
                    self.now_learning_rate = lr
                # if not quick update tqdm for test, update every step else every 100 step
                if it % 100 == 0:
                    pbar.set_description(
                        f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
            # calculate mean train loss and val loss and acc_rate
            print('mean train loss: ', sum(train_losses) / len(train_losses))
            print('mean train acc_rate: ', sum(train_acc_rates) / len(train_acc_rates))
            print('mean val loss: ', sum(val_losses) / len(val_losses))
            print('mean val acc_rate: ', sum(val_acc_rates) / len(val_acc_rates))
        ########################################## end for run epoch

        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs):
            run_epoch(epoch)
            if config.save:
                self.save_all_checkpoint(epoch)


