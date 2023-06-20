import os
import torch
import logging
import argparse
import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset

from mingpt.model_atari import GPTConfig, GPT
from mingpt.trainer_atari import TrainerConfig, ModelTrainer
from mingpt.utils import set_seed, global_game_info, print_config
from reward_redistribute import RedistributeConfig, DiscreteRedistributeNetwork, ContinuousRedistributeNetwork

################# basic task config
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--game', type=str, default='Breakout')

################# hyper parameters
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--drop_out', type=float, default=0.1)
parser.add_argument('--learning_rate', type=float, default=6e-4)
parser.add_argument('--redistribute_learning_rate', type=float, default=6e-4)
parser.add_argument('--redistribute_step_size', type=int, default=1000)
parser.add_argument('--redistribute_gamma', type=float, default=0.75)
parser.add_argument('--trajectory_lamb', type=float, default=0.01, help='Weight of trajectory redistribute regular term')

################# model structure
parser.add_argument('--n_layer', type=int, default=6)
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--n_embd', type=int, default=128)
parser.add_argument('--discrete_redistribute', type=int, default=0, help='If use discrete redistribute reward')
parser.add_argument('--redistribute_activate_func', type=str, default='sigmoid', help='activate function for redistribute model')

################# load / save config
# add for saving mode
parser.add_argument('--save', type=int, default=0, help='If save model at the end of each epoch')
parser.add_argument('--save_model_path', type=str, default='./trained_model/')
parser.add_argument('--save_model_name', type=str, default='save_model.pth')
# add for loading data
parser.add_argument('--data_dir', type=str, default='./data/', help='dir to load train / val set')


class StateActionReturnDataset(Dataset):

    def __init__(self, context_length, obs, actions, done_idxs,
                 rtgs, sparse_rtgs, timesteps, trajectory_index, device):
        self.context_length = context_length
        self.states = obs
        self.actions = actions
        self.done_idxs = done_idxs
        self.sparse_rtgs = sparse_rtgs
        self.dense_rtgs = rtgs
        self.dense_rewards = np.zeros_like(rtgs)
        self.timesteps = timesteps
        self.trajectory_index = trajectory_index
        self.device = device

    def __len__(self):
        return len(self.states) - self.context_length

    def __getitem__(self, idx):
        """
            states: (batch_size, context_length, 4*84*84)
            actions: (batch_size, context_length, 1)
            rtgs: (batch_size, context_length, 1)
            timesteps: (batch_size, 1, 1)
            trajectory_index: (batch_size, context_length, 1)
        """
        context_length = self.context_length
        done_idx = idx + context_length
        for i in self.done_idxs:
            if i > idx:
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - context_length
        # (context_length, 4*84*84)
        states = torch.tensor(np.array(self.states[idx:done_idx]), dtype=torch.float32).reshape(context_length, -1)
        states = states / 255.
        # (context_length, 1)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)
        rtgs = torch.tensor(self.sparse_rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx + 1], dtype=torch.int64).unsqueeze(1)
        trajectory_index = torch.tensor(self.trajectory_index[idx:done_idx], dtype=torch.long).unsqueeze(1)

        return states, actions, rtgs, timesteps, trajectory_index


class TrajectoryDataset:
    def __init__(self, states, actions, rtgs, done_idxs):
        self.states = []
        self.actions = []
        # self.rtgs = []
        self.head_rtgs = []
        # record local hed rtgs if necessary
        # record trajectory length
        self.traj_len = []
        np.set_printoptions(threshold=np.inf)
        torch.set_printoptions(threshold=np.inf)

        start = 0
        # store states actions data by trajectory
        for done in done_idxs:
            self.states.append(list(states[start: done]))
            self.actions.append(list(actions[start: done]))
            # self.rtgs.append(list(rtgs[start: done]))
            self.head_rtgs.append(rtgs[start])
            self.traj_len.append(done - start)
            start = done

    def get_redistribute_rtgs_local(self, states, actions, timesteps, indexes,
                                    redistribute_network, device, calculate_sum_square=False):
        """
            states: (batch_size, context_length, 4*84*84)
            actions: (batch_size, context_length, 1)
            timesteps: (batch_size, 1, 1)
            indexes: (batch_size, context_length, 1)
            step_redis_reward torch.Size([batch_size, context_length]) -> torch.Size([batch_size, 1, context_length])
            head_rtgs torch.Size([batch_size, 1])
            traj_len torch.Size([batch_size, 1])
            head_timestep torch.Size([batch_size, 1])
            head_redis_rewards torch.Size([batch_size, 1]) -> torch.Size([batch_size, context_length, 1])
            trajectory_redis_rtgs torch.Size([batch_size, context_length]) -> torch.Size([batch_size, context_length, 1])

        """
        batch_size, context_length = states.shape[0], states.shape[1]
        step_redis_reward = redistribute_network.get_redistribute(states, actions).reshape(batch_size, 1, context_length)
        head_rtgs = self.get_head_rtgs_from_index(indexes).to(device)
        traj_len = self.get_traj_length_from_index(indexes).to(device)
        head_timestep = timesteps.reshape(timesteps.shape[0], 1).to(dtype=torch.float32)
        head_redis_rewards = (
            head_rtgs * head_timestep / traj_len if not args.discrete_redistribute else torch.round(head_rtgs * head_timestep / traj_len)
        ).repeat(1, context_length).unsqueeze(-1)
        trajectory_redis_rtgs = torch.matmul(
            step_redis_reward,
            torch.tensor(
                np.triu(torch.ones((context_length, context_length))) -
                np.eye(context_length), dtype=torch.float32
            ).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        ).reshape(batch_size, context_length, 1)

        trajectory_redis_rtgs = trajectory_redis_rtgs + head_redis_rewards
        if calculate_sum_square:
            redis_sum_square = (
                                          torch.sum(step_redis_reward, dim=-1) * traj_len / context_length - head_rtgs
                                  ).pow(2).sum() * args.trajectory_lamb / batch_size
        else:
            redis_sum_square = torch.zeros(1).to(device)

        return trajectory_redis_rtgs, redis_sum_square

    def get_head_rtgs_from_index(self, indexes):
        head_indexes = indexes[:, 0, ]

        return torch.from_numpy(np.array(self.head_rtgs))[head_indexes]

    def get_traj_length_from_index(self, indexes):
        head_indexes = indexes[:, 0, ]

        return torch.from_numpy(np.array(self.traj_len, dtype=np.float32))[head_indexes]


if __name__ == "__main__":
    args = parser.parse_args()
    game_info = global_game_info[args.game]
    # set random seed for training
    set_seed(args.seed)
    # output option
    print_config(args)
    # load train / val data from dir
    train_data = np.load(args.data_dir + '/train_set.npz')
    obs, actions, returns, done_idxs, rtgs, sparse_rtgs, timesteps, tra_idx = \
        train_data['obss'].astype(np.uint8), train_data['actions'].astype(np.int32), \
        train_data['returns'].astype(np.float64), train_data['done_idxs'].astype(np.int64), \
        train_data['rtgs'].astype(np.float32), train_data['sparse_rtgs'].astype(np.float32), \
        train_data['timesteps'].astype(np.int64), train_data['tra_idx'].astype(np.int64)

    # load val data from a whole block
    val_data = np.load(args.data_dir + '/val_set.npz')
    obs1, actions1, returns1, done_idxs1, rtgs1, sparse_rtgs1, timesteps1, tra_idx1 = \
        val_data['obss'].astype(np.uint8), val_data['actions'].astype(np.int32), \
        val_data['returns'].astype(np.float64), val_data['done_idxs'].astype(np.int64), \
        val_data['rtgs'].astype(np.float32), val_data['sparse_rtgs'].astype(np.float32), \
        val_data['timesteps'].astype(np.int64), val_data['tra_idx'].astype(np.int64)
    # config max timestep for time(pos) embedding
    max_timestep = max(np.max(timesteps), np.max(timesteps1)) + 1
    # set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # config device for model training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # config reward redistribute network and option
    rconf = RedistributeConfig(
        action_dim=game_info['action_dim'], n_embd=args.n_embd,
        device=device, context_length=args.context_length, redistribute_activate_func='tanh',
        reward_category_num=game_info['reward_category_num'], reward_vector=game_info['reward_vector'],
        reward_range=game_info['reward_range'],
    )
    redistribute = DiscreteRedistributeNetwork(rconf).to(device) if args.discrete_redistribute else ContinuousRedistributeNetwork(rconf).to(device)
    # config dataset for train and val stage
    train_dataset = StateActionReturnDataset(
        context_length=args.context_length,
        obs=obs, actions=actions, done_idxs=done_idxs,
        rtgs=rtgs, sparse_rtgs=sparse_rtgs,
        timesteps=timesteps, trajectory_index=tra_idx,
        device=device,
    )
    val_dataset = StateActionReturnDataset(
        context_length=args.context_length,
        obs=obs1, actions=actions1, done_idxs=done_idxs1,
        rtgs=rtgs1, sparse_rtgs=sparse_rtgs1,
        timesteps=timesteps1, trajectory_index=tra_idx1,
        device=device,
    )
    train_trajectory_dataset = TrajectoryDataset(obs, actions, sparse_rtgs, done_idxs)
    val_trajectory_dataset = TrajectoryDataset(obs1, actions1, sparse_rtgs1, done_idxs1)
    # config policy network and option
    mconf = GPTConfig(
        game_info['action_dim'], args.context_length,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        max_timestep=max_timestep, drop_out=args.drop_out,
    )
    model = GPT(mconf).to(device)
    # initialize a trainer instance
    tconf = TrainerConfig(
        max_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        lr_decay=True, warmup_tokens=512 * 20,
        final_tokens=2 * len(train_dataset) * args.context_length * 3,
        num_workers=4, seed=args.seed, game=args.game,
        max_timestep=max_timestep, device=device,
        save=args.save, context_length=args.context_length,
        save_model_path=args.save_model_path, save_model_name=args.save_model_name,
        redistribute_learning_rate=args.redistribute_learning_rate,
        redistribute_step_size=args.redistribute_step_size, redistribute_gamma=args.redistribute_gamma,
        discrete_redistribute=args.discrete_redistribute,
    )
    trainer = ModelTrainer(
        model=model, redistribute=redistribute, device=device, train_dataset=train_dataset, val_dataset=val_dataset,
        train_trajectory_dataset=train_trajectory_dataset, val_trajectory_dataset=val_trajectory_dataset, config=tconf)
    trainer.train()
