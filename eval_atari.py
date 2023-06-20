import cv2
import math
import torch
import queue
import random
import pickle
import logging
import argparse
import atari_py
import numpy as np

from tqdm import tqdm
from collections import deque
from mingpt.model_atari import GPTConfig, GPT
from mingpt.utils import sample, set_seed, global_game_info
from reward_redistribute import RedistributeConfig, DiscreteRedistributeNetwork, ContinuousRedistributeNetwork

######################################## to be checked
################# basic task config
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--game_episodes_num', type=int, default=5, help='Number of episodes to play game')
parser.add_argument('--game', type=str, default='Breakout')

################# hyper parameters
parser.add_argument('--drop_out', type=float, default=0.1)
parser.add_argument('--target_reward', type=float, default=90.0, help='Target reward for game')

################# model structure
parser.add_argument('--n_layer', type=int, default=6)
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--n_embd', type=int, default=128)
parser.add_argument('--discrete_redistribute', type=int, default=0, help='If use classification redistribute reward')
parser.add_argument('--redistribute_activate_func', type=str, default='sigmoid', help='activate function for redistribute model')

#################  option in real game test
parser.add_argument('--use_sample_policy', type=int, default=0, help='If use sample policy to choose action instead of argmax')
parser.add_argument('--min_length', type=int, default=108e3, help='If quick update tqdm info for test')
parser.add_argument('--model_path', type=str, default=None, help='path of pretrained policy&redistribute model parameters to load')

logger = logging.getLogger(__name__)


class Args:
    def __init__(self, game, seed):
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.seed = seed
        self.max_episode_length = args.min_length
        self.game = game
        self.history_length = 4


class Env:
    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Qbert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()


def play_real_game(target_rewards):
    model.train(False)
    redistribute.train(False)
    # config game env
    game_args = Args(args.game.lower(), args.seed)
    env = Env(game_args)
    env.eval()
    total_rewards = []
    done = True
    all_epochs = tqdm(enumerate(range(args.game_episodes_num)), total=len(range(args.game_episodes_num)), mininterval=0.1)
    for _, _ in all_epochs:
        state = env.reset()
        # state = state.type(torch.float32).to(device).unsqueeze(0).unsqueeze(0)
        state = state.to(dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        rtgs = [target_rewards]
        # first state is from env, first rtg is target return, and first timestep is 0
        sampled_action = sample(model, state, 1, temperature=1.0, sample=args.use_sample_policy, actions=None,
                                rtgs=torch.tensor(rtgs, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(-1),
                                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device))
        j = 0
        all_states = state
        reward_sum = 0
        accumulate_reward = 0
        record_step_reward = queue.Queue()
        actions = []
        while True:
            # only for first step of epoch
            if done:
                state, reward_sum, done = env.reset(), 0, False
            action = sampled_action.cpu().numpy()[0, -1]
            actions += [sampled_action]
            state, reward, done = env.step(action)
            reward_sum += reward
            j += 1
            if done:
                total_rewards.append(float(reward_sum))
                print('now_rewards', reward_sum, 'timestep', j)
                break
            state = state.unsqueeze(0).unsqueeze(0).to(device)
            all_states = torch.cat([all_states, state], dim=0)
            # play in sparse reward env, mask middle reward from env
            """
            now_state: (1, 1, 4, 84, 84), now_action: (1, 1, 1), redistribute_reward: (1, 1, 1)
            """
            now_state = all_states[-2].unsqueeze(0)
            # if args.state_diff and all_states.shape[0] >= 3:
            #     last_state = all_states[-3].unsqueeze(0)
            #     now_state = now_state - last_state
            now_action = torch.tensor(action, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            redistribute_reward = redistribute.get_redistribute(now_state, now_action).item()
            rtgs += [rtgs[-1] - redistribute_reward]
            # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
            # timestep is just current timestep
            """
            all_states: (cur_len, 1, 4, 84, 84)
            states_tensor: (1, cur_len, 1, 4*84*84) 
            actions_tensor: (1, cur_len - 1, 1)
            rtgs_tensor: (1, cur_len, 1)
            """
            states_tensor = all_states.unsqueeze(0)
            actions_tensor = torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0)
            rtgs_tensor = torch.tensor(rtgs, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(-1)
            sample_model = model.module if hasattr(model, "module") else model
            sampled_action = sample(sample_model, states_tensor, 1, temperature=1.0, sample=args.use_sample_policy,
                                    actions=actions_tensor, rtgs=rtgs_tensor,
                                    timesteps=(min(j, max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(device)))
    env.close()
    eval_reward = sum(total_rewards) / args.game_episodes_num
    model.train(True)
    # calculate variance
    vari_reward = 0.
    for reward in total_rewards:
        vari_reward += (reward - eval_reward) ** 2
    vari_reward /= len(total_rewards)
    print('target reward', target_rewards)
    print('eval_reward', np.mean(total_rewards))
    print('var_reward', np.var(total_rewards))
    print('std_reward', np.std(total_rewards))


if __name__ == "__main__":
    args = parser.parse_args()
    # print basic info
    print('test_model_path', args.model_path)
    print('game_name', args.game)
    game_info = global_game_info[args.game]
    # set random seed for test game
    set_seed(args.seed)
    # get available device
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    # load trained model parameters
    trained_model = torch.load(args.model_path, map_location=torch.device('cpu')) if device == 'cpu' else torch.load(args.model_path)
    # load max_timestep form model para dict
    assert 'max_timestep' in trained_model
    max_timestep = trained_model['max_timestep']
    # config policy model and load parameters
    mconf = GPTConfig(
        game_info['action_dim'], args.context_length,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        max_timestep=max_timestep, drop_out=args.drop_out,
    )
    model = GPT(mconf).to(device)
    model.load_state_dict(trained_model['policy_network'])
    # load redistribute network and load parameters
    rconf = RedistributeConfig(
        action_dim=game_info['action_dim'], n_embd=args.n_embd,
        device=device, context_length=args.context_length, redistribute_activate_func='tanh',
        reward_category_num=game_info['reward_category_num'], reward_vector=game_info['reward_vector'],
        reward_range=game_info['reward_range'],
    )
    redistribute = DiscreteRedistributeNetwork(rconf).to(device) if args.discrete_redistribute else ContinuousRedistributeNetwork(rconf).to(device)
    redistribute.load_state_dict(trained_model['redistribute_network'])
    # start play game
    play_real_game(target_rewards=game_info['target_reward'])
