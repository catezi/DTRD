import math
import time
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class RedistributeConfig:
    # action_dim: number of discrete actions
    # n_embd: dim of inner embeddings
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class DiscreteRedistributeNetwork(nn.Module):
    # input state = (batch, block_size, 4*84*84)
    # output: r0 = r(state, a0), r1 = z(state, a1), ..., rn = z(state, an)
    def __init__(self, config):
        super(DiscreteRedistributeNetwork, self).__init__()
        self.config = config
        self.state_encoder = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
                                           nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                           nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                           nn.Flatten(), nn.Linear(3136, config.n_embd), nn.Tanh())

        self.action_embeddings = nn.Sequential(nn.Embedding(config.action_dim, config.n_embd), nn.Tanh())
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2 * config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, 16 * config.n_embd)
        self.fc3 = nn.Linear(16 * config.n_embd, config.reward_category_num)
        # mark vector for redistribute categories
        self.reward_vector = torch.tensor(np.array(config.reward_vector), dtype=torch.float32).to(config.device)
        # init all parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight, gain=np.sqrt(2))
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, states, actions):
        # state: (1, tra_len, 4*84*84)
        # actions: (1, tra_len, 1)

        # state_embeddings: (1 * block_size, n_embd)
        state_embeddings = self.state_encoder(states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous())
        # state_embeddings: (1, block_size, n_embd)
        state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd)
        # action_embeddings: (1, block_size, n_embd)
        action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1))
        # token_embeddings: (1, block_size, 2 * n_embd)
        token_embeddings = torch.cat([state_embeddings, action_embeddings], -1)

        x = self.relu(self.fc1(token_embeddings))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def get_redistribute(self, states, actions):
        # r: (1, tra_len, 3)
        # one_hot: (1, tra_len, 3)
        # redistribute: (1, tra_len)
        r = self.forward(states, actions)
        one_hot = nn.functional.gumbel_softmax(r, tau=1.0, hard=True, dim=-1)
        redistribute = (one_hot * self.reward_vector).sum(-1)

        return redistribute


class ContinuousRedistributeNetwork(nn.Module):
    # input state = (batch, block_size, 4*84*84)
    # output: r0 = r(state, a0), r1 = z(state, a1), ..., rn = z(state, an)
    def __init__(self, config):
        super(ContinuousRedistributeNetwork, self).__init__()
        self.config = config
        self.state_encoder = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
                                           nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                           nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                           nn.Flatten(), nn.Linear(3136, config.n_embd), nn.Tanh())
        self.action_embeddings = nn.Sequential(nn.Embedding(config.action_dim, config.n_embd), nn.Tanh())
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2 * config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, 16 * config.n_embd)
        self.fc3 = nn.Linear(16 * config.n_embd, 1)
        # range for redistribute reward
        self.reward_range = config.reward_range
        self.redistribute_activate_func = config.redistribute_activate_func
        # init all parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight, gain=np.sqrt(2))
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, states, actions):
        # state: (1, tra_len, 4*84*84)
        # actions: (1, tra_len, 1)

        # states.reshape(): (batch * block_size, 4, 84, 84)
        # state_embeddings: (batch * block_size, n_embd)
        state_embeddings = self.state_encoder(states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous())
        # state_embeddings: (batch, block_size, n_embd)
        state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd)
        # action_embeddings: (1, block_size, n_embd)
        action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1))
        # token_embeddings: (1, block_size, 2 * n_embd)
        token_embeddings = torch.cat([state_embeddings, action_embeddings], -1)

        x = self.relu(self.fc1(token_embeddings))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        if self.redistribute_activate_func == 'tanh':
            x = torch.tanh(x)
        else:
            x = torch.sigmoid(x)

        return x

    def get_redistribute(self, state, action):
        r = self.forward(state, action)
        r = r.squeeze(-1)
        # print('---------------------')
        # print(self.reward_range['max_reward'])
        # print(self.reward_range['min_reward'])
        # print('min r', torch.min(r))
        # print('max r', torch.max(r))
        if self.redistribute_activate_func == 'tanh':
            r = (self.reward_range['max_reward'] - self.reward_range['min_reward']) * (r + 1.) / 2. + self.reward_range['min_reward']
        else:
            r = (self.reward_range['max_reward'] - self.reward_range['min_reward']) * r + self.reward_range['min_reward']

        return r


class GradientApproximate:
    def __init__(self, device, new_model, eps=0.01, weight_decay=0.1):
        self.device = device
        self.new_model = new_model
        self.eps = eps
        self.weight_decay = weight_decay

    def redistribute_step(self, model, redistribute, redistribute_sum_square, policy_optimizer,
                          x_t, y_t, r_t, t_t, x_v, y_v, r_v, t_v, eta):
        # model(φ')
        self.compute_unrolled_model(model, policy_optimizer, x_t, y_t, r_t, t_t, eta)
        # L_val(θ, φ')
        val_logits, val_loss = self.new_model(x_v, y_v, y_v, r_v, t_v)
        self.new_model.zero_grad()
        redistribute.zero_grad()
        val_loss.backward(retain_graph=True)
        # dφ' = vector = d(L_val(θ, φ')) / d(φ')
        vector = [v.grad if v.grad is not None else torch.zeros_like(v) for v in self.new_model.parameters()]
        # d_Lval_theta = d(L_val(θ, φ')) / dθ
        d_Lval_theta = [v.grad.clone().detach() if v.grad is not None else torch.zeros_like(v) for v in redistribute.parameters()]

        # Second-order approximation
        # Calculating Vector Product through Taylor Expansion Approximation
        appro_vector_product = self.hessian_vector_product(model, redistribute, vector, x_t, y_t, r_t, t_t)

        # First-order approximation
        # Single step replacement+chain derivative rule, simplifying validation set expectations
        # redistribute_sum_square = lambda * sum(redistribute(si, ai| θ)) ^ 2
        # d(redistribute_sum_square) / dθ
        redistribute.zero_grad()
        redistribute_sum_square.backward()
        # d[L_val(θ, φ*)] / d(θ) ~=
        # d(L_val(θ, φ')) / d(θ) - eta * appro_vector_product +
        # d[lambda * sum(redistribute(si, ai| θ)) ^ 2] / dθ
        with torch.no_grad():
            for ap, dl, appro in zip(redistribute.parameters(), d_Lval_theta, appro_vector_product):
                ap.grad = ap.grad + (dl - eta * appro)
        return val_loss, val_logits

    # return copy model(φ')
    def compute_unrolled_model(self, model, policy_optimizer, x_t, y_t, r_t, t_t, eta):
        # L_train(θ, φ)
        _, train_loss = model(x_t, y_t, y_t, r_t, t_t)
        model.zero_grad()
        # dL_train(θ, φ) / dφ
        d_phi = torch.autograd.grad(train_loss, model.parameters(), allow_unused=True, retain_graph=True,
                                    grad_outputs=torch.ones_like(train_loss))
        with torch.no_grad():
            for (name, w), vw, g in zip(model.named_parameters(), self.new_model.parameters(), d_phi):
                if g is None:
                    g = torch.zeros_like(w)
                # get weight decay
                if name in model.decay:
                    weight_decay = self.weight_decay
                else:
                    weight_decay = 0.0
                # φ' = φ - η * (dL_train(θ, φ) / dφ + φ * φ_decay)
                vw.copy_(w - eta * (g + weight_decay * w))

    def hessian_vector_product(self, model, redistribute, vector, x_t, y_t, r_t, t_t):
        # eps = 1 / [0.01  * norm(d(L_val) / d(φ'))]
        R = self.eps / _concat(vector).norm()
        # φ+ = φ + eps * dφ'
        with torch.no_grad():
            for p, v in zip(model.parameters(), vector):
                # p.data.add(R, v)
                p += R * v
        # L_train(φ+)
        _, loss = model(x_t, y_t, y_t, r_t, t_t)
        # d[L_train(φ+)] / d(θ)
        grads_p = torch.autograd.grad(loss, redistribute.parameters(), allow_unused=True, retain_graph=True,
                                      grad_outputs=torch.ones_like(loss))

        # φ- =  φ - eps * dφ'
        with torch.no_grad():
            for p, v in zip(model.parameters(), vector):
                # p.data.sub(2 * R, v)
                p -= 2. * R * v
        # L_train(w-)
        _, loss = model(x_t, y_t, y_t, r_t, t_t)
        # d[L_train(w-)] / d(θ)
        grads_n = torch.autograd.grad(loss, redistribute.parameters(), allow_unused=True, retain_graph=True,
                                      grad_outputs=torch.ones_like(loss))

        with torch.no_grad():
            for p, v in zip(model.parameters(), vector):
                # p.data.add_(R, v)
                p += R * v

        # vector_product ~ {d[L_train(φ+)] / d(θ) - d[L_train(φ-)] / d(θ)} / (2 * eps)
        return [(x - y) / (2. * R) for x, y in zip(grads_p, grads_n)]

