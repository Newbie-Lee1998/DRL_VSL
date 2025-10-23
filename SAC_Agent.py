import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
import numpy as np

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.action_mean = nn.Linear(128, action_dim)
        self.action_log_std = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.action_mean(x)
        log_std = F.softplus(self.action_log_std(x))
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def scale_to_range(self, actions, min_val=-1.0, max_val=1.0):
        mean = actions.mean()
        std = actions.std()
        normalized_actions = (actions - mean) / std
        scaled_actions = torch.tanh(normalized_actions)
        scaled_actions = min_val + (scaled_actions + 1) * (max_val - min_val) / 2
        return scaled_actions

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        normal_action = dist.rsample()
        log_prob = dist.log_prob(normal_action)
        action = self.scale_to_range(normal_action)
        log_prob = log_prob - torch.log(1 - self.scale_to_range(action).pow(2) + 1e-7)
        return action, log_prob


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.minimal_size = 2000

    def add(self, state, action, reward, next_state, done):
        experience = np.hstack((state, action, [reward], next_state, [done]))
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer.pop(0)

    def sample(self):
        batch_experiences = random.sample(self.buffer, k=self.batch_size)
        batch_experiences = np.array(batch_experiences)
        states = torch.tensor(batch_experiences[:, :12], dtype=torch.float32)
        actions = torch.tensor(batch_experiences[:, 12:15], dtype=torch.float32)
        rewards = torch.tensor(batch_experiences[:, 15:16], dtype=torch.float32)
        next_states = torch.tensor(batch_experiences[:, 16:28], dtype=torch.float32)
        dones = torch.tensor(batch_experiences[:, 28:29], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    def __init__(self, state_dim, action_dim, buffer_size, batch_size, actor_lr, critic_lr, alpha_lr,
                 target_entropy, gamma, tau, bl, br, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.Bl = bl
        self.Br = br
        self.device = device
        self.buffer = ReplayBuffer(buffer_size, batch_size)
        self.target_entropy = target_entropy
        self.loss_values = []
        self.actor = PolicyNet(state_dim, action_dim).to(device)
        self.critic_1 = QValueNet(state_dim, action_dim).to(device)
        self.critic_2 = QValueNet(state_dim, action_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim, action_dim).to(device)
        self.target_critic_2 = QValueNet(state_dim, action_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action, _ = self.actor.sample(state)
        discrete_action = self.discretize_action(action.detach().cpu().numpy()[0], self.Bl, self.Br)
        return discrete_action

    def discretize_action(self, continuous_actions, Bl, Br):
        discrete_actions = []
        for action in continuous_actions:
            m = Bl + ((action + 1) * ((Br - Bl) / 2))
            dis_VSL = round((m - Bl) / 2) * 2 + Bl
            dis_VSL = round(dis_VSL / 10) * 10
            discrete_actions.append(dis_VSL)
        return discrete_actions

    def calc_target(self, rewards, next_states, dones):
        next_actions, next_log_probs = self.actor.sample(next_states)
        target_q1 = self.target_critic_1(next_states, next_actions)
        target_q2 = self.target_critic_2(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_probs
        target_q = torch.mean(target_q, dim=1, keepdim=True)
        td_target = rewards + (1 - dones) * self.gamma * target_q
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param * self.tau)

    def update(self):
        if len(self.buffer) < self.buffer.minimal_size:
            return
        else:
            print("********************************Update********************************")
        states, actions, rewards, next_states, dones = self.buffer.sample()
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        target_q = self.calc_target(rewards, next_states, dones).detach()
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        critic_1_loss = torch.mean(nn.functional.mse_loss(current_q1, target_q))
        critic_2_loss = torch.mean(nn.functional.mse_loss(current_q2, target_q))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        actions, log_probs = self.actor.sample(states)
        entropy = -log_probs
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1, q2))
        self.loss_values.append(actor_loss.detach().cpu().numpy())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update(self.target_critic_1, self.critic_1)
        self.soft_update(self.target_critic_2, self.critic_2)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def save_model(self, save_path):
        torch.save(self.actor.state_dict(), f'{save_path}/policy_net.pth')
        torch.save(self.critic_1.state_dict(), f'{save_path}/q_net1.pth')
        torch.save(self.critic_2.state_dict(), f'{save_path}/q_net2.pth')
        torch.save(self.target_critic_1.state_dict(), f'{save_path}/target_q_net1.pth')
        torch.save(self.target_critic_2.state_dict(), f'{save_path}/target_q_net2.pth')
        print(f"Model saved to {save_path}")

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(f'{path}/policy_net.pth'))
        self.critic_1.load_state_dict(torch.load(f'{path}/q_net1.pth'))
        self.critic_2.load_state_dict(torch.load(f'{path}/q_net2.pth'))
        self.target_critic_1.load_state_dict(torch.load(f'{path}/target_q_net1.pth'))
        self.target_critic_2.load_state_dict(torch.load(f'{path}/target_q_net2.pth'))
        print(f"Model loaded from {path}")
