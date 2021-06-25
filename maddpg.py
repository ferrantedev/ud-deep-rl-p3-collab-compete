import torch

import random
from collections import namedtuple, deque
import numpy as np
from time import gmtime, strftime

from ddpg import Agent, CriticAgent
from buffer import ReplayBuffer

from os import listdir, getcwd
from os.path import isfile, join
import datetime
import re

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor

class MADDPG():
    
    def __init__(self, state_size, action_size, num_agents, random_seed=37):
        
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        
        # creating agents and store them into agents list
        self.agents = [ Agent(state_size, action_size, random_seed) for i in range(num_agents) ]
        self.critic = CriticAgent(state_size *num_agents, action_size*num_agents, random_seed)
        # creating replay buffer
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)

    # reset each agent's noise
    def reset(self):
        for agent in self.agents:
            agent.reset()
            
    def act(self, states, add_noise=False):
        actions = []
        for agent, state in zip(self.agents, states):
            action = agent.act(state, add_noise)
            action = np.reshape(action, newshape=(-1))
            actions.append(action)
        actions = np.stack(actions)
        return actions
        

    def step(self, states, actions, rewards, next_states, dones, i_episode):
        full_state = np.reshape(states, newshape=(-1))
        next_full_state = np.reshape(next_states, newshape=(-1))
        self.memory.add(states, full_state, actions, rewards, next_states, next_full_state, dones)

        if len(self.memory) > BATCH_SIZE and i_episode > 256:
            experiences = self.memory.sample()
            for agent, action, reward, done in zip(self.agents, actions, rewards, dones):
                agent_experience = self.unpack_agent_experience(agent, experiences)
                self.critic.learn(agent_experience, GAMMA)
                agent.learn(agent_experience, GAMMA, self.critic)


    def unpack_agent_experience(self, agent, experiences):
        states, full_states, actions, rewards, next_states, next_full_states, dones = experiences

        actor_target_actions = torch.zeros(actions.shape, dtype=torch.float, device=device)
        for agent_idx, agent_i in enumerate(self.agents):
            if agent == agent_i:
                agent_id = agent_idx
            agent_i_current_state = states[:,agent_idx]
            actor_target_actions[:,agent_idx,:] = agent_i.actor_target.forward(agent_i_current_state)
            
        actor_target_actions = actor_target_actions.view(BATCH_SIZE, -1)

        agent_state = states[:,agent_id,:]
        agent_action = actions[:,agent_id,:]
        agent_reward = rewards[:,agent_id].view(-1,1)
        agent_done = dones[:,agent_id].view(-1,1)
        
        
        actor_local_actions = actions.clone()
        actor_local_actions[:, agent_id, :] = agent.actor_local.forward(agent_state)
        actor_local_actions = actor_local_actions.view(BATCH_SIZE, -1)
        
        actions = actions.view(BATCH_SIZE, -1)
        
        
        agent_experience = (full_states, actions, actor_local_actions, actor_target_actions,
                            agent_state, agent_action, agent_reward, agent_done,
                            next_states, next_full_states)
        return agent_experience

            
    def save_checkpoint(self):
        timestamp = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        torch.save(self.critic.critic_local.state_dict(), f'checkpoint_critic_{timestamp}.pth')
        
        for idx, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), f'checkpoint_actor_{str(idx)}_{timestamp}.pth')
            
    def load_latest_checkpoint(self):
        self.load_agents_latest_checkpoints()
        self.load_critic_latest_checkpoint()
            
    def load_critic_latest_checkpoint(self):
        onlyfiles = [f for f in listdir(getcwd()) if isfile(join(getcwd(), f))]

        # Load the latest checkpoint file
        candidatecheckpoint = None
        candidate_datestring = None
        for f in onlyfiles:
            if 'checkpoint_critic' in f:
                datestring = f.strip('checkpoint_critic_').strip('.pth')
                checkpoint_datetime = datetime.datetime.strptime(datestring, '%Y-%m-%d_%H-%M-%S')
                if candidatecheckpoint is None:
                    candidatecheckpoint = f
                    candidate_datestring = checkpoint_datetime
                    continue
                elif checkpoint_datetime > candidate_datestring:
                    candidatecheckpoint = candidatecheckpoint
                    candidate_datestring = checkpoint_datetime
        if candidatecheckpoint is None:
            raise RuntimeError(f'Could not load checkpoint for agent number {str(agent_ix)}') from exc
        state_dict = torch.load(candidatecheckpoint)
        
        self.critic.critic_local.load_state_dict(state_dict)
        self.critic.critic_target.load_state_dict(state_dict)
        
    def load_agents_latest_checkpoints(self):
        onlyfiles = [f for f in listdir(getcwd()) if isfile(join(getcwd(), f))]

        for agent_ix in range(self.num_agents):
            # Load the latest checkpoint file
            candidatecheckpoint = None
            candidate_datestring = None
            for f in onlyfiles:
                regex_str = f'checkpoint_actor_{agent_ix}'
                if re.search(r'' + regex_str, f) is not None:
                    if 'checkpoint' in f:
                        datestring = f.strip(f'checkpoint_actor_{str(agent_ix)}_').strip('.pth')
                        checkpoint_datetime = datetime.datetime.strptime(datestring, '%Y-%m-%d_%H-%M-%S')
                        if candidatecheckpoint is None:
                            candidatecheckpoint = f
                            candidate_datestring = checkpoint_datetime
                            continue
                        elif checkpoint_datetime > candidate_datestring:
                            candidatecheckpoint = candidatecheckpoint
                            candidate_datestring = checkpoint_datetime
            if candidatecheckpoint is None:
                raise RuntimeError(f'Could not load checkpoint for agent number {str(agent_ix)}') from exc
            else:
                state_dict = torch.load(candidatecheckpoint)
                self.agents[agent_ix].actor_local.load_state_dict(state_dict)
                self.agents[agent_ix].actor_target.load_state_dict(state_dict)
        
            

