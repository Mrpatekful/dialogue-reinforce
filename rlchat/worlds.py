"""

@author:    Patrik Purgai
@copyright: Copyright 2019, rlchat
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.02.20.
"""

# pylint: disable=import-error

import torch

from torch.autograd import backward

from parlai.core.worlds import MultiAgentDialogWorld
from collections import namedtuple
from itertools import chain


Action = namedtuple('Action', ['id', 'action', 'responses'])
"""
A node in the action-response tree.
:param id: str, the id of the actor.
:param action: dict, containing the agent actions.
:param responses: dict, containing the agent's response for the action.
"""

ACTIVE, STATIC = 'active', 'static'


class RLDialogWorld(MultiAgentDialogWorld):

    def __init__(self, opt, active_agent, static_agent, 
                 teacher, shared=None):
        """"""
        self.id = 'RLDialogWorld'
        self.episode_batch = None
        self.active_agent = active_agent
        self.static_agent = static_agent
        agents = teacher + [static_agent, active_agent]

        super(RLDialogWorld, self).__init__(opt, agents, shared)

    def parley(self):
        """"""
        # Initial sentence from the dataset
        initial_action = self.agents[0].act()

        self.active_agent.zero_grad()
        actions = self.rollout(initial_action)

        reward = self.calculate_reward(actions, 1)

        self.active_agent.observe({'reward': reward})
        self.active_agent.update_params()

    def iterate_reponses(self, action):
        """"""
        for response in action.responses:
            yield {
                'text': action.action['text'], 
                'text_vec': action.action['text_vec'], 
                'labels': response.action['text'],
                'labels_vec': response.action.get('text_vec', 
                    self.static_agent._vectorize_text(
                        response.action['text']))
            }

    def calculate_reward(self, action, weight):
        """"""
        reward = 0

        if len(action.responses) == 0:
            return 0

        if action.id == ACTIVE:
            obs_batch = list(self.iterate_reponses(action))
            batch = self.static_agent.batchify(obs_batch)
            log_prob = self.static_agent.compute_log_prob(batch, False)
            reward = reward + log_prob

        # Reduce reward significance for next dialogue turns
        weight = weight * self.opt['reward_decay']
        for response in action.responses:
            reward += self.calculate_reward(response, weight)
            
        return reward
    
    def rollout(self, initial_action):
        """"""
        def roll(action, num_rollouts):
            """"""
            if action.id == ACTIVE:
                self.static_agent.observe(action.action)
                act = self.static_agent.act()
                static_action = Action(
                    id=STATIC, 
                    action=act, 
                    responses=[])

                action.responses.append(
                    roll(static_action, num_rollouts))

            else:
                if num_rollouts == 0:
                    return action

                self.active_agent.observe(action.action)
                for _ in range(self.opt['dialog_branches']):
                    act = self.active_agent.act()
                    active_action = Action(
                        id=ACTIVE,
                        action=act, 
                        responses=[])

                    action.responses.append(
                        roll(active_action, num_rollouts - 1))

            return action

        initial = Action(STATIC, initial_action, [])

        return roll(initial, self.opt['dialog_rounds'])
