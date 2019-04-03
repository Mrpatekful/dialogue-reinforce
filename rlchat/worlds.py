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
:param action: str, 

"""


def calculate_reward(actions):
    for index, (source, target) in enumerate(source_target_generator(actions)):
        print(index)
    return 1


def iterate_reponses(action):
    for response in action.responses:
        yield action.action, response.action


def source_target_generator(action):
    source_target_pairs = []

    if len(action.responses) == 0:
        return source_target_pairs
    
    for response in action.responses:
        source_target_pairs = chain(
            iterate_reponses(response), 
            source_target_generator(response))
        
    return source_target_pairs
    

class RLDialogWorld(MultiAgentDialogWorld):

    def __init__(self, opt, active_agent, static_agent,
                 teacher, shared=None):
        self.id = 'RLDialogWorld'
        self.episode_batch = None
        self.active_agent = active_agent
        self.static_agent = static_agent
        super(RLDialogWorld, self).__init__(
            opt, teacher + [static_agent, active_agent], 
            shared)

    def parley(self):
        """"""
        initial_action = self.agents[0].act()

        self.active_agent.zero_grad()
        actions = self.rollout(initial_action)

        reward = calculate_reward(actions)

        self.active_agent.observe({'reward': reward})
        self.active_agent.update_params()

    def rollout(self, initial_action):
        """"""
        def roll(action, num_rollouts):
            if num_rollouts == -1:
                return action

            print(action.id)

            if action.id == self.active_agent.id:
                self.static_agent.observe(action.action)
                static_action = Action(
                    id=self.static_agent.id, 
                    action=self.static_agent.act(), 
                    responses=[])

                action.responses.append(
                    roll(static_action, num_rollouts))

            else:
                self.active_agent.observe(action.action)
                num_rollouts -= 1
                for _ in range(self.opt['dialog_branches']):
                    active_action = Action(
                            id=self.active_agent.id,
                            action=self.active_agent.act(), 
                            responses=[])

                    action.responses.append(
                        roll(active_action, num_rollouts))

            return action

        return roll(Action(
                        id=self.static_agent.id, 
                        action=initial_action, 
                        responses=[]), 
                    self.opt['dialog_rounds'])
