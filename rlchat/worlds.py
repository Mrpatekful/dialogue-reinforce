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


def calculate_reward(actions):
    return 0


Action = namedtuple('Action', ['actor_id', 'action', 'responses'])
"""

"""


class RLDialogWorld(MultiAgentDialogWorld):

    def __init__(self, opt, static_agent, active_agent, 
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
            if num_rollouts == 0:
                return action

            num_rollouts -= 1

            if action.actor_id == self.active_agent.id:
                self.static_agent.observe(action.action)
                static_action = Action(
                    actor_id=self.static_agent.id, 
                    action=self.static_agent.act(), 
                    responses=[])

                action.responses.append(
                    roll(static_action, num_rollouts))

            else:
                self.active_agent.observe(action.action)
                for _ in range(self.opt['dialog_branches']):
                    active_action = Action(
                            actor_id=self.active_agent.id,
                            action=self.active_agent.act(), 
                            responses=[])

                    action.responses.append(
                        roll(active_action, num_rollouts))

            return action
                        
        return roll(Action(
                        actor_id=self.static_agent.id, 
                        action=initial_action, 
                        responses=[]), 
                    self.opt['dialog_rounds'])
