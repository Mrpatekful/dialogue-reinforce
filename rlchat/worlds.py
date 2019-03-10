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


def calculate_reward(actions):
    return 0


class RLDialogWorld(MultiAgentDialogWorld):

    def __init__(self, opt, static_agent, active_agent, teacher, shared=None):
        self.id = 'RLDialogWorld'
        self.actions = []
        self.rollout_actions = []
        self.episode_batch = None
        super(RLDialogWorld, self).__init__(
            opt, teacher + [static_agent, active_agent], shared)

    def parley(self):
        """"""
        print(self.agents[0].act())
        # self.active_agent.zero_grad()
        # static_action = self.static_agent.act()

        # for _ in range(self.opt['rollout_len']):
        #     action = self.rollout(action)

        # reward = calculate_reward(self.rollout_actions)
        # backward(self.rollout_actions, 
        #     [None for _ in self.rollout_actions], retain_graph=True)
        # self.active_agent.update_params()

    def rollout(self, batch):
        """"""
        pass
        # self.active_agent.observe(batch)
        # rollout_actions = []
        # for _ in range(self.opt['branch_size']):
        #     rollout_actions.append(self.active_agent.act())
