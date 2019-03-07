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

from parlai.core.worlds import DialogPartnerWorld, BatchWorld 


def calculate_reward(actions):
    return 0


class SelfDialogWorld(DialogPartnerWorld):

    def __init__(self, opt, static_agent, active_agent, shared=None):
        self.id = 'SelfDialogWorld'
        self.active_agent = active_agent
        self.static_agent = static_agent
        self.actions = []
        self.episode_batch = None
        super(SelfDialogWorld, self).__init__(
            opt, [self.static_agent, self.active_agent], shared)

    def parley(self):
        """"""
        if self.static_agent.observation is None:
            pass

        self.active_agent.zero_grad()

        for _ in range(self.opt['rollout_len']):
            batch = self.rollout(batch)

        reward = calculate_reward(self.actions)
        backward(self.actions, 
            [None for _ in self.actions], retain_graph=True)
        self.active_agent.update_params()

    def rollout(self, batch):
        """"""
        self.active_agent.observe(batch)
        rollout_actions = []
        for _ in range(self.opt['branch_size']):
            rollout_actions.append(self.active_agent.act())


class BatchSelfDialogWorld(BatchWorld):

    def __init__(self, opt, world):
        super(BatchSelfDialogWorld, self).__init__(opt, world)


    def batch_observe(self, index, batch_actions, index_acting):
        pass

    def parley(self):
        self.active_agent.zero_grad()

        for _ in range(self.opt['rollout_len']):
            self.rollout()

        loss.backward()
        self.update_params()
        self.active_agent.update_params()


    def rollout(self):
        pass