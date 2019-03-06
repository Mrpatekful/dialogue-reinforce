"""
@author:    Patrik Purgai
@copyright: Copyright 2019, rlchat
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.01.20.
"""

# pylint: disable=import-error

import torch

from parlai.core.worlds import DialogPartnerWorld  


class SelfDialogWorld(DialogPartnerWorld):

    def __init__(self, opt, active, frozen, teacher, shared=None):
        self.id = 'SelfDialogWorld'
        self.active = active
        self.frozen = frozen
        self.acts = []
        self.episode_batch = None
        super(SelfDialogWorld, self).__init__(opt, [self.active, self.frozen], shared)

    def parley(self):
        if self.frozen.observation.get('episode_done', False):
            self.episode_batch = self.frozen.observation['batch']
            self.frozen.reset(batch_size=self.episode_batch['task'].size(0), retain_actions=False)
            self.active.reset(batch_size=self.episode_batch['task'].size(0), retain_actions=False)

            # ask multiple rounds of questions and record conversation
            self.acts = []

            # qbot start with task embedding
            self.frozen.observe(None)

        # qbot ask a question and observe it as well
        frozen_batch = self.frozen.act()
        frozen_batch['text'] = frozen_batch['text'].detach()

        self.active.observe({
            'text':     frozen_batch['text'],
            'id':       self.active.id,
        })

        active_batch = self.active.act()

        active_batch['text'] = active_batch['text'].detach()

        self.frozen.observe(active_batch)
        self.acts.extend([frozen_batch, active_batch])

    def save_agents(self, save_path):
        """Save complete world with all the agents, saves everything required to reload later."""
        active_state_dict = self.active.state_dict()
        abot_state_dict = self.abot.state_dict()
        torch.save({'qbot': qbot_state_dict, 'abot': abot_state_dict, 'opt': self.opt}, save_path)
