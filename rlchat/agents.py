"""
@author:    Patrik Purgai
@copyright: Copyright 2019, rlchat
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.01.20.
"""

import torch

from parlai.core.agents import Agent
from parlai.core.torch_generator_agent import TorchGeneratorModel

from torch.autograd import Variable, backward

from torch.nn import Module


def create_agents(opt):
    return ReinforcedAgent(opt), ReinforcedAgent(opt)


def create_model(opt):
    return 


class ReinforceModel(TorchGeneratorModel):

    def decode_probabilistic(self, encoder_states, batch_size, max_len,
                             num_det_steps):
        """Probabilistic search
        
        Arguments:
            encoder_states: Output of the encoder model.
            batch_size: Batch size. Because encoder_states is model-specific, 
                it cannot infer this automatically.
            max_len: Maximum decoding length.
            num_det_iter: Number of deterministic iterations.
        """
        step_output = self._starts(batch_size)
        incr_state = None
        logits = []
        for step in range(max_len):
            scores, incr_state = self.decoder(
                step_output, encoder_states, incr_state)
            scores = scores[:, -1:, :]
            scores = self.output(scores)

            if step < num_det_steps:
                _, preds = scores.max(dim=-1)
            else:
                preds = scores.multinomial()
            
            logits.append(scores)
            step_output = torch.cat([step_output, preds], dim=1)  # pylint: disable=no-member

            # check if everyone has generated an end token
            all_finished = (
                (step_output == self.END_IDX).sum(dim=1) > 0).sum().item() == batch_size
            if all_finished:
                break

        logits = torch.cat(logits, 1)  # pylint: disable=no-member

        return logits, step_output


class DynamicAgent(Agent):

    def __init__(self, model, opt, shared=None):
        super(DynamicAgent, self).__init__(opt, model, shared)
        self.id = 'DynamicAgent'
        self.observation = None

        self.eval_flag = False
        self.actions = []

    def act(self):
        """Speak a token."""
        # compute softmax and choose a token
        outputs = self.model(self.observation)

        if self.eval_flag:
            _, actions = out_distr.max(1)
            actions = actions.unsqueeze(1)
        else:
            actions = out_distr.multinomial()
            self.actions.append(actions)
        return {'text': actions.squeeze(1), 'id': self.id}

    def observe(self, observation):
        """Dynamic agent updates its parameters"""
        super(DynamicAgent, self).observe(observation)
        if observation.get('reward') is not None:
            for action in self.actions:
                action.reinforce(observation['reward'])
            backward(self.actions, [None for _ in self.actions], retain_graph=True)

            # clamp all gradients between (-5, 5)
            for parameter in self.model.parameters():
                parameter.grad.data.clamp_(min=-5, max=5)


class StaticAgent(Agent):

    def __init__(self, model, opt, shared=None):
        super(StaticAgent, self).__init__(opt, model, shared)
        self.id = 'StaticAgent'
        self.observation = None
        self.teacher = create_teacher(opt)

        self.eval_flag = False
        self.actions = []

    def act(self):
        if self.observation is None:
            return self.teacher.act()
        else:
            action = self.model()
        
        if self.eval_flag:
            _, actions = out_distr.max(1)
            actions = actions.unsqueeze(1)
        else:
            actions = out_distr.multinomial()
            self.actions.append(actions)
        return {'text': actions.squeeze(1), 'id': self.id}

    def observe(self, observation):
        """"""
        super(StaticAgent, self).observe(observation)
        if observation.get('model') is not None:
            self.model = observation['model']


class Dummy(Module):

    def __init__(self):
        super(Dummy, self).__init__()
        