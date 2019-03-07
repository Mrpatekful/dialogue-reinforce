"""

@author:    Patrik Purgai
@copyright: Copyright 2019, rlchat
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.02.20.
"""

import torch

from parlai.core.agents import Agent
from parlai.core.torch_generator_agent import TorchGeneratorModel, \
                                              TorchGeneratorAgent

from parlai.core.pytorch_data_teacher import PytorchDataTeacher

from torch.autograd import Variable, backward

from torch.nn import Module


def create_agents(opt):
    return ReinforcedAgent(opt), ReinforcedAgent(opt)


def create_model(opt):
    return 


class ProbabilisticGeneratorModel(TorchGeneratorModel):

    def decode_probabilistic(self, encoder_states, batch_size, max_len,
                             num_det_steps):
        """Probabilistic search
        
        Arguments:
            encoder_states: Output of the encoder model.
            batch_size: Batch size. Because encoder_states is model-specific, 
                it cannot infer this automatically.
            max_len: Maximum decoding length.
            num_det_steps: Number of deterministic iterations.
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
                preds = scores.multinomial(num_samples=1)
            
            logits.append(scores)
            step_output = torch.cat([step_output, preds], dim=1)  # pylint: disable=no-member

            # check if everyone has generated an end token
            all_finished = (
                ((step_output == self.END_IDX).sum(dim=1) > 0)
                    .sum()
                    .item()
                ) == batch_size

            if all_finished:
                break

        logits = torch.cat(logits, 1)  # pylint: disable=no-member

        return logits, step_output

    def compute_det_steps(self, step, max_len=None):
        """"""
        # TODO implement valid det step calculation
        default_num_dep_step = min(max_len // 5, 3) if max_len else 3

        return self.opt.get('num_det_step', default_num_dep_step)

    def forward(self, *inputs, targets=None, cand_params=None, 
                prev_enc=None, max_len=None, batch_size=None, step=None):
        """"""
        if targets is not None:
            # TODO: get rid of longest_label
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, targets.size(1))

        # use cached encoding if available
        encoder_states = prev_enc if prev_enc is not None else self.encoder(
            *inputs)

        if targets is not None:
            # use teacher forcing
            scores, preds = self.decode_forced(encoder_states, targets)
        else:
            scores, preds = self.decode_probabilistic(
                encoder_states,
                batch_size,
                max_len or self.longest_label,
                self.compute_det_steps(step))

        return scores, preds, encoder_states


class SelfDialogTeacher(PytorchDataTeacher):

    def __init__(self, opt, model, shared=None):
        super(SelfDialogTeacher, self).__init__(opt)
        self.model = model


    

class DynamicAgent:

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

    def train_step(self, batch):
        pass

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
