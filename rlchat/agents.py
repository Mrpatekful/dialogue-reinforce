"""

@author:    Patrik Purgai
@copyright: Copyright 2019, rlchat
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.02.20.
"""

# pylint: disable=no-value-for-parameter

import torch

from parlai.core.agents import (
    _load_opt_file, 
    get_agent_module, 
    add_task_flags_to_agent_opt,
    get_task_module)

from parlai.core.torch_generator_agent import (
    TorchGeneratorModel,
    TorchGeneratorAgent)

from parlai.core.torch_agent import Output
from parlai.core.utils import padded_tensor

from torch.autograd import Variable, backward
from torch.distributions import Categorical
from torch.nn import Module

from os.path import isfile


def decode_probabilistic(model, encoder_states, batch_size, 
                         max_len, num_det_steps):
    """
    Probabilistic search.

    Arguments:
        encoder_states: Output of the encoder model.
        batch_size: Batch size. Because encoder_states is 
            model-specific, it cannot infer this automatically.
        max_len: Maximum decoding length.
        num_det_steps: Number of deterministic iterations.
    """
    step_output = model._starts(batch_size)
    incr_state = None
    logits = []
    for step in range(max_len):
        scores, incr_state = model.decoder(
            step_output, encoder_states, incr_state)
        scores = scores[:, -1:, :]
        scores = model.output(scores)

        if step < num_det_steps:
            _, preds = scores.max(dim=-1)
        else:
            distribution = Categorical(scores)
            preds = distribution.sample()
        
        logits.append(scores)
        step_output = torch.cat([step_output, preds], dim=1)  # pylint: disable=no-member

        # check if everyone has generated an end token
        all_finished = (
            ((step_output == model.END_IDX).sum(dim=1) > 0)
                .sum()
                .item()
            ) == batch_size

        if all_finished:
            break

    logits = torch.cat(logits, 1)  # pylint: disable=no-member

    return logits, step_output


def compute_det_steps(model, step, max_len=None):
    """
    Computes the number of deterministic steps for
    curicullum learning.
    
    Arguments:
        model: ``TorchGeneratorModel`` type model instance.
        step: int, the number of performed steps.
        max_len: maximum length of a sequence.    
    """
    # TODO implement valid det step calculation
    default_num_dep_step = min(max_len // 5, 3) if max_len else 3

    return model.opt.get('num_det_step', default_num_dep_step)


def forward(self, *inputs, ys=None, cand_params=None, 
            prev_enc=None, maxlen=None, bsz=None, 
            step=None, use_probabilistic_decode=False):
    """
    Calculates the forward pass for the model. This function
    is assigned to `TorchGeneratorModel` instances with descriptor 
    protocol to replace the `decode_greedy` function with 
    `decode_probabilistic`.
    """
    if ys is not None:
        # TODO: get rid of longest_label
        # keep track of longest label we've ever seen
        # we'll never produce longer ones than that 
        # during prediction
        self.longest_label = max(
            self.longest_label, ys.size(1))

    # use cached encoding if available
    encoder_states = (
        prev_enc if prev_enc is 
        not None else self.encoder(*inputs)
    )

    if ys is not None:
        # use teacher forcing
        scores, preds = self.decode_forced(encoder_states, ys)
    else:
        if use_probabilistic_decode:
            scores, preds = self.decode_probabilistic(
                encoder_states,
                bsz,
                maxlen or self.longest_label,
                compute_det_steps(self, step, maxlen))
        else:
            scores, preds = self.decode_greedy(
                encoder_states,
                bsz,
                maxlen or self.longest_label)

    return scores, preds, encoder_states


def replace_forward(model):
    """
    Extends the ``torch.nn.Module`` type model instance
    by adding/replacing bound methods.
    """
    assert model is not None, ('Model must be initialized by '
                               '``build_model`` method.')

    model.forward = forward.__get__(model)


def get_agent_type(opt):
    """
    Returns the type of model agent, specified by --model and
    --model_file.
    """
    model_file = opt['model_file']
    optfile = model_file + '.opt'
    if isfile(optfile):
        new_opt = _load_opt_file(optfile)
        if 'batchindex' in new_opt:
            del new_opt['batchindex']
        if opt.get('override'):
            for k, v in opt['override'].items():
                if str(v) != str(new_opt.get(k, None)):
                    print(
                        "[ warning: overriding opt['{}'] to {} ("
                        "previously: {} )]".format(
                            k, v, new_opt.get(k, None)))
                new_opt[k] = v
        for k, v in opt.items():
            if k not in new_opt:
                new_opt[k] = v
        new_opt['model_file'] = model_file
        if (new_opt.get('dict_file') and not 
                isfile(new_opt['dict_file'])):
            raise RuntimeError(
                'WARNING: Dict file does not exist, check '
                'to make sure it is correct: {}'.format(
                    new_opt['dict_file']))
        model_class = get_agent_module(new_opt['model'])

        return model_class
    else:
        return None


def create_agent(opt):
    """
    Creates a new class, that extends the provided 
    subclass of ``TorchGeneratorAgent`` with reinforcement
    learning functionality.
    """
    torch_generator_agent_subclass = get_agent_type(opt)
    assert torch_generator_agent_subclass is not None

    class RLTorchGeneratorAgent(torch_generator_agent_subclass):

        def __init__(self, *args, shared=None, **kwargs):
            super(RLTorchGeneratorAgent, self).__init__(
                *args, shared=shared, **kwargs)
            self.log_probs = [] # TODO implement shared actions

        @torch.no_grad()
        def sample_step(self, batch):
            """
            Sample a single batch of examples which will be used as 
            labels for the train step.
            """
            self.model.eval()

            _, preds, _ = self.model(
                *self._model_input(batch), ys=batch.label_vec, 
                use_probabilistic_decode=True)

            self.add_labels(batch, preds)

            return batch

        def add_labels(self, batch, label_vecs):
            labels = [self._v2t(l) for l in label_vecs]
            ys, y_lens = padded_tensor(
                label_vecs, self.NULL_IDX, self.use_cuda)
                
            batch.labels=labels
            batch.label_vec=ys
            batch.label_lengths=y_lens

        def compute_log_prob(self, batch):
            if batch.label_vec is None:
                raise ValueError('Cannot compute loss without a label.')
            model_output = self.model(
                *self._model_input(batch), ys=batch.label_vec)
            scores, preds, *_ = model_output
            score_view = scores.view(-1, scores.size(-1))
            log_prob = self.criterion(score_view, batch.label_vec.view(-1))

            notnull = batch.label_vec.ne(self.NULL_IDX)
            target_tokens = notnull.long().sum().item()
            correct = ((batch.label_vec == preds) * notnull).sum().item()
            self.metrics['correct_tokens'] += correct
            self.metrics['nll_loss'] += log_prob.item()
            self.metrics['num_tokens'] += target_tokens
            log_prob /= target_tokens  # average loss per token
            return log_prob

        def train_step(self, batch):
            """
            Train on a single batch of examples.
            """
            try:
                batch = self.sample_step(batch)
                log_prob = self.compute_log_prob(batch)
                self.metrics['loss'] += log_prob.item()
                self.log_probs.append(log_prob)

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch. '
                        'if this happens frequently, decrease batchsize or '
                        'truncate the inputs to the model.')
                    self.metrics['total_skipped_batches'] += 1
                    # gradients are synced on backward, 
                    # now this model is going to be
                    # out of sync! catch up with the other workers
                    self._init_cuda_buffer(8, 8, True)
                else:
                    raise e
            
            return Output(batch.labels, None)

        def observe(self, observation):
            """
            Calculates the reward and copies the model if provided
            additionally to the ``super().observe()``.
            """
            if observation.get('reward') is not None:
                reward = observation['reward']
                for log_prob in self.log_probs:
                    loss = log_prob * reward
                    loss.backward()
                
                self.log_probs = []

                for parameter in self.model.parameters():  # pylint: disable=access-member-before-definition
                    parameter.grad.data.clamp_(min=-5, max=5)

            if observation.get('model') is not None:
                # Deepcopied and frozen clone of the model
                self.model = observation['model']

            return super(RLTorchGeneratorAgent, self).observe(
                observation)

        def build_model(self, *args, **kwargs):
            super(RLTorchGeneratorAgent, self).build_model(
                *args, **kwargs)
            replace_forward(self.model)

    return RLTorchGeneratorAgent(opt)


def freeze_agent(agent):
    for parameter in agent.model.parameters():
        parameter.requires_grad = False
    agent.model.eval()
