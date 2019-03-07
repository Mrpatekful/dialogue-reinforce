"""

@author:    Patrik Purgai
@copyright: Copyright 2019, rlchat
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.02.20.
"""

# pylint: disable=no-value-for-parameter

import torch

from parlai.core.agents import (_load_opt_file, get_agent_module,  # pylint: disable=no-name-in-module
                                add_task_flags_to_agent_opt,
                                get_task_module) 

from parlai.core.torch_generator_agent import TorchGeneratorModel, \
                                              TorchGeneratorAgent

from parlai.core.pytorch_data_teacher import PytorchDataTeacher

from torch.autograd import Variable, backward

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
            preds = scores.multinomial(num_samples=1)
        
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
            prev_enc=None, max_len=None, batch_size=None, 
            step=None):
    """
    Calculates the forward pass for the model. This function
    is assigned to `TorchGeneratorModel` instances with descripor 
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
        scores, preds = self.decode_probabilistic(
            encoder_states,
            batch_size,
            max_len or self.longest_label,
            compute_det_steps(self, step, max_len))

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

        def sample_step(self, batch):
            """
            Sample a single batch of examples.
            """
            self.model.eval()
            batch_size = batch.text_vec.size(0)
            batch_reply = [
                {'id': self.getID()} for _ in range(batch_size)
            ]
            output = self.model(
                *self._model_input(batch), ys=batch.label_vec)

            self.match_batch(
                batch_reply, batch.valid_indices, output)
            
            return self.batchify(batch_reply)

        def train_step(self, batch):
            """
            Train on a single batch of examples.
            """
            try:
                with torch.no_grad():
                    batch = self.sample_step(batch)
                with torch.enable_grad():
                    loss, model_output = self.compute_loss(
                        batch, return_output=True)
                self.metrics['loss'] += loss.item()

            except RuntimeError as e:
                # catch out of memory exceptions during fwd/bck (skip batch)
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

            return model_output
            
        def build_model(self, *args, **kwargs):
            super(RLTorchGeneratorAgent, self).build_model(*args, **kwargs)
            replace_forward(self.model)

    return RLTorchGeneratorAgent(opt)


def create_task_agent_from_taskname(opt):
    """
    Create task agent(s) assuming the input 
    ``task_dir:teacher_class``.
    """
    if not (opt.get('task') or
            opt.get('pytorch_teacher_task') or
            opt.get('pytorch_teacher_dataset')):
        raise RuntimeError(
            'No task specified. Please select a task with ' +
            '--task {task_name}.')
    if not opt.get('task'):
        opt['task'] = 'pytorch_teacher'
    teacher_class = get_task_module(opt['task'])
    add_task_flags_to_agent_opt(teacher_class, opt, opt['task'])
    return teacher_class


def create_teacher(opt, model=None):
    """Creates the teacher for the specified task."""
    pytorch_data_teacher_subclass = create_task_agent_from_taskname(opt)

    class RLPytorchDataTeacher(pytorch_data_teacher_subclass):

        def __init__(self, opt, model, shared=None):
            super(RLPytorchDataTeacher, self).__init__(opt, shared)
            self.model = model

        def batch_act(self, observations):
            if 

    return RLPytorchDataTeacher(opt, model)
