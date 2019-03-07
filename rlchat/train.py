"""

@author:    Patrik Purgai
@copyright: Copyright 2019, rlchat
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.02.20.
"""

import numpy as np
import torch

from parlai.core.params import ParlaiParser
from parlai.core.logs import TensorboardLogger
from parlai.core.agents import _create_task_agents

from world import SelfDialogWorld
from agents import create_agents


def setup_args():
    parser = ParlaiParser(add_model_args=True)
    train = parser.add_argument_group('Training Arguments')
    train.add_argument(
        '-eps', '--num-epochs', type=float, default=-1)
    train.add_argument(
        '-vtim', '--validation-every-n-secs',
        type=float, default=-1,
        help='Validate every n seconds. Whenever the the best '
                'validation metric is found, saves the model to '
                'the model_file path if set.')
    train.add_argument(
        '-veps', '--validation-every-n-epochs',
        type=float, default=-1,
        help='Validate every n epochs. Whenever the the best '
            'validation metric is found, saves the model to '
            'the model_file path if set.')
    train.add_argument(
        '-vp', '--validation-patience',
        type=int, default=10,
        help=('number of iterations of validation where result'
                ' does not improve before we stop training'))
    train.add_argument(
        '-lfc', '--load-from-checkpoint',
        type='bool', default=False,
        hidden=True,
        help='load model from checkpoint if available')
    TensorboardLogger.add_cmdline_args(parser)
    return parser


def create_optimizer(opt, model):
    return torch.optim.Adam([
            {'params': model.parameters(), 
             ' lr': opt['learning_rate']}
        ])


def create_task(opt, ):
    """Creates a world + task_agents (aka a task)
    assuming ``opt['task']="task_dir:teacher_class:options"``
    """
    task = opt.get('task')
    pyt_task = opt.get('pytorch_teacher_task')
    pyt_dataset = opt.get('pytorch_teacher_dataset')
    if not (task or pyt_task or pyt_dataset):
        raise RuntimeError(
            'No task specified. Please select a task with ' + 
            '--task {task_name}.'
        )

    if not task:
        opt['task'] = 'pytorch_teacher'

    world = create_task_world(opt, user_agents, 
        default_world=default_world)


def create_task_world(opt, user_agents):
    task_agents = _create_task_agents(opt)


def main(opt):
    dynamic, static = create_agents(opt)
    optimizer = create_optimizer(opt, dynamic)

    world = SelfDialogWorld(opt, dynamic, static)

    for epoch in range(opt['num_epochs']):
        for iteration in range(static.iteration_per_epoch):
            optimizer.zero_grad()

            for _ in range(opt['rounds']):
                world.parley()





if __name__ == '__main__':
    main(setup_args().parse_args())