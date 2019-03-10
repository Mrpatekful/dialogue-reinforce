"""

@author:    Patrik Purgai
@copyright: Copyright 2019, rlchat
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.02.20.
"""

import numpy as np
import torch
import copy

from parlai.core.params import ParlaiParser
from parlai.core.logs import TensorboardLogger
from parlai.core.agents import _create_task_agents

from worlds import RLDialogWorld
from agents import create_agent, freeze_agent


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
        '-dl', '--dialog-rounds',
        type=int, default=2,
        hidden=True,
        help='load model from checkpoint if available')
    TensorboardLogger.add_cmdline_args(parser)
    return parser


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
            '--task {task_name}.')

    if not task:
        opt['task'] = 'pytorch_teacher'

    active_agent, static_agent = create_agent(opt)

    world = create_task_world(opt, active_agent, static_agent)

    if opt.get('batchsize', 1) > 1:
        raise NotImplementedError('Btaching is not implemented yet.')
        # world = BatchWorld(opt, world)

    return world


def create_task_world(opt, active_agent, static_agent):
    teacher = _create_task_agents(opt)
    return RLDialogWorld(opt, active_agent, static_agent, teacher)


def main(opt):
    opt['batch_size'] = 1
    active_agent = create_agent(opt)
    static_agent = copy.deepcopy(active_agent)
    freeze_agent(static_agent)

    world = create_task_world(opt, active_agent, static_agent)

    world.parley()


if __name__ == '__main__':
    main(setup_args().parse_args())
