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
import signal
import json

from os.path import isfile

from parlai.core.params import ParlaiParser
from parlai.core.logs import TensorboardLogger
from parlai.core.utils import Timer
from parlai.core.agents import _create_task_agents
from parlai.scripts.train_model import TrainLoop, setup_args
from parlai.scripts.build_pytorch_data import get_pyt_dict_file
from parlai.scripts.build_dict import build_dict

from worlds import RLDialogWorld
from agents import create_agent, freeze_agent  # pylint: disable=import-error


def setup_rl_args():
    parser = setup_args()
    reinforce = parser.add_argument_group('Reinforce Arguments')
    reinforce.add_argument(
        '-dl', '--dialog_rounds',
        type=int, default=3,
        help='Number of rollouts rounds for estimating the reward.')
    reinforce.add_argument(
        '-dl', '--dialog_branches',
        type=int, default=3,
        help='Branches of the active agent responses during rollout.')
    return parser


def create_task(opt, active_agent, static_agent):
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

    world = create_task_world(opt, active_agent, static_agent)

    if opt.get('batchsize', 1) > 1:
        raise NotImplementedError('Btaching is not implemented yet.')
        # world = BatchWorld(opt, world)

    return world


def create_task_world(opt, active_agent, static_agent):
    teacher = _create_task_agents(opt)
    return RLDialogWorld(opt, active_agent, static_agent, teacher)


class ReinforceLoop(TrainLoop):

    def __init__(self, opt):
        signal.signal(signal.SIGINT, signal.default_int_handler)

        if isinstance(opt, ParlaiParser):
            opt = opt.parse_args()
        # Possibly load from checkpoint
        trainstats_suffix = '.trainstats'
        if (opt.get('model_file') and isfile(
                opt['model_file'] + '.checkpoint')):
            opt['init_model'] = opt['model_file'] + '.checkpoint'
            trainstats_suffix = '.checkpoint.trainstats'
        else:
            pass
            # TODO for testing only
            # raise RuntimeError('WARNING: Reinforcement learning'
            #                    ' must be initialized by a model.checkpoint '
            #                    'file and {} does not exist.'.format(
            #                        opt['model_file'] + '.checkpoint'))
        # Possibly build a dictionary (not all models do this).
        if (
            opt['dict_build_first'] and
            not (opt.get('dict_file') or opt.get('model_file'))
        ):
            raise RuntimeError('WARNING: For train_model, '
                               'please specify either a '
                               'model_file or dict_file.')

        if opt['dict_build_first'] and 'dict_file' in opt:
            if opt.get('pytorch_teacher_task'):
                opt['dict_file'] = get_pyt_dict_file(opt)
            elif opt['dict_file'] is None and opt.get('model_file'):
                opt['dict_file'] = opt['model_file'] + '.dict'
            print("[ building dictionary first... ]")
            build_dict(opt, skip_if_built=True)

        # Create model and assign it to the specified task
        self.agent = create_agent(opt)

        # Freeze the model for the static dialogue partner
        static_agent = copy.deepcopy(self.agent)
        self.agent.id += 'Active'

        static_agent.id += 'Static'
        freeze_agent(static_agent)

        self.world = create_task(opt, self.agent, static_agent)

        # set up timers
        self.train_time = Timer()
        self.validate_time = Timer()
        self.log_time = Timer()
        self.save_time = Timer()
        print('[ training... ]')

        self.parleys = 0
        self.max_num_epochs = (
            opt['num_epochs'] if 
            opt['num_epochs'] > 0 
            else float('inf'))

        self.max_train_time = (
            opt['max_train_time'] if 
            opt['max_train_time'] > 0 
            else float('inf'))

        self.log_every_n_secs = (
            opt['log_every_n_secs'] if 
            opt['log_every_n_secs'] > 0 
            else float('inf'))

        self.val_every_n_secs = (
            opt['validation_every_n_secs'] if 
            opt['validation_every_n_secs'] > 0
            else float('inf'))

        self.save_every_n_secs = (
            opt['save_every_n_secs'] if 
            opt['save_every_n_secs'] > 0 
            else float('inf'))

        self.val_every_n_epochs = (
            opt['validation_every_n_epochs'] if
            opt['validation_every_n_epochs'] > 0
            else float('inf'))

        # smart defaults for --validation-metric-mode
        if opt['validation_metric'] in {'loss', 'ppl', 'mean_rank'}:
            opt['validation_metric_mode'] = 'min'
        elif opt['validation_metric'] in {
            'accuracy', 'hits@1', 'hits@5', 'f1', 'bleu'}:
            opt['validation_metric_mode'] = 'max'
        if opt.get('validation_metric_mode') is None:
            opt['validation_metric_mode'] = 'max'

        self.last_valid_epoch = 0
        self.valid_optim = (1 if opt['validation_metric_mode'] == 
            'max' else -1)
        self.valid_reports = []
        self.best_valid = None
        if (opt.get('model_file') and 
                isfile(opt['model_file'] + '.best_valid')):
            with open(opt['model_file'] + ".best_valid", 'r') as f:
                x = f.readline()
                self.best_valid = float(x)
                f.close()
        self.impatience = 0
        self.saved = False
        self.valid_world = None
        self.opt = opt

        # we may have been preempted, make sure we note that amount
        self._preempted_epochs = 0.0
        if (
            opt.get('model_file') and
            isfile(opt['model_file'] + trainstats_suffix)
        ):
            # looks like we were preempted. make sure we load up our total
            # training stats, etc
            with open(opt['model_file'] + trainstats_suffix) as ts:
                obj = json.load(ts)
                self._preempted_epochs = obj.get('total_epochs', 0)
                self.train_time.total = obj.get('train_time', 0)
                self.impatience = obj.get('impatience', 0)
                self.valid_reports = obj.get('valid_reports', [])

        if opt['tensorboard_log'] is True:
            self.writer = TensorboardLogger(opt)


if __name__ == '__main__':
    ReinforceLoop(setup_rl_args().parse_args()).train()
