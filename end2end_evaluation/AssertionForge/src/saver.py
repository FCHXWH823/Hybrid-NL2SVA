# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from config import FLAGS

# First, import the command line argument parser
from command_line_args import parse_command_line_args

# This will parse command line arguments and update config.FLAGS
FLAGS = parse_command_line_args()


from utils import get_ts, create_dir_if_not_exists, save, save_pickle
from utils import (
    get_root_path,
    get_model_info_as_str,
    extract_config_code,
    plot_scatter_line,
    plot_dist,
    save_pickle,
    dirname,
    estimate_model_size,
    set_ts,
    print_gpu_free_mem_info,
    get_gpu_info,
    print_stats,
)
import json


from collections import OrderedDict, defaultdict
from pprint import pprint
from os.path import join, dirname, basename
import torch
import networkx as nx
import numpy as np
import time


class MyTimer:
    def __init__(self) -> None:
        self.start = time.time()

    def elapsed_time(self):
        end = time.time()
        minutes, seconds = divmod(end - self.start, 60)

        return int(minutes)


class Saver(object):
    def __init__(self):

        self.logdir = join(
            get_root_path(),
            'logs',
            # '{}_{}_{}_{}_{}_{}_{}'.format(FLAGS.norm_method, FLAGS.task, FLAGS.subtask, FLAGS.tag, FLAGS.target, model_str, get_ts()))
            '{}_{}_{}_{}'.format(FLAGS.task, get_ts(), FLAGS.hostname, FLAGS.user),
        )

        self.accelerator = None

        self.pidstr = ''

        context = NoOpContextManager()

        with context:
            create_dir_if_not_exists(self.logdir)
            self.model_info_f = self._open(f'model_info{self.pidstr}.txt')
            self.plotdir = join(self.logdir, 'plot')
            # create_dir_if_not_exists(self.plotdir)
            self.objdir = join(self.logdir, 'obj')
            self._log_model_info()
            self._save_conf_code()
            self.timer = MyTimer()
            self.all_messages = set()
            self.msg_type_count = defaultdict(int)
            print('Logging to:\n{}'.format(self.logdir))
            print(basename(self.logdir))

        self.stats = defaultdict(list)

    def get_device(self):
        if FLAGS.mode == 'acc_launch':
            pid = self.accelerator.process_index
            which_gpu = pid
            saver.log_info(f'get_device: pid={pid}; which_gpu={which_gpu}')
            return f'cuda:{which_gpu}'
        else:
            return FLAGS.device

    def _open(self, f):
        return open(join(self.logdir, f), 'w')

    def close(self):
        self.print_stats()

        self.log_info(self.logdir)
        self.log_info(basename(self.logdir))
        if hasattr(self, 'log_f'):
            self.log_f.close()
        if hasattr(self, 'log_e'):
            self.log_e.close()
        if hasattr(self, 'log_d'):
            self.log_d.close()
        if hasattr(self, 'results_f'):
            self.results_f.close()

    def get_log_dir(self):
        return self.logdir

    def get_plot_dir(self):
        create_dir_if_not_exists(self.plotdir)
        return self.plotdir

    def get_obj_dir(self):
        create_dir_if_not_exists(self.objdir)
        return self.objdir

    def log_list_of_lists_to_csv(self, lol, fn, delimiter=','):
        import csv

        fp = open(join(self.logdir, fn), 'w+')
        csv_writer = csv.writer(fp, delimiter=delimiter)
        for l in lol:
            csv_writer.writerow(l)
        fp.close()

    def log_dict_of_dicts_to_csv(self, fn, csv_dict, csv_header, delimiter=','):
        import csv

        fp = open(join(self.logdir, f'{fn}.csv'), 'w+')
        f_writer = csv.DictWriter(fp, fieldnames=csv_header)
        f_writer.writeheader()
        for d, value in csv_dict.items():
            if d == 'header':
                continue
            f_writer.writerow(value)
        fp.close()

    def save_emb_accumulate_emb(self, data_key, d, convert_to_np=False):
        if not hasattr(self, 'emb_dict'):
            self.emb_dict = OrderedDict()

        if convert_to_np:
            new_d = {}
            for key, val in d.items():
                if torch.is_tensor(val):
                    val = val.detach().cpu().numpy()
                new_d[key] = val
            d = new_d

        self.emb_dict[data_key] = d

    def save_emb_save_to_disk(self, p):
        assert hasattr(self, 'emb_dict')
        filepath = join(self.objdir, p)
        create_dir_if_not_exists(dirname(filepath))
        save_pickle(self.emb_dict, filepath, print_msg=True)

    def save_emb_dict(self, d, p, convert_to_np=False):
        if not hasattr(self, 'save_emb_cnt'):
            self.save_emb_cnt = 0

        if convert_to_np:
            new_d = {}
            for key, val in d.items():
                if torch.is_tensor(val):
                    val = val.detach().cpu().numpy()
                new_d[key] = val
            d = new_d

        filepath = join(self.objdir, f'{self.save_emb_cnt}_{p}')
        create_dir_if_not_exists(dirname(filepath))
        save_pickle(d, filepath, print_msg=True)
        self.save_emb_cnt += 1

    def log_dict_to_json(self, dictionary, fn):
        import json

        # as requested in comment
        with open(join(self.get_obj_dir(), fn), 'w') as file:
            file.write(json.dumps(dictionary))

    def log_model_architecture(self, model):
        # print(model)
        if hasattr(self, 'has_logged_arch') and self.has_logged_arch:
            return  # avoid logging again and again within one python process
        if not hasattr(self, 'has_logged_arch'):
            self.has_logged_arch = True
        self.model_info_f.write('{}\n'.format(model))
        estimate_model_size(model, 'whole model', self)
        self.model_info_f.flush()
        # self.model_info_f.close()  # TODO: check in future if we write more to it

    # def log_info(self, s, silent=False, build_str=None):
    #     if not silent:
    #         print(s)
    #     if not hasattr(self, 'log_f'):
    #         self.log_f = self._open(f'log{self.pidstr}.txt')
    #     try:
    #         self.log_f.write('{}\n'.format(s))
    #         self.log_f.flush()
    #     except:
    #         raise RuntimeError()
    #     if build_str is not None:
    #         assert type(build_str) is str
    #         build_str += '{}\n'.format(s)
    #         return build_str

    def log_info(self, s, silent=False, build_str=None):
        # Pretty-printing logic
        if isinstance(s, (list, dict)):
            s = json.dumps(s, indent=4)
        elif hasattr(s, 'chat_history'):
            s = json.dumps(s.chat_history, indent=4)

        if not silent:
            print(s)
        if not hasattr(self, 'log_f'):
            self.log_f = self._open(f'log{self.pidstr}.txt')
        try:
            self.log_f.write('{}\n'.format(s))
            self.log_f.flush()
        except:
            raise RuntimeError()
        if build_str is not None:
            assert isinstance(build_str, str)
            build_str += '{}\n'.format(s)
            return build_str

    def log_info_once(self, s, silent=False):
        if s not in self.all_messages:
            self.all_messages.add(s)
            self.log_info(s, silent)

    def log_info_at_most(self, s, msg_type, times, silent=False):
        if self.msg_type_count[msg_type] < times:
            self.log_info(s, silent)
            self.msg_type_count[msg_type] += 1

    def info(self, s, silent=False):
        elapsed = self.timer.elapsed_time()
        if not silent:
            print(f'[{elapsed}m] INFO: {s}')
        if not hasattr(self, 'log_f'):
            self.log_f = self._open('log.txt')
        self.log_f.write(f'[{elapsed}m] INFO: {s}\n')
        self.log_f.flush()

    def error(self, s, silent=False):
        elapsed = self.timer.elapsed_time()
        if not silent:
            print(f'[{elapsed}m] ERROR: {s}')
        if not hasattr(self, 'log_e'):
            self.log_e = self._open('error.txt')
        self.log_e.write(f'[{elapsed}m] ERROR: {s}\n')
        self.log_e.flush()

    def warning(self, s, silent=False):
        elapsed = self.timer.elapsed_time()
        if not silent:
            print(f'[{elapsed}m] WARNING: {s}')
        if not hasattr(self, 'log_f'):
            self.log_f = self._open('log.txt')
        self.log_f.write(f'[{elapsed}m] WARNING: {s}\n')
        self.log_f.flush()

    def debug(self, s, silent=True):
        elapsed = self.timer.elapsed_time()
        if not silent:
            print(f'[{elapsed}m] DEBUG: {s}')
        if not hasattr(self, 'log_d'):
            self.log_d = self._open('debug.txt')
        self.log_d.write(f'[{elapsed}m] DEBUG: {s}\n')
        self.log_d.flush()

    def log_info_new_file(self, s, fn):
        # print(s)
        log_f = open(join(self.logdir, fn), 'a')
        log_f.write('{}\n'.format(s))
        log_f.close()

    def save_dict(self, d, p, subfolder=''):
        filepath = join(self.logdir, subfolder, p)
        create_dir_if_not_exists(dirname(filepath))
        save_pickle(d, filepath, print_msg=True)
        # print(f'dict of keys {d.keys()} saved to {filepath}')

    def _save_conf_code(self):
        with open(join(self.logdir, 'config.py'), 'w') as f:
            f.write(extract_config_code())
        p = join(self.get_log_dir(), 'FLAGS')
        save({'FLAGS': FLAGS}, p, print_msg=False)

    def save_graph_as_gexf(self, g, fn):
        nx.write_gexf(g, join(self.get_obj_dir(), fn))

    def save_overall_time(self, overall_time):
        self._save_to_result_file(overall_time, 'overall time')

    def save_exception_msg(self, msg):
        with self._open('exception.txt') as f:
            f.write('{}\n'.format(msg))

    def save_dict_as_pickle(self, d, name):
        p = join(self.get_obj_dir(), name)
        p = save(d, p, print_msg=True, use_klepto=False)
        self.log_info(f'Saving dict {name}')
        return p

    def _get_model_str(self):
        li = []
        key_flags = [FLAGS.model, FLAGS.dataset]
        for f in key_flags:
            li.append(str(f))
        rtn = '_'.join(li)
        return f'{rtn}'

    def _log_model_info(self):
        s = get_model_info_as_str(FLAGS)
        print(s)
        self.model_info_f.write(s)
        self.model_info_f.write('\n\n')
        self.model_info_f.flush()
        # self.writer.add_text('model_info_str', s)

    def log_new_FLAGS_to_model_info(self):
        self.model_info_f.write('----- new model info after loading\n')
        self._log_model_info()
        self._save_conf_code()

    def _save_to_result_file(self, obj, name=None, to_print=False):
        if not hasattr(self, 'results_f'):
            self.results_f = self._open('results.txt')
        if type(obj) is dict or type(obj) is OrderedDict:
            # self.f.write('{}:\n'.format(name))
            # for key, value in obj.items():
            #     self.f.write('\t{}: {}\n'.format(key, value))
            pprint(obj, stream=self.results_f)
        elif type(obj) is str:
            if to_print:
                print(obj)
            self.results_f.write('{}\n'.format(obj))
        else:
            self.results_f.write('{}: {}\n'.format(name, obj))
        self.results_f.flush()

    def _shrink_space_pairs(self, pairs):
        for _, pair in pairs.items():
            # print(pair.__dict__)
            pair.shrink_space_for_save()
            # pass
            # print(pair.__dict__)
            # exit(-1)
        return pairs

    def save_stats(self, stat_name, value):
        self.stats[stat_name].append(value)

    def print_stats(self):
        for stat_name, li in self.stats.items():
            print_stats(li, stat_name, saver=self)


class NoOpContextManager:
    """A no-operation context manager that does nothing."""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


saver = Saver()  # can be used by `from saver import saver`
