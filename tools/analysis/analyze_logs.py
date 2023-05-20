# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def cal_train_time(log_dicts, args):
    for i, log_dict in enumerate(log_dicts):
        print(f'{"-" * 5}Analyze train time of {args.json_logs[i]}{"-" * 5}')
        all_times = []
        for epoch in log_dict.keys():
            if args.include_outliers:
                all_times.append(log_dict[epoch]['time'])
            else:
                all_times.append(log_dict[epoch]['time'][1:])
        all_times = np.array(all_times)
        epoch_ave_time = all_times.mean(-1)
        slowest_epoch = epoch_ave_time.argmax()
        fastest_epoch = epoch_ave_time.argmin()
        std_over_epoch = epoch_ave_time.std()
        print(f'slowest epoch {slowest_epoch + 1}, '
              f'average time is {epoch_ave_time[slowest_epoch]:.4f}')
        print(f'fastest epoch {fastest_epoch + 1}, '
              f'average time is {epoch_ave_time[fastest_epoch]:.4f}')
        print(f'time std over epochs is {std_over_epoch:.4f}')
        print(f'average iter time: {np.mean(all_times):.4f} s/iter')
        print()

def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend # 图例
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys
    
    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        xs_dict = {}
        ys_dict = {}
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            if metric not in log_dict[epochs[0]]:
                raise KeyError(
                    f'{args.json_logs[i]} does not contain metric {metric}')
            xs = []
            ys = []
            num_iters_per_epoch = log_dict[epochs[0]]['iter'][-1]
            for epoch in epochs:
            # for epoch in range(1,4):
            
                iters = log_dict[epoch]['iter']
                if log_dict[epoch]['mode'][-1] == 'val':
                    iters = iters[:-1]
                # xs.append(np.array(iters) + (epoch - 1) * num_iters_per_epoch)
                # ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
                xs.append(np.expand_dims(np.array(epoch),axis=0))
                ys.append(np.array(log_dict[epoch][metric][-2:-1]))
                
                
            xs_dict[metric] = np.concatenate(xs)
            ys_dict[metric] = np.concatenate(ys)
        
        
        
        
        
        fig, ax1 = plt.subplots()
        ax1.plot(xs_dict[metrics[0]],ys_dict[metrics[0]],'g-', label=legend[0])
        ax1.set_xlabel('epoch')
        ax1.set_ylabel(legend[0])
        
        ax2 = ax1.twinx()
        ax2.plot(xs_dict[metrics[1]],ys_dict[metrics[1]],'b-', label=legend[1])
        ax2.set_ylabel(legend[1])
 
        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        
        # # plt.xlabel('iter')
        # plt.xlabel('epoch')
        # plt.plot(xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)
    if args.title is not None:
        plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out, dpi=600, bbox_inches='tight')
        plt.cla()

def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_curve', help='parser for plotting curves')
    parser_plt.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_plt.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['top1_acc'],
        help='the metric that you want to plot')
    parser_plt.add_argument('--title', type=str, help='title of figure')
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser_plt.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser_plt.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser_plt.add_argument('--out', type=str, default=None)

def add_time_parser(subparsers):
    parser_time = subparsers.add_parser(
        'cal_train_time',
        help='parser for computing the average time per training iteration')
    parser_time.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_time.add_argument(
        '--include-outliers',
        action='store_true',
        help='include the first value of every epoch when computing '
        'the average time')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, top1_acc
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for line in log_file:
                log = json.loads(line.strip())
                # skip lines without `epoch` field
                if 'epoch' not in log:
                    continue
                epoch = log.pop('epoch')
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    log_dict[epoch][k].append(v)
    return log_dicts

def main():
    args = parse_args()

    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')
   
    args.keys = ['acc_pose', 'loss']  # 修改需要抓取的指标
    args.legend = ['acc', 'loss']  # 修改图例
    args.out = 'data_visulization/acc_loss.pdf'
    
    
    
    
    
    log_dicts = load_json_logs(json_logs)

    eval(args.task)(log_dicts, args)
    
    

if __name__ == '__main__':
    main()
