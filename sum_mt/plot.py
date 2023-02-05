import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse

def extract_from_dict(score_dict, negate=False):
    data = []
    for v in score_dict.values():
        score = -v['mean'] if negate else v['mean']
        data.append((v['edit_ratio'], score, v['std']))
    return sorted(data, key=lambda x:x[0])

def load_result(result_dir, negate=False, prefix='', select = None):
    '''
    list of (name, result) tuple
    each result is list of (editratio, mean) tuple

    add ref score to all other results
    '''
    results = []
    ref_score = None
    for file in os.listdir(result_dir):
        if file.endswith('.json') and (file.startswith(prefix) or 'ref' in file):
            if (not 'ref' in file and select is not None):
                if not any([ww in file for ww in select.split(',')]):
                    continue
                    
            with open(os.path.join(result_dir, file)) as f:
                result = extract_from_dict(json.load(f), negate=negate)
                name = file.split('.json')[0]
                if name == 'ref':
                    assert result[0][0] == 0, 'editratio for ref must be 0'
                    ref_score = result[0]
                else:
                    results.append((name, result))
    assert ref_score is not None, 'need score for reference!'
    for result in results:
        result[1].insert(0, ref_score)
    return results

def check_rank(score):
    for i in range(len(score) - 1):
        if score[i] <= score[i + 1] + 0.00001:
            return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('metric_results_dir')
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('-m', '--max_edr', type=float, default=1)
    parser.add_argument('-s', '--name_suffix', type=str, default='')
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument('-e', '--error_bar', action='store_true')
    parser.add_argument('-n', '--negate', action='store_true')
    parser.add_argument('--select', type=str, default = None)
    parser.add_argument('-p', '--prefix', type=str, default='', help='only plot scores that start with a prefix')
    args = parser.parse_args()
    metric_name = os.path.basename(args.metric_results_dir)
    if args.output_path is None:
        args.output_path = os.path.join(args.metric_results_dir, f'plot{args.name_suffix}.png')
    
    print('plot with max_edr:', args.max_edr)
    result = load_result(args.metric_results_dir, args.negate, prefix = args.prefix, select = args.select)
    fig = plt.figure(dpi=args.dpi)
    for name, data in result:
        edr, score, score_std = [list(el) for el in zip(*data)]
        num = sum([e < args.max_edr for e in edr]); assert(num >= 2)
        edr = edr[:num]; score = score[:num]; score_std = score_std[:num]

        if check_rank(score):
            plt.plot(edr[:num], score[:num], '.--', label=name, linewidth=0.6, markersize=3)
        else:
            plt.plot(edr[:num], score[:num], '.-', label=name, linewidth=0.8, markersize=5)
        if args.error_bar:
            plt.fill_between(edr, np.array(score)-np.array(score_std), np.array(score)+np.array(score_std), alpha=0.2)

    plt.title(args.metric_results_dir)
    plt.xlabel('edit-ratio', fontsize=13)
    plt.ylabel(metric_name, fontsize=13)

    plt.grid()
    plt.legend(fontsize=9)
    print('saving to', args.output_path)
    plt.savefig(args.output_path)
