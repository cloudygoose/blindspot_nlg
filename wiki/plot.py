import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
import math

def extract_from_dict(score_dict, negate=False):
    data = []
    for v in score_dict.values():
        score = -v['mean'] if negate else v['mean']
        data.append((v['edit_ratio'], score, v['std']))
    return sorted(data, key=lambda x:x[0])

def load_result(result_dir, negate=False, prefix='', select = None, skip = []):
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
                if not any([ww == file.split('.json')[0] for ww in select.split(',')]):
                    continue
            if any(file.split('.json')[0] == n for n in skip):
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
    return sorted(results, key=lambda x:x[0])

def check_rank(score):
    for i in range(len(score) - 1):
        if score[i] <= score[i + 1] + 0.00001: # - 0.0002: sometimes mauve is unstable around 0, it's unfair to penalize that as rank-incorrect
            return False
    return True


name_transforms = {
    'score_saves': 'wiki', # temp hack

    # wiki metrics
    'gpt-ppl': 'GPT-PPL',
    'mauve-gpt2': 'MAUVE-GPT2',
    'mauve-roberta': 'MAUVE-RoBERTa',
    'mauve-electra': 'MAUVE-ELECTRA',
    'mlm-ppl': 'MLM-PPL',

    # sum metrics
    'bart_score_cnn_avg_f': 'BARTS-cnn-f',
    'bart_score_cnn_hypo_ref': 'BARTS-cnn-r',
    'bart_score_cnn_ref_hypo': 'BARTS-cnn-p',
    'bart_score_cnn_src_hypo': 'BARTS-cnn-faithful',
    'bart_score_para_avg_f': 'BARTS-para-f',
    'bart_score_para_hypo_ref': 'BARTS-para-r',
    'bart_score_para_ref_hypo': 'BARTS-para-p',
    'bart_score_para_src_hypo': 'BARTS-para-faithful',
    'bert_score_f': 'BERTScore-f',
    'bert_score_p': 'BERTScore-p',
    'bert_score_r': 'BERTScore-r',
    'comet': 'COMET',
    'cometqe': 'COMET-QE',
    'mover_score': 'MoverScore',
    'rouge2-f': 'ROUGE-2',
    'rougeL-f': 'ROUGE-L',
    'unieval_overall': 'UniEval-overall',
    
    # wmt metrics
    'bart_score_cnn': 'BARTS-cnn-f',
    'bart_score_para': 'BARTS-para-f',
    'bert_score': 'BERTScore-f',
    'bleu': 'BLEU',
    'bleurt': 'BLEURT',
    # comet (above)
    # cometqe (above)
    # mover_score (above)
    'prism': 'PRISM',
    'prismqe': 'PRISM-QE',
    
    # consistency tests
    'con-switchsent': 'Sentence Switching',
    'con-switchsentnolast': 'Sentence Switching (nolast)',
    'con-replacesent': 'Sentence Replacement',
    'con-negate': 'Negation',
    'con-genericner': 'Generic Name Entity',
    'con-switchner': 'Named Entity Switching',
    'con-switchverb':  'Verb Switching',
    'con-switchnoun': 'Noun Switching',
    'con-bertdiverge': 'BERT-diverge',
    # extra consistency tests
    'con-switchsubsent': 'Subsentence Switching (all)',
    'con-switchsubsentnolast': 'Subsentence Switching',

    # fluency tests
    'flu-truncate': 'Truncation',
    'flu-removearticle': 'Article Removal',
    'flu-removepreposition': 'Preposition Removal',
    'flu-removestopwords': 'Stop-word Removal',
    'flu-lemmatizeverb': 'Verb Lemmatization',
    'flu-randomworddrop': 'Token Drop',
    'flu-randomtokenrep': 'Repeated Token',
    'flu-randomlocalswap': 'Local Swap',
    'flu-sentencemiddleswap': 'Middle Swap',
    'flu-noisepunct': 'Noised Punctuation'
}
def transform_name(name, disabled=False):
    if disabled: return name
    return name_transforms.get(name, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--metric_results_dir', type=str, nargs='+')
    parser.add_argument('-o', '--output_path', type=str, default=None)
    parser.add_argument('-m', '--max_edr', type=float, default=1)
    parser.add_argument('-s', '--name_suffix', type=str, default='')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--size', type=float, default=15)
    parser.add_argument('-e', '--error_bar', action='store_true')
    # parser.add_argument('-n', '--negate', action='store_true')
    parser.add_argument('--select', type=str, default = None, help='all tests that contain the name')
    parser.add_argument('-p', '--prefix', type=str, default='', help='only plot scores that start with a prefix')
    parser.add_argument('--skip', type=str, nargs='*', default=[], help='skip specific test (exact name match)')
    parser.add_argument('--disable_transform', action='store_true')
    parser.add_argument('--legend_title', type=str, default=None)
    parser.add_argument('--real_legend', action='store_true', help='show real legend for individual subplot')
    parser.add_argument('--legend_fontsize', type=float, default=19)
    
    # to draw appendix
    parser.add_argument('--tight', action='store_true')
    parser.add_argument('--ncols', type=int, default=4)
    parser.add_argument('--height_ratio', type=float, default=1.25)
    parser.add_argument('--pad', type=float, default=1.08)
    parser.add_argument('--rect', type=float, nargs=4, default=[0.01, 0, 0.99, 0.9])
    parser.add_argument('--legend_bbox', type=float, nargs=2, default=[0.5, 0.94])
    args = parser.parse_args()
    if args.legend_title is not None:
        args.legend_title = args.legend_title[0].upper() + args.legend_title[1:]
    n_plots = len(args.metric_results_dir)
    if n_plots == 1:
        subplot_args = (1, 1)
        figsize = (6.4, 4.8)
    elif n_plots == 6:
        subplot_args = (2, 3)
        figsize = (args.size, args.size/2)
    else:
        # for appendix
        subplot_args = (int(math.ceil(n_plots/args.ncols)), args.ncols)
        figsize = (args.size, args.size*args.height_ratio)

    fig, axs = plt.subplots(*subplot_args, figsize=figsize)
    if not isinstance(axs, np.ndarray): axs = np.array(axs)
    fig.set_dpi(args.dpi)
    fakefig, fakeax = plt.subplots()

    print('plot with max_edr:', args.max_edr)
    # text_kwargs = dict(ha='center', va='center', fontsize=28, color='C1')
    # build a rectangle in axes coords
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height

    built_fakeax = False
    for i, (ax, res_dir) in enumerate(zip(axs.flatten(), args.metric_results_dir)):
        metric_name = os.path.basename(res_dir)
        if args.output_path is None:
            # default output path is in first result dir
            args.output_path = os.path.join(res_dir, f'plot{args.name_suffix}.png')
        
        negate = 'ppl' in metric_name.lower()
        result = load_result(res_dir, negate, prefix = args.prefix, select = args.select, skip=args.skip)
        for name, data in result:
            edr, score, score_std = [list(el) for el in zip(*data)]
            num = sum([e < args.max_edr for e in edr]); assert(num >= 2)
            edr = edr[:num]; score = score[:num]; score_std = score_std[:num]

            if check_rank(score):
                ax.plot(edr[:num], score[:num], '.--', label=name, linewidth=1.5, markersize=5)
                line_type = '.--'
            else:
                ax.plot(edr[:num], score[:num], '.-', label=name, linewidth=1.9, markersize=7)
                line_type = '.-'
            if args.error_bar:
                ax.fill_between(edr, np.array(score)-np.array(score_std), np.array(score)+np.array(score_std), alpha=0.2)
            if not built_fakeax:
                fake_line_type = line_type if n_plots == 1 else '.-'
                fakeax.plot(edr[:num], score[:num], fake_line_type, label=name, linewidth=1.2, markersize=5)
        built_fakeax = True

        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        # ax.set_title(res_dir)
        if i >= n_plots -args.ncols:
            ax.set_xlabel('noise-ratio', size=14)
        # ax.set_ylabel(metric_name)
        ax.grid(linestyle='--', linewidth=0.5)
        # ax.text(0.5, 0.5, metric_name, **text_kwargs)

        display_name = f'{transform_name(metric_name, args.disable_transform)}\n({transform_name(os.path.basename(os.path.dirname(res_dir)), args.disable_transform)})'
        ax.text(0.5*(left+right), 0.5*(bottom+top), display_name,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=20, color='grey',
            transform=ax.transAxes
        )
        if args.real_legend:
            ax.legend()
    # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[:1]]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lines_labels = [fakeax.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    labels = [transform_name(label, args.disable_transform) for label in labels]
    # figlegend = plt.figure(figsize=(args.size/2, args.size/4))
    
    for i, ax in enumerate(axs.flatten()):
        if i >= n_plots:
            ax.set_axis_off()

    if labels == ['Negation', 'Sentence Replacement', 'Sentence Switching', 'Verb Lemmatization', 'Article Removal', 'Preposition Removal', 'Truncation']:
        correct_order = [6, 2, 4, 1, 5, 0, 3] # [6, 4, 5, 3, 2, 1, 0]
        lines = [lines[idx] for idx in correct_order]
        labels = [labels[idx] for idx in correct_order]

    if n_plots == 1:
        #(0.76, 0.32)
        # pass
        # bbox_to_anchor=(0.5, 0.25) by default
        fig.legend(lines, labels, loc='center', ncol=2, bbox_to_anchor=(0.5, 0.23), title=args.legend_title, title_fontsize=12, prop={'size': 12})
    else:
        fig.legend(lines, labels, loc='center', ncol=math.ceil(len(labels)/2), bbox_to_anchor=args.legend_bbox, prop={'size': args.legend_fontsize}, title=args.legend_title, title_fontsize=19)

    print('saving to', args.output_path)
    if args.tight:
        fig.tight_layout(rect=args.rect, pad=args.pad) # left, down, right, up
    fig.savefig(args.output_path, dpi=args.dpi)
    # figlegend.savefig(f'{args.output_path}.leg.png', dpi=args.dpi)
    
