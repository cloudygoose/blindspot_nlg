import argparse
import pickle
import os
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import math
from util import load_file_by_line
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load model and retrieve attention weights
device = 'cuda' if torch.cuda.is_available() else -1

def get_model_tok(model_version):
    # model_version = 'roberta-large'
    model = AutoModel.from_pretrained(model_version, output_attentions=True).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_version)
    return model, tokenizer

@torch.no_grad()
def get_sent_attn(model, tokenizer, sentence, max_length, i):
    input_ids = tokenizer.encode(sentence, return_tensors='pt', truncation=True, max_length=max_length).to(device)
    if input_ids.size(1) != max_length:
        print(f'Length {input_ids.size(1)} not long enough! requires {max_length}')
        return None, None
    # assert input_ids.size(1) == max_length # must achieve max_length

    
    out = model(input_ids)
    attention = out[-1]
    attention = tuple(at.detach().cpu() for at in attention)
    input_id_list = input_ids[0].tolist() # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list) 
    
    return attention, tokens

def get_sum_attn_distrib(attention, steps: int=1):
    '''
    Attention shape:
        `num_layer` tuple of shape `(bz, num_heads, seqlen, seqlen)`
    '''
    attn_avg = torch.cat(attention, dim=1).squeeze().mean(axis=0).to(device) # (seqlen, seqlen)
    seqlen = attn_avg.size(0)
    # could also use convolution, but it's a bit overshoot
    step_size = int(math.floor(seqlen / steps))
    attn_avg_reduced = np.empty((steps, steps))
    for idxi in range(steps):
        i = idxi * step_size
        for idxj in range(steps):
            j = idxj * step_size
            attn_block = attn_avg[i:i+step_size, j:j+step_size]
            attn_avg_reduced[idxi, idxj] = attn_block.sum().detach().cpu().numpy() / step_size # because the sum of step_size rows is step_size, not 1
    
    # if not attn_avg_reduced.shape == (steps, steps):
    #     breakpoint()
    return attn_avg_reduced


def get_output_path(model_str, input_path, steps, output_dir):
    name = os.path.splitext(input_path)[0] + f'_{model_str}_step{steps}'
    path = os.path.join(output_dir, name) # path w/o ext
    return path

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models', type=str, nargs='+')
    parser.add_argument('-i', '--input_text', type=str, required=True)
    parser.add_argument('-s', '--steps', type=int, default=100)
    parser.add_argument('-l', '--max_length', type=int, default=200, help='need to be a multiple of args.steps!!')
    parser.add_argument('-o', '--output_dir', type=str, default='./analysis_plots')
    parser.add_argument('--figsize', type=float, default=6.4)
    parser.add_argument('-f', '--force', action='store_true', help='do not use cache')
    parser.add_argument('--dpi', type=int, default=300)
    
    args = parser.parse_args()
    assert args.max_length % args.steps == 0
    texts = load_file_by_line(args.input_text)

    n_plots = len(args.models)
    fig, axs = plt.subplots(1, n_plots)
    fig.set_size_inches(n_plots*args.figsize, args.figsize)

    printed_y = False

    model_name_transform = {
        'gpt2-large': 'GPT2-large',
        'roberta-large': 'RoBERTa-large'
    }
    plt.set_cmap("rainbow")
    for ax, model_str in tqdm(zip(axs.flatten(), args.models), desc='models', total=n_plots):
        path = get_output_path(model_str, args.input_text, args.steps, args.output_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model, tokenizer = get_model_tok(model_str)
        cache_path = f'{path}.cache.pkl'
        if not args.force and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                attns = pickle.load(f)
            print('read cache...')
        else:            
            attns = [get_sent_attn(model, tokenizer, text, args.max_length, i)[0] for i, text in enumerate(tqdm(texts, desc='texts'))]
            attns = [attn for attn in attns if attn is not None]
        print(f'Filtered num examples: {len(attns)}')
        attn_distrib_list = [get_sum_attn_distrib(attention, steps=args.steps) for attention in tqdm(attns, desc='aggregate')]
        attn_d = np.stack(attn_distrib_list, axis=0).mean(axis=0) # (seqlen, seqlen)

        ax.set_title(model_name_transform.get(model_str, model_str), fontsize=18)
        ax.set_xticks([args.steps//5*t for t in range(6)], labels=['0%', '20%', '40%', '60%', '80%', '100%'])
        ax.set_yticks([args.steps//5*t for t in range(6)], labels=['0%', '20%', '40%', '60%', '80%', '100%'])
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        if not printed_y:
            ax.set_ylabel('Position: token attention from', size=18)
            printed_y = True
        ax.set_xlabel('Position: token attention to', size=18)
        im = ax.imshow(attn_d)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.9)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal') # , location='bottom' , fraction=0.046, pad=0.04
        cbar.ax.tick_params(labelsize=12)
    fig.tight_layout(pad=0.25)
    fig.subplots_adjust(wspace=-0.1)
    fig.savefig(f'{path}.all.png', dpi=args.dpi)
