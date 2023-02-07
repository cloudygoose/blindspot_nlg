import argparse
import os
from mauve import compute_mauve
import matplotlib.pyplot as plt 
from util import load_file_by_line

'''partially adapted from nl-command'''

def eval_mauve(ref, gen, **kwargs):
    if ref is None:
        return -1
    gen = [text.replace('[BOS]', '').replace('<|endoftext|>', '') for text in gen] # get rid of BOS, EOS
    gen = [text for text in gen if len(text) > 0]
    print(f'number of mauve generations: {len(gen)}')
    if len(ref) < len(gen): print('WARNING: MAUVE #reference < #generated! They should be the same!')
    if len(ref) > len(gen):
        print('MAUVE #reference > #generated, truncating reference to have length #generated')
        ref = ref[:len(gen)]
    out = compute_mauve(p_text=ref, q_text=gen, device_id=0, max_text_length=512, verbose=False, **kwargs)
    print(f'MAUVE={out.mauve}')
    return out

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference', type=str, required=True, help='reference for mauve, should be SAME LENGTH as generation')
    parser.add_argument('-g', '--generation', type=str, nargs='+', required=True)
    parser.add_argument('-d', '--save_dir', type=str, default=None, help='save dir, default to dir of last generation file')
    parser.add_argument('-s', '--output_suffix', type=str, default='', help='output name mauve_{suffix}.csv and div_{suffix}.png')
    parser.add_argument('-f', '--feature_extractor', type=str, default=None, help='feature extractor for mauve. default is gpt2-large')
    parser.add_argument('-p', '--use_pooler_output', action='store_true')
    parser.add_argument('-fc', '--force', action='store_true', help='force output, override existing results if necessary')
    parser.add_argument('--long_name', action='store_true', help='prepend dir name')

    args = parser.parse_args()
    if args.feature_extractor is not None and args.output_suffix == '':
        print('Using non-default feature extractor, need to provide a --output_suffix!!')
        raise AssertionError
    if args.feature_extractor is None:
        args.feature_extractor = 'gpt2-large'

    return args

if __name__ == '__main__':
    args = parse_args()
    save_dir = os.path.dirname(args.generation[-1]) if args.save_dir is None else args.save_dir # use directory of last generation file by default
    os.makedirs(save_dir, exist_ok=True)
    if len(args.output_suffix) > 0: args.output_suffix = '_' + args.output_suffix
    csv_save_path = os.path.join(save_dir, f'mauve{args.output_suffix}.csv')
    png_save_path = os.path.join(save_dir, f'div{args.output_suffix}.png')
    if not args.force and (os.path.exists(csv_save_path) or os.path.exists(png_save_path)):
        print(f'Already have the same filenames {csv_save_path} and {png_save_path}!! stopping...')
        raise AssertionError

    ref = load_file_by_line(args.reference)
    plt.figure(dpi=160)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.autoscale(enable=True, tight=True)
    plt.grid()

    results = 'name\tmauve\tH(pq)\tH(p)\tH(qp)\tH(q)\n'
    kwargs = {'featurize_model_name': args.feature_extractor, 'use_pooler_output': args.use_pooler_output}
    for gen_path in args.generation:
        if not os.path.exists(gen_path):
            print(f'WARNING: path {gen_path} does not exist, skipping...')
            continue
        if os.path.splitext(gen_path)[1] != '.txt':
            print(f'WARNING: {gen_path} is not a text file, skipping...')
            continue
        basename = os.path.splitext(os.path.basename(gen_path))[0]
        if args.long_name:
            basename = os.path.basename(os.path.dirname(gen_path)) + basename
        print(f'file basename: {basename}')
        gen = load_file_by_line(gen_path)
        out = eval_mauve(ref, gen, **kwargs)
        results += f'{basename}\t{out.mauve}\t{out.ents[0]}\t{out.ents[1]}\t{out.ents[2]}\t{out.ents[3]}\n'
        plt.plot(out.divergence_curve[:, 1], out.divergence_curve[:, 0], label=basename)

    print(results)
    with open(csv_save_path, 'w') as f:
        print(results, file=f)
    plt.legend()
    plt.savefig(png_save_path)
