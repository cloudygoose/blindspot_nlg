import sys, os
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lan', type=str, required=True, help='The language pair')
parser.add_argument('--output', type=str, default = None, help='output file path')

args = parser.parse_args()

s_lan, t_lan = args.lan.split('-')
print(f's_lan: {s_lan} t_len: {t_lan}')

s_file = f'WMT21-data/sources/newstest2021.{args.lan}.src.{s_lan}'
print(f'loading source from {s_file}')
s_lis = [ss.strip() for ss in open(s_file, 'r').readlines()]

res_d = {'src': s_lis}
sys_dir = f'WMT21-data/system-outputs/newstest2021/{args.lan}/'
for fn in os.listdir(sys_dir):
    ld_fn = os.path.join(sys_dir, fn)
    print(ld_fn)
    sys_name = ld_fn.split('.')[-2]
    print(sys_name)
    res_d[sys_name] = [ss.strip() for ss in open(ld_fn, 'r').readlines()]
    assert(len(res_d[sys_name]) == len(s_lis))

if args.output is not None:
    print(f'saving to {args.output}')
    torch.save(res_d, args.output)

breakpoint()

#t_file_a = f'WMT21-data/references/newstest2021.{args.lan}.ref.ref-A.{t_lan}'
#t_file_b = f'WMT21-data/references/newstest2021.{args.lan}.ref.ref-B.{t_lan}'

""" #this code was for loading only the references
print(f'loading ref_A from {t_file_a}')
refa_lis = [ss.strip() for ss in open(t_file_a, 'r').readlines()]
print(f'loading ref_B from {t_file_b}')
refb_lis = [ss.strip() for ss in open(t_file_b, 'r').readlines()]

assert(len(s_lis) == len(refa_lis) and len(s_lis) == len(refb_lis))
print(f'{len(s_lis)} samples loaded')

res_d = {}
for idx, (s_line, refa_line, refb_line) in enumerate(zip(s_lis, refa_lis, refb_lis)):
    #print(idx, s_line, refa_line, refb_line)
    res_d[idx] = {'src': s_line, 'ref': refb_line, 'better': {'sys': refa_line}}
"""

