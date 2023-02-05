# import analysis tools
from analysis import SUMStat, D2TStat, WMTStat

def truncate_print(l, n=10):
    """ Print the first n items of a list"""
    for i, x in enumerate(l):
        if i == n:
            print('...')
            break
        print(x)

#ld_fn = 'SUM/REALSumm/scores.pkl'
'SUM/REALSumm/final_p.pkl'
print('loading pkl', ld_fn)
summ_stat = SUMStat(ld_fn)
#summ_stat = SUMStat('SUM/SummEval/final_p.pkl')

#print('[All metrics]')
#print(summ_stat.metrics) # change to print if you want to see all metrics
print('[Automatic metrics]')
print(summ_stat.auto_metrics)
print('[Human metrics]')
print(summ_stat.human_metrics)

valid_metrics = [
    'bart_score_cnn_src_hypo',
]

# The first argument is the human metric considered.
# The second argument is a list of considered automatic metrics, can omit it if all automatic metrics are considered
#summ_stat.evaluate_summary('coherence', valid_metrics) 
#summ_stat.evaluate_summary('litepyramid_recall', valid_metrics)

summ_stat.get_avg_score('bart_score_cnn_src_hypo')
