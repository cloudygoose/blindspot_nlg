from bart_score import BARTScorer
bart_scorer = BARTScorer(device='cpu', checkpoint='facebook/bart-large-cnn')

out = bart_scorer.score(['This is interesting.'], ['This is fun.'], batch_size=4)
print(out)

for rep in [2,3,4,5,6,7]:
    ww_now = 'This is fun. ' * rep
    print(ww_now)
    out = bart_scorer.score(['This is interesting.'], [ww_now], batch_size=4)
    print(out)
