def postprocess_text(text, truncate_at_period=False, postprocess_special=False):
    '''
    postprocess generated string:
    1. replace \n with \\n
    2. truncate to longest full sentence (last '.')
    '''
    replaced_text = text.replace('\n', '\\n')
    res = replaced_text
    if truncate_at_period:
        last_period_pos = -100
        while True:
            last_period_pos = res.rfind('.')
            if last_period_pos == -1:
                last_period_pos = len(replaced_text)-1
                break
            if last_period_pos > 0 and last_period_pos < len(res)-1 and res[last_period_pos-1] == '@' and res[last_period_pos+1] == '@':
                # looking for @.@, which is not a period
                res = res[:last_period_pos]
            else:
                break
        res = replaced_text[:last_period_pos+1]
    if postprocess_special:
        # trim left space
        for t in ['.', ',', '?', '!', ':', ';', ')', "'s", '%']:
            res = res.replace(f' {t}', t)
        # trim right space
        for t in ['(']:
            res = res.replace(f'{t} ', t)
        # @.@, @-@
        for t in ['.', ',', '-']:
            res = res.replace(f' @{t}@ ', t)
        # heuristic for handling "
        res_segs = res.split('"')
        for i in range(1, len(res_segs), 2): # only modify 1, 3, 5, ... because they are "in" quote
            res_segs[i] = res_segs[i].strip()
        res = '"'.join(res_segs)
        res = res.strip()
    return res