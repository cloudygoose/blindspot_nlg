import re
import logging, random, math, os
import json
logger = logging.getLogger()

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def save_res(res_d, path_r):
    for me in res_d:
        path = path_r + '/' + me + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        save_d = {}
        for idx in res_d[me]:
            if idx == 'ref':
                tt = idx
            else:
                tt = idx.split('_')[1]
                if isfloat(tt.split('-')[-1]):
                    tt = tt[:-(len(tt.split('-')[-1])+1)]
            if not tt in save_d: save_d[tt] = {}
            save_d[tt][idx] = res_d[me][idx]
            for kk in save_d[tt][idx]:
                if isfloat(save_d[tt][idx][kk]):
                    save_d[tt][idx][kk] = float(save_d[tt][idx][kk]) #float32 is not JSON serializable
        
        for tt in save_d:
            fn = path + tt + '.json'
            logger.info('saving to %s', fn)
            with open(fn, 'w') as f:
                json.dump(save_d[tt], f)

#for load:
#f = open('mydata.json')
#team = json.load(f)