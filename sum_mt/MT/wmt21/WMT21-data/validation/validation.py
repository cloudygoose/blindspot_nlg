import pandas as pd
import sys
import argparse
import numpy as np
 
parser = argparse.ArgumentParser()
parser.add_argument("-n","--metricname")
parser.add_argument("-s", "--src",  help="include this if your metric is reference-free", default=False, action='store_true')
args = parser.parse_args() 

print(args)
metricname=args.metricname 
if args.src:
    metrictype='src'
else:
    metrictype='ref' 
 
print('Thanks for running the validation script')
cols={'sys':['metric', 'lp', 'testset','refid', 'sysid',  'score'],
      'seg':['metric', 'lp', 'testset','refid', 'sysid', 'segid',  'score'], 
     }
colformat={'sys':'<METRIC NAME>   <LANG-PAIR>   <TEST SET>    <REF SET>   <SYSTEM>   <SYSTEM LEVEL SCORE>',
      'seg':'<METRIC NAME>   <LANG-PAIR>   <TESTSET>    <REF SET>   <SYSTEM-ID>   <SEGMENT-ID>   SEGMENT SCORE>',
      }
      
for level in ['sys','seg' ]:
    print('==========================================')
    print('checking', level ,"level:")
      
    #read participant metric
    try:  
        mymetric = pd.read_csv(f'./{metricname}.{level}.score', sep='\t',header=None)    
    except FileNotFoundError:
        try:
             mymetric = pd.read_csv(f'./{metricname}.{level}.score.gz', sep='\t',header=None)    
        except FileNotFoundError:
             print(f"File not found: './{metricname}.{level}.score.gz' or './{metricname}.{level}.score'")
             continue
     
    #read demo metric   
    try:
        demo = pd.read_csv(f'./{metrictype}-metric.{level}.score', sep='\t', header=None)
    except FileNotFoundError:
        print(f"Please download the f'./{metrictype}-metric.{level}.score' from the Google Drive Folder")
        continue        


    #column names
    if len(mymetric.columns)!=len(cols[level]):
        print(f"Columns of './{metricname}.{level}.score' should be in format {colformat[level]}  ")
        continue
        
 
    demo.columns = cols[level]
    mymetric.columns = cols[level]   


    #check for NaNs
    if any(mymetric.score.isna()):
        print('Please check scores of segments for rows:')
        metric_na=mymetric[mymetric['score'].isna()] 
        for row in metric_na.iterrows():
            print(row) 
    #check for any errors converting scores to float
    try:
        subm_scores = [float(sc) for sc in mymetric.score]
    except Exception as e:
        print(f'Not all scores are floats; please doubble check. This is the error when converting all scores to float:',e) 
        print() 
 

    #note any missing language-pairs; these will be printed at the end
    missing_lps = []
    for lp, demo_lp in demo.groupby('lp'): 
        if lp not in mymetric.lp.unique():
            missing_lps.append(lp)
            continue

        #check each testset  
        for testset, demo_lp_test in demo_lp.groupby('testset'): 
            if testset not in mymetric[(mymetric.lp == lp)].testset.unique():  
                print(f'{lp}, {testset}: Metric scores missing for the testset {testset}. ')
                continue 

            #check each reference ID
            for refid, demo_lp_test_ref in demo_lp_test.groupby('refid'):  
                metriclp = mymetric[(mymetric.testset ==testset) &(mymetric.refid ==refid) &(mymetric.lp == lp)]
                
                if len(metriclp)==0: 
                    print(f'{lp}, {testset}, {refid}: Metric scores missing for the reference {refid}.' )
                    continue   
                    
                #check if all systems are scored for the current lp, testset and reference
                if set(demo_lp_test_ref.sysid.unique())-set(metriclp.sysid.unique())-set([refid]): 
                    if  not(  'Allegro.eu' in set(metriclp.sysid.unique()) and 'Allegro' in set(demo_lp_test_ref.sysid.unique())-set(metriclp.sysid.unique())):
                        print(f'{lp}, {testset}, {refid}: Metric scores missing for systems:', set(demo_lp_test_ref.sysid.unique())-set(metriclp.sysid.unique())-set([refid])) 

                #check for extra systems for the current lp, testset and reference
                if set(metriclp.sysid.unique())- set(demo_lp_test_ref.sysid.unique())- set([refid]): 
                    if 'Allegro.eu' not in set(metriclp.sysid.unique()):
                        print(f'{lp}, {testset}, {refid}: extra systems:',set(metriclp.sysid.unique())- set(demo_lp_test_ref.sysid.unique())- set([refid]))
                 
                 

                #checking for missing/extra segment level scores
                if level=='seg':
                    
                    merged_scores = pd.merge(demo_lp_test_ref[(demo_lp_test_ref.sysid.isin(metriclp.sysid.unique())) & (demo_lp_test_ref.refid.isin(metriclp.refid.unique()))], metriclp, how='left', on=['lp', 'testset','refid', 'sysid',  'segid'])
                    merged_scores = merged_scores[merged_scores.refid!=merged_scores.sysid]
                    mergedna =  merged_scores[merged_scores['score_y'].isna()] 
                    
                    #if there are few missing scores, then list them
                    if 1<=len(mergedna)<=10: 
                        print(f'{lp}, {testset}, {refid}: Metric scores missing for segments:') 
                        print('\t','sysid', 'segid')            
                        for _,row in mergedna.iterrows():
#                             print(row)
                            print('\t',row['sysid'],  row['segid']) 
                    
                     
                    elif len(mergedna)>10: 
                        missing_segs = []
                        for sys in demo_lp_test_ref.sysid.unique():
                            missing_indices=set(metriclp[metriclp.sysid==sys].segid.values) - set(demo_lp_test_ref[demo_lp_test_ref.sysid==sys].segid.values)
                            if missing_indices:
                                missing_segs.append(sys)
                                
                        nsegs=len(demo_lp_test_ref[demo_lp_test_ref.sysid==sys].segid.values)
                        #missing segment ids for all systems
                        if len(missing_segs) == len(demo_lp_test_ref.sysid.unique()):
                            print(f'{lp}, {testset}, {refid}: there should be {nsegs} segment ids')
                        #missing segment ids for some systems
                        elif missing_segs:
                            print(f'{lp}, {testset}, {refid}:  there should be {nsegs} segment ids for systems {missing_segs}')   

                     
                    #check for extra segmend ids
                    too_many_segs = []
                    for sys in demo_lp_test_ref.sysid.unique():
                        if set(metriclp[metriclp.sysid==sys].segid.values) - set(demo_lp_test_ref[demo_lp_test_ref.sysid==sys].segid.values):
                            too_many_segs.append(sys)
                    nsegs=len(demo_lp_test_ref[demo_lp_test_ref.sysid==sys].segid.values)                             
                    if len(too_many_segs) == len(demo_lp_test_ref.sysid.unique()):
                            print(f'{lp}, {testset}, {refid}:  there should be {nsegs} segment ids')   
                    elif too_many_segs:
                        print(f'{lp}, {testset}, {refid}:  there should be {nsegs} segment ids for systems {too_many_segs} ')
                    
        
    
    #checking for extra values in the testset and refid columns
    for lp in demo.lp.unique():
        if set(mymetric[mymetric.lp == lp].testset.unique()) - set(demo[demo.lp == lp].testset.unique()):
            print('extra testsets for lp' + lp + ':', set(mymetric.testset.unique()) - set(demo.testset.unique()) )
#         for df in 
        if set(mymetric[mymetric.lp == lp].refid.unique()) - set(demo[demo.lp == lp].refid.unique()):
            print('extra references for '+lp + " " +testset+":", set(mymetric.testset.unique()) - set(demo.testset.unique()) )


    #print missing lps
    if missing_lps:
        print('Note that the metric is missing scores for the following language-pairs:', missing_lps)        
        print('...if this is intentional please ignore this message')
print('==========================================')
print('Done!')
print('If you cant find the source of any errors that this script flags, please email nitika.mathur@gmail.com so we can work with you to solve this')
