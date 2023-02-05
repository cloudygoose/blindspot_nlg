#for me in bert_score_f bert_score_p bert_score_r mover_score rouge2-f rougeL-f comet cometqe bart_score_para_ref_hypo bart_score_para_hypo_ref bart_score_para_src_hypo bart_score_para_avg_f  
#do
#    python plot.py score_saves/sum/$me/ --error_bar --max_edr 0.6 #--prefix con- --name_suffix _con
    #python plot.py score_saves/sum/$me/ --error_bar --max_edr 0.6 --select truncate,negate,bertdiverge,preposition,switchsent,removearticle,lemmatizeverb,replacesent --name_suffix _select
#done

#for me in unieval_coherence unieval_consistency unieval_fluency unieval_overall unieval_relevance
#do
#    python plot.py score_saves/sum/$me/ --error_bar --max_edr 0.6 --prefix flu- --name_suffix _flu #--prefix con- --name_suffix _con
#    python plot.py score_saves/sum/$me/ --error_bar --max_edr 0.6 --prefix con- --name_suffix _con
    #python plot.py score_saves/sum/$me/ --error_bar --max_edr 0.6 --select truncate,negate,bertdiverge,preposition,switchsent,removearticle,lemmatizeverb,replacesent --name_suffix _select
#done


TN=ted-zh-en
#TN=wmt
for me in bert_score mover_score bleu comet cometqe prism prismqe bart_score_cnn bart_score_para bleurt
do
    python plot.py score_saves/$TN/$me/ --max_edr 0.65 --error_bar --prefix flu- --name_suffix _flu    
    python plot.py score_saves/$TN/$me/ --max_edr 0.65 --error_bar --prefix con- --name_suffix _con    
done


#python plot.py score_saves/sum/bert_score_f/
#python plot.py score_saves/sum/mover_score/
#python plot.py score_saves/sum/rouge2-f/

