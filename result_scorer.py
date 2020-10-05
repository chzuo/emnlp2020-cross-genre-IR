import json
import sys

def read_results(filename):
    claim_dict = dict()
    with open(filename,'r',encoding='utf-8') as rf:
        for line in rf.readlines():
            claim = line.split('\t')[0]+'\n'
            claim_dict.setdefault(claim,{'pmid':[],'score':[]})
            claim_dict[claim]['pmid'].append(line.split('\t')[1])
            claim_dict[claim]['score'].append(float(line[:-1].split('\t')[2]))
    return claim_dict

def format_score(score):
    score_str = '{0:.3f}'.format(score)
    return '{:<11}'.format(score_str)

def score_calculate(mode,result_file):

    if mode == 'dev':
        #pc_oc_dict = json.load(open('pc_oc.json','r',encoding='utf-8'))
        evl_dict = json.load(open('dev_data_claim_pmid_dict.json','r',encoding='utf-8'))
    else:
        evl_dict = json.load(open('test_data_claim_pmid_dict.json','r',encoding='utf-8'))
    result_dict = read_results(result_file)
    result = [ 0 for i in range(8)]
    
    for claim in result_dict:
        #print(claim)
        rank_results = [x for _, x in sorted(zip(result_dict[claim]['score'],result_dict[claim]['pmid']), key=lambda pair: pair[0],reverse=True)]
        #claim = pc_oc_dict[claim]
        print(claim)
        if '""' in claim:
            claim = claim.replace('""','"')
            claim = claim[1:-2]+'\n'
        pmid_set = evl_dict[claim]
        num_pmid = len(pmid_set)
        for i,index in zip([1,3,5,10,20,50,100,200],range(8)):
            count_pmid = 0
            for pmid in pmid_set:
                if pmid in rank_results[:i]:
                    count_pmid+=1
            result[index] += count_pmid/float(num_pmid)

    result_score = [s/float(len(evl_dict)) for s in result]
    
    
    MAIN_THRESHOLDS = [1,3,5,10,20,50,100,200]
    metric_names = ['R@' + str(threshold) for threshold in MAIN_THRESHOLDS]
    metrics_header_items = ['{:<11}'.format(metric) for metric in metric_names]
    print('{:<26} '.format('') + ' '.join(metrics_header_items))

    formatted_scores = [format_score(score) for score in result_score]
    print('  {:<25}'.format('Score') + ' '.join(formatted_scores))

if __name__ == '__main__':
    try:
        mode, filename = sys.argv[1:3]
        score_calculate(mode,filename)
    except Exception as e:
        print(sys.argv)
        print(e)
