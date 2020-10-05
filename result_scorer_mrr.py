import json
import sys


def read_results(filename):
    claim_dict = dict()
    with open(filename, 'r', encoding='utf-8') as rf:
        for line in rf.readlines():
            claim = line.split('\t')[0] + '\n'
            claim_dict.setdefault(claim, {'pmid': [], 'score': []})
            claim_dict[claim]['pmid'].append(line.split('\t')[1])
            claim_dict[claim]['score'].append(float(line[:-1].split('\t')[2]))
    return claim_dict


def format_score(score):
    score_str = '{0:.3f}'.format(score)
    return '{:<11}'.format(score_str)


def score_calculate(mode, result_file):
    if mode == 'dev':
        evl_dict = json.load(open('dev_data_claim_pmid_dict.json', 'r', encoding='utf-8'))
    else:
        evl_dict = json.load(open('test_data_claim_pmid_dict.json', 'r', encoding='utf-8'))
    result_dict = read_results(result_file)
    result = 0

    for claim in result_dict:
        rank_results = [x for _, x in
                        sorted(zip(result_dict[claim]['score'], result_dict[claim]['pmid']), key=lambda pair: pair[0],
                               reverse=True)]
        if '""' in claim:
            claim = claim.replace('""', '"')
            claim = claim[1:-2] + '\n'
        pmid_set = evl_dict[claim]
        num_pmid = len(pmid_set)
        count_pmid = []
        for pmid in pmid_set:
            for i in range(1,500):
                if pmid in rank_results[:i]:
                    count_pmid.append(i)
                    continue
        #print(min(count_pmid))
        if len(count_pmid)==1:
            result += 1/float(count_pmid[0])
        elif len(count_pmid)>1:
            result += 1/float(min(count_pmid))

    print(result)
    mrr = result/len(evl_dict)
    print(mrr)


if __name__ == '__main__':
    try:
        mode, filename = sys.argv[1:3]
        score_calculate(mode, filename)
    except Exception as e:
        print(sys.argv)
        print(e)
