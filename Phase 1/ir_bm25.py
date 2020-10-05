import re,os,glob,json
from nltk.tokenize import word_tokenize,sent_tokenize
from rank_bm25 import BM25L,BM25Plus,BM25Okapi
#import matplotlib.pyplot as plt

from num2words import num2words
from nlpre import replace_from_dictionary,token_replacement
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.stem import PorterStemmer
porter = PorterStemmer()

import string

def claim_clean_lema_noabbr(claim_list):
    results = []
    for claim in claim_list:
        new_claim = [word for word in word_tokenize(claim) if word.lower() not in stopwords.words('english')]
        new_claim = [wordnet_lemmatizer.lemmatize(word,pos="v") for word in new_claim]
        new_claim = [w.lower() if not w.isdigit() else num2words(w) for w in new_claim]
        new_claim = list(filter(lambda token: token not in string.punctuation, new_claim))
        results.append(new_claim)
    return results

def abstract_clean_lema(abst_list):
    new_abst_list = []
    for abst in abst_list:
        new_abst = [wordnet_lemmatizer.lemmatize(word,pos="v") for word in abst]
        new_abst_list.append(new_abst)
    return new_abst_list

def claim_clean_lema(claim_list):
    results = []
    for claim in claim_list:
        new_claim = [word for word in word_tokenize(claim) if word.lower() not in stopwords.words('english')]
        new_claim = [wordnet_lemmatizer.lemmatize(word,pos="v") for word in new_claim]
        new_claim = [abbr_dict.get(e,e) for e in new_claim]
        new_claim = [w.lower() if not w.isdigit() else num2words(w) for w in new_claim]
        new_claim = list(filter(lambda token: token not in string.punctuation, new_claim))
        results.append(new_claim)
    return results

def claim_clean_stem_noabbr(claim_list):
    results = [] 
    for claim in claim_list:
        new_claim = [word for word in word_tokenize(claim) if word.lower() not in stopwords.words('english')]

        new_claim = [porter.stem(word) for word in new_claim]
        
        new_claim = [w.lower() if not w.isdigit() else num2words(w) for w in new_claim]
        new_claim = list(filter(lambda token: token not in string.punctuation, new_claim))
        results.append(new_claim)      
    return results

def claim_clean_stem(claim_list):
    results = [] 
    for claim in claim_list:
        new_claim = [word for word in word_tokenize(claim) if word.lower() not in stopwords.words('english')]
        new_claim = [porter.stem(word) for word in new_claim]
        new_claim = [abbr_dict.get(e,e) for e in new_claim]
        new_claim = [w.lower() if not w.isdigit() else num2words(w) for w in new_claim]
        new_claim = list(filter(lambda token: token not in string.punctuation, new_claim))
        results.append(new_claim)      
    return results


def abstract_clean_stem(abst_list):
  new_abst_list = []
  for abst in abst_list:
    new_abst = [porter.stem(word) for word in abst]
    new_abst_list.append(new_abst)
  return new_abst_list

with open('abbr_dict.json','r',encoding='utf-8') as rf:
    abbr_dict = json.load(rf)

# With processing
with open('data_abst_200k_pure.json','r',encoding='utf-8') as rf:
    pmid_abst_dict = json.load(rf)

claim_dict = json.load(open('data_claim_5k_dict_all.json','r',encoding='utf-8'))
claim_list = list(claim_dict.keys())
abst_list = list(pmid_abst_dict.values())

with open('abst_token_pre_abbr.json','r',encoding='utf-8') as rf:
    abst_data = json.load(rf)

claim_token =  claim_clean_stem(claim_list)
abst_token = abstract_clean_stem(abst_data['abst_token'])
bmp25_sin = BM25Plus(abst_token)
abst_pro = [' '.join(w for w in abst) for abst in abst_token]

pmid_list = list(pmid_abst_dict.keys())
abst_pro_pmid_dict = dict(zip(abst_pro,pmid_list)) #search pmid by abst

pmid_list_claim = list(claim_dict.values())

#print(abst_token[:3])
count_bmp_all = [0 for i in range(5)]
for claim,pmid_set in zip(claim_token,pmid_list_claim):
    candidate_set = []
    print(claim)
    num_pmid = len(pmid_set)
    print(pmid_set)
    for abst in bmp25_sin.get_top_n(claim, abst_pro, n=500):
        candidate_set.append(abst_pro_pmid_dict[abst])
    #print(candidate_set)
    for i,index in zip([1,5,20,100,500],range(5)):
        count_pmid = 0
        for pmid in pmid_set:
            if pmid in candidate_set[:i]:
                count_pmid+=1
        count_bmp_all[index] += count_pmid/float(num_pmid)
print(count_bmp_all)

results = [s/5034 for s in count_bmp_all]
print(results)

save_dict = {'abst_token':abst_token}
json.dump(save_dict,open('abst_token_pre_abbr_stem.json','w',encoding='utf-8'))

with open('bm25_stem_abbr.txt','w',encoding='utf-8') as wf:
    for w in results:
        ss = str(w)+'\t'
        wf.write(ss)


