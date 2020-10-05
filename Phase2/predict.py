import sys
import os
import pandas as pd
import numpy as np

from transformers import BertForSequenceClassification
from sequence_classification import Processor, SequenceClassifier, BIO_BERT_MODEL_KEY
from pytorch_utils import dataloader_from_dataset
from timer import Timer

if __name__ == '__main__':
    MODEL_NAME = sys.argv[3] # BIO_BERT_MODEL_KEY  # "bert-base-cased"
    # model configurations
    MAX_SEQ_LENGTH = 256
    TO_LOWER = True
    BATCH_SIZE = 24

    # data configurations
    TEXT_COL_1 = "sentence1"
    TEXT_COL_2 = "sentence2"
    LABEL_COL_NUM = "gold_label"

    dev_file = sys.argv[1]
    MODEL_LOCATION = sys.argv[2]

    REPORTS_DIR = 'reports/' + MODEL_LOCATION
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR, exist_ok=True)

    CACHE_DIR = 'cache/'
    if not os.path.exists(CACHE_DIR):
        os.mkdir(CACHE_DIR)

    dev = pd.read_csv(dev_file)
    dev_df = dev.drop(columns=['PMID'], errors='ignore')
    dev_df.columns = ['sentence1', 'sentence2', 'gold_label']
    print("Development dataset size: {}".format(dev_df.shape[0]))

    processor = Processor(model_name=MODEL_NAME, to_lower=TO_LOWER, cache_dir=CACHE_DIR)
    dev_dataset = processor.dataset_from_dataframe(
        df=dev_df,
        text_col=TEXT_COL_1,
        text2_col=TEXT_COL_2,
        max_len=MAX_SEQ_LENGTH,
    )

    dev_dataloader = dataloader_from_dataset(
        dev_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    classifier = SequenceClassifier(model_name=MODEL_NAME, num_labels=2, cache_dir=CACHE_DIR,
                                    load_model_from_dir=MODEL_LOCATION)

    with Timer() as t:
        # Use line below if you want Softmax probability of entailment 1 instead of simple classification
        predictions = classifier.predict(dev_dataloader, return_probabilities=True)
        # predictions = classifier.predict(dev_dataloader)
    print("Prediction time : {:.3f} hrs".format(t.interval / 3600))

    results = pd.DataFrame(columns=['claim', 'pmid', 'score'])
    results['claim'] = dev['Hypothesis'].str.strip()
    results['pmid'] = dev['PMID']
    results['score'] = np.array(predictions)

    results.to_csv(REPORTS_DIR + dev_file.split('/')[-1] + '_predictions.tsv', sep='\t', index=False,
                   header=False)
