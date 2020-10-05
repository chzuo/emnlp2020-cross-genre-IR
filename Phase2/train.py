import os
import sys
import pandas as pd
import numpy as np

import torch
from sequence_classification import Processor, SequenceClassifier, BIO_BERT_MODEL_KEY
from pytorch_utils import dataloader_from_dataset
from timer import Timer

if __name__ == '__main__':
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    MODEL_NAME = sys.argv[2]  # BIO_BERT_MODEL_KEY # "bert-base-cased"
    TO_LOWER = True
    BATCH_SIZE = 24

    TRAIN_DATA_USED_FRACTION = 1
    DEV_DATA_USED_FRACTION = 1
    NUM_EPOCHS = 1
    WARMUP_STEPS = 2500

    if not torch.cuda.is_available():
      print("NO GPU!")
      BATCH_SIZE = BATCH_SIZE // 2
    else:
      torch.cuda.manual_seed(SEED)
      print("YAY GPU!")

    # model configurations
    MAX_SEQ_LENGTH = 256

    # optimizer configurations
    LEARNING_RATE= 5e-5

    # data configurations
    TEXT_COL_1 = "sentence1"
    TEXT_COL_2 = "sentence2"
    LABEL_COL_NUM = "gold_label"

    train_file = sys.argv[1]

    CACHE_DIR = 'cache/'
    if not os.path.exists(CACHE_DIR):
        os.mkdir(CACHE_DIR)

    train = pd.read_csv(train_file)
    train_df = train.drop(columns=['PMID'], errors='ignore')
    train_df.columns = ['sentence1', 'sentence2', 'gold_label']

    print("Training dataset size: {}".format(train_df.shape[0]))
    print()

    processor = Processor(model_name=MODEL_NAME, to_lower=TO_LOWER, cache_dir=CACHE_DIR)

    train_dataset = processor.dataset_from_dataframe(
        df=train_df,
        text_col=TEXT_COL_1,
        label_col=LABEL_COL_NUM,
        text2_col=TEXT_COL_2,
        max_len=MAX_SEQ_LENGTH,
    )

    train_dataloader = dataloader_from_dataset(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    classifier = SequenceClassifier(model_name=MODEL_NAME, num_labels=2, cache_dir=CACHE_DIR)

    with Timer() as t:
        classifier.fit(
                train_dataloader,
                num_epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE,
                warmup_steps=WARMUP_STEPS,
                seed=SEED
            )

    print("Training time : {:.3f} hrs".format(t.interval / 3600))
    classifier.save_model(directory=train_file.split('/')[-1].split('.')[0] + '_' + MODEL_NAME.replace('/', '_'))
    print("Model saved!")
