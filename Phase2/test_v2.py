import os
import sys
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
import torch
from sequence_classification import Processor, SequenceClassifier, BIO_BERT_MODEL_KEY
from pytorch_utils import dataloader_from_dataset
from timer import Timer
from nltk import sent_tokenize

if __name__ == '__main__':
    print(SequenceClassifier.list_supported_models())

    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    MODEL_NAME = BIO_BERT_MODEL_KEY # "bert-base-cased"
    TO_LOWER = False
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
    dev_file = sys.argv[2]
    dataset_file = sys.argv[1]

    DATASET_PATH_PREFIX = ''
    EXTENSION = '.csv'

    REPORTS_DIR = 'reports/'
    if not os.path.exists(REPORTS_DIR):
        os.mkdir(REPORTS_DIR)

    MODELS_DIR = 'models/'
    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)

    # top_sentences = None

    # def get_sentences(abstract, top_sentences):
    #   sentences = sent_tokenize(abstract)

    #   if top_sentences > len(sentences):
    #     return abstract

    #   return ' '.join(sentences[:top_sentences])

    train = pd.read_csv(DATASET_PATH_PREFIX + train_file + EXTENSION)
    train_df = train.drop(columns=['PMID'], errors='ignore')
    train_df.columns = ['sentence1', 'sentence2', 'gold_label']

    # if top_sentences:
    #     train['sentence1'] = train['sentence1'].apply(lambda abstract: get_sentences(abstract, top_sentences))

    # train_samples = int(0.8 * len(train))
    # train_df = train.head(train_samples)
    dev = pd.read_csv(DATASET_PATH_PREFIX + dev_file + EXTENSION)
    dev_df = dev.drop(columns=['PMID'], errors='ignore')
    dev_df.columns = ['sentence1', 'sentence2', 'gold_label']

    print("Training dataset size: {}".format(train_df.shape[0]))
    print("Development dataset size: {}".format(dev_df.shape[0]))
    print()
    # exit()

    processor = Processor(model_name=MODEL_NAME, to_lower=TO_LOWER)

    train_dataset = processor.dataset_from_dataframe(
        df=train_df,
        text_col=TEXT_COL_1,
        label_col=LABEL_COL_NUM,
        text2_col=TEXT_COL_2,
        max_len=MAX_SEQ_LENGTH,
    )
    dev_dataset = processor.dataset_from_dataframe(
        df=dev_df,
        text_col=TEXT_COL_1,
        text2_col=TEXT_COL_2,
        max_len=MAX_SEQ_LENGTH,
    )

    train_dataloader = dataloader_from_dataset(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    dev_dataloader = dataloader_from_dataset(
        dev_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    classifier = SequenceClassifier(model_name=MODEL_NAME, num_labels=2)

    with Timer() as t:
        classifier.fit(
                train_dataloader,
                num_epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE,
                warmup_steps=WARMUP_STEPS,
                seed=SEED
            )
    print("Training time : {:.3f} hrs".format(t.interval / 3600))

    with Timer() as t:
        # Use commented line below if you want Softmax probability of entailment 1 instead of simple classification
        predictions = classifier.predict(dev_dataloader, return_probabilities=True)
        # predictions = classifier.predict(dev_dataloader)
    print("Prediction time : {:.3f} hrs".format(t.interval / 3600))
    print(predictions)
    classifier.save_model(MODELS_DIR + dataset_file + '_top' + str(top_sentences))

    dev_df['predicted'] = np.array(predictions)
    dev_df['pmid'] = dev['PMID']
    dev_df.to_csv(REPORTS_DIR + dataset_file + '_test_pred_top' + str(top_sentences) + EXTENSION, index=False)

    # report = classification_report(dev_df[LABEL_COL_NUM], predictions, digits=3, output_dict=True)
    # df = pd.DataFrame(report).transpose()
    # df.to_csv(REPORTS_DIR + dataset_file + '_top' + str(top_sentences) + EXTENSION, index=False)
    # print(report)


