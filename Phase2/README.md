Step 1: Train a new model: `python train.py ../data/train_dataset.csv emilyalsentzer/Bio_ClinicalBERT`

- The first argument is the training data file and the second argument is the base pre-trained model to be used for training. List of models available is at the end of this document.
- We have modeled the training task as a simple paired-sequence classification task.
- Model trains over 1 epoch, with parameters for learning rate of 5e-5 and maximum sequence length of 256 tokens.
- The trained model is then saved under /cache/fine_tuned/. The trained model can then be used for prediction for dev and test data.

Step 2: Use trained model for predictions: `python predict.py ../data/dev_dataset.csv 'cache/fine_tuned/`

- The first argument is the dev/test data file and the second argument is the folder location of the trained model to use.
- The output of the prediction probabilities is stored under reports/. 

Step 3: Evaluate: `python evaluation_scorer/result_scorer.py ../reports/dev_predictions.tsv`

List of Models Supported:


XLNET:
---

"xlnet-base-cased"
"xlnet-large-cased"

BERT:
---

"bert-base-uncased"
"bert-large-uncased"
"bert-base-cased"
"bert-large-cased"
"bert-base-multilingual-uncased"
"bert-base-multilingual-cased"
"bert-base-chinese"
"bert-base-german-cased"
"bert-large-uncased-whole-word-masking"
"bert-large-cased-whole-word-masking"
"bert-large-uncased-whole-word-masking-finetuned-squad"
"bert-large-cased-whole-word-masking-finetuned-squad"
"bert-base-cased-finetuned-mrpc"
"bert-base-german-dbmdz-cased"
"bert-base-german-dbmdz-uncased"
"bert-base-japanese"
"bert-base-japanese-whole-word-masking"
"bert-base-japanese-char"
"bert-base-japanese-char-whole-word-masking"
"bert-base-finnish-cased-v1"
"bert-base-finnish-uncased-v1"
"bert-base-dutch-cased"

DISTILL-BERT:
---

"distilbert-base-uncased"
"distilbert-base-uncased-distilled-squad"
"distilbert-base-cased"
"distilbert-base-cased-distilled-squad"
"distilbert-base-german-cased"
"distilbert-base-multilingual-cased"
"distilbert-base-uncased-finetuned-sst-2-english"

ROBERTA:
---

"roberta-base"
"roberta-large"
"roberta-large-mnli"
"distilroberta-base"
"roberta-base-openai-detector"
"roberta-large-openai-detector"

BIO-BERT:
---

"emilyalsentzer/Bio_ClinicalBERT"