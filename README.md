# # Querying Across Genres for Medical Claims in News
## Accepted at EMNLP 2020

This repository contains the accompanying code for the [paper](), where we present a query-based biomedical information retrieval task across two vastly different genres – newswire and research literature – where the goal is to find the research publication that supports the primary claim madein a health-related news article. For this task,we present a new dataset of 5,034 claims fromnews paired with research abstracts.

### Dataset

The dataset for the Cross Genre IR Task along with other supporting files can be accessed [here](https://drive.google.com/drive/folders/1PFfwaBehlQP6T-q6QwJYVtj7RACVzwtL?usp=sharing).

### Instructions on Usage.

Our approach consists of two steps:  
1. Selecting the most relevant candidates from a collection of 222k research abstracts, and  
2. Re-rankingthis the candidates list.

- For code related to step 1, see [Phase1](Phase1/)
- For code related to step 2, see [Phase2](Phase2/)

### Citation

If you use this work please cite our research paper:

```
@inproceedings{Zuo2020QueryingAG,
  title={Querying across Genres to Retrieve Research That Supports Medical Claims Made in News},
  author={Chaoyuan Zuo and Narayan Acharya and Ritwik Banerjee},
  booktitle={EMNLP},
  year={2020}
}
```
