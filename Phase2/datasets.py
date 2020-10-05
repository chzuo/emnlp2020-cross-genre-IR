# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# The below file is a modified version of the file sourced from:
# https://github.com/microsoft/nlp-recipes/blob/21a6e09cea2c106ca67fb32c69bc133290f3bee9/utils_nlp/models/transformers/datasets.py

import torch
from torch.utils.data import Dataset

class SCDataSet(Dataset):
    """Dataset for single sequence classification tasks"""

    def __init__(self, df, text_col, label_col, transform, **transform_args):
        self.df = df
        cols = list(df.columns)
        self.transform = transform
        self.transform_args = transform_args

        if isinstance(text_col, int):
            self.text_col = text_col
        elif isinstance(text_col, str):
            self.text_col = cols.index(text_col)
        else:
            raise TypeError("text_col must be of type int or str")

        if label_col is None:
            self.label_col = None
        elif isinstance(label_col, int):
            self.label_col = label_col
        elif isinstance(label_col, str):
            self.label_col = cols.index(label_col)
        else:
            raise TypeError("label_col must be of type int or str")

    def __getitem__(self, idx):
        input_ids, attention_mask, token_type_ids = self.transform(
            self.df.iloc[idx, self.text_col], **self.transform_args
        )
        if self.label_col is None:
            return tuple(
                [
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(attention_mask, dtype=torch.long),
                    torch.tensor(token_type_ids, dtype=torch.long),
                ]
            )
        labels = self.df.iloc[idx, self.label_col]
        return tuple(
            [
                torch.tensor(input_ids, dtype=torch.long),  # input_ids
                torch.tensor(attention_mask, dtype=torch.long),  # attention_mask
                torch.tensor(token_type_ids, dtype=torch.long),  # segment ids
                torch.tensor(labels, dtype=torch.long),  # labels
            ]
        )

    def __len__(self):
        return self.df.shape[0]


class SPCDataSet(Dataset):
    """Dataset for sequence pair classification tasks"""

    def __init__(
        self, df, text1_col, text2_col, label_col, transform, **transform_args
    ):
        self.df = df
        cols = list(df.columns)
        self.transform = transform
        self.transform_args = transform_args

        if isinstance(text1_col, int):
            self.text1_col = text1_col
        elif isinstance(text1_col, str):
            self.text1_col = cols.index(text1_col)
        else:
            raise TypeError("text1_col must be of type int or str")

        if isinstance(text2_col, int):
            self.text2_col = text2_col
        elif isinstance(text2_col, str):
            self.text2_col = cols.index(text2_col)
        else:
            raise TypeError("text2_col must be of type int or str")

        if label_col is None:
            self.label_col = None
        elif isinstance(label_col, int):
            self.label_col = label_col
        elif isinstance(label_col, str):
            self.label_col = cols.index(label_col)
        else:
            raise TypeError("label_col must be of type int or str")

    def __getitem__(self, idx):
        input_ids, attention_mask, token_type_ids = self.transform(
            self.df.iloc[idx, self.text1_col],
            self.df.iloc[idx, self.text2_col],
            **self.transform_args,
        )

        if self.label_col is None:
            return tuple(
                [
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(attention_mask, dtype=torch.long),
                    torch.tensor(token_type_ids, dtype=torch.long),
                ]
            )

        labels = self.df.iloc[idx, self.label_col]
        return tuple(
            [
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(attention_mask, dtype=torch.long),
                torch.tensor(token_type_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            ]
        )

    def __len__(self):
        return self.df.shape[0]
