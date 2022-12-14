# -*- coding: utf-8 -*-
"""labeling_functions.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XuBvNtWDSrKGREmzAu0kPeb19X9pO5yF
"""

from cProfile import label
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Tuple
from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
import os
from snorkel.preprocess import preprocessor
from textblob import TextBlob

file = open(
    "/Users/james.morrissey/PoetryRepos/personal-project/solvvy/keyphrases.json"
)
keywords: Dict = json.load(file)

data = pd.read_csv(
    "/Users/james.morrissey/PoetryRepos/personal-project/solvvy/data/single_query_in_app_solvvy_sessions_20220314-20220922.csv"
)
# data = data.iloc[0:200]
data.columns = [column.lower() for column in data.columns]
data = data.rename(columns={"description": "text"})

ABSTAIN = -1
QUESTION = 0
SUPPORT = 1
HAS_VERB = 2

from snorkel.preprocess.nlp import SpacyPreprocessor

spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)


@labeling_function(pre=[spacy])
def contains_question_word(row, question_words=keywords["question_word"]):
    for token in row.text:
        if token in question_words:
            return QUESTION
    return ABSTAIN


from snorkel.labeling.lf.nlp import nlp_labeling_function


@labeling_function(pre=[spacy])
def contains_verb(row):
    for part_of_speech in row.doc:
        if part_of_speech.pos_ == "VERB":
            return HAS_VERB
    return ABSTAIN


# Define keyword labeling functions
def keyword_lookup(row, keywords: List, label: int) -> int:
    """
    Check if any keywords appear in a body of text.
    """
    if any(word in row.text.lower() for word in keywords):
        return label
    return ABSTAIN


def make_keyword_lf(
    keywords: List[str], label: int, label_name: str
) -> LabelingFunction:
    """
    Create a LabelingFunction using the function keyword_lookup.
    """
    return LabelingFunction(
        name=f"keyword_{label_name}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )


keyword_support: LabelingFunction = make_keyword_lf(
    keywords["contact_support"], SUPPORT, "support"
)


def has_label(row):
    score = row["question"] + row["support_request"] + row["has_verb"]
    if score > -3:
        return True
    return False


######################################################################

lfs = [contains_question_word, keyword_support, contains_verb]
applier = PandasLFApplier(lfs=lfs)

L = applier.apply(df=data)
L = pd.DataFrame(L)
L.columns = ["question", "support_request", "has_verb"]

label_df = pd.concat([data, L], axis=1)
label_df["has_label"] = label_df.apply(has_label, axis=1).astype("int")

no_label_df = label_df[label_df.has_label == 0]

# breakpoint()

output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/")
label_df.to_csv(os.path.join(output_path, "labeled_data.csv"), index=False)