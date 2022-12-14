# %%
import os
import re

import numpy as np
import pandas as pd

from pyspark.sql.types import *
from pyspark.sql.functions import expr
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.feature import *
from pyspark.ml.clustering import LDA
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import sparknlp
from sparknlp import DocumentAssembler, Finisher
from sparknlp.annotator import *


spark = sparknlp.start()

# %%
df = spark.read.csv(
    "/Users/james.morrissey/Data/DisneylandReviews.csv", header=True, inferSchema=True
)

# %%
df_new = df.select("Review_ID", "Review_Text")

# %%
# Spark NLP requires the input dataframe or column to be converted to document.
document_assembler = (
    DocumentAssembler()
    .setInputCol("Review_Text")
    .setOutputCol("document")
    .setCleanupMode("shrink")
)
# Split sentence to tokens(array)
tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")
# clean unwanted characters and garbage
normalizer = Normalizer().setInputCols(["token"]).setOutputCol("normalized")
# remove stopwords
stopwords_cleaner = (
    StopWordsCleaner()
    .setInputCols("normalized")
    .setOutputCol("cleanTokens")
    .setCaseSensitive(False)
)
# stem the words to bring them to the root form.
stemmer = Stemmer().setInputCols(["cleanTokens"]).setOutputCol("stem")
# Finisher is the most important annotator. Spark NLP adds its own structure when we convert each row in the dataframe to document. Finisher helps us to bring back the expected structure viz. array of tokens.
finisher = (
    Finisher()
    .setInputCols(["stem"])
    .setOutputCols(["tokens"])
    .setOutputAsArray(True)
    .setCleanAnnotations(False)
)
# We build a ml pipeline so that each phase can be executed in sequence. This pipeline can also be used to test the model.
nlp_pipeline = Pipeline(
    stages=[
        document_assembler,
        tokenizer,
        normalizer,
        stopwords_cleaner,
        stemmer,
        finisher,
    ]
)

# %%
nlp_model = nlp_pipeline.fit(df)

# %%
processed_df = nlp_model.transform(df)

# %%
tokens_df = processed_df.select("Review_ID", "tokens")
tokens_df.show()

# %%
from pyspark.ml.feature import CountVectorizer

# %%
cv = CountVectorizer(inputCol="tokens", outputCol="features", vocabSize=500, minDF=3.0)

cv_model = cv.fit(tokens_df)

# %%
vectorized_tokens = cv_model.transform(tokens_df)

# %%
from pyspark.ml.clustering import LDA

# %%
num_topics = 6

# %%
lda = LDA(k=num_topics, maxIter=10)

# %%
model = lda.fit(vectorized_tokens)

# %%
ll = model.logLikelihood(vectorized_tokens)
lp = model.logPerplexity(vectorized_tokens)

# %%
print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
print("The upper bound on perplexity: " + str(lp))

# %%
# extract vocabulary from CountVectorizer
vocab = cv_model.vocabulary
topics = model.describeTopics()
topics_rdd = topics.rdd
topics_words = (
    topics_rdd.map(lambda row: row["termIndices"])
    .map(lambda idx_list: [vocab[idx] for idx in idx_list])
    .collect()
)
for idx, topic in enumerate(topics_words):
    print("topic: {}".format(idx))
    print("*" * 25)
    for word in topic:
        print(word)
    print("*" * 25)
