# %%
import sparknlp

spark = sparknlp.start()
from pyspark.sql.types import *
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

# %%
df = spark.read.csv(
    "/Users/james.morrissey/Data/Appfolio/nps_regular_query_results.csv", header=True
)

# %%
df = df.select("SFDC_ACCOUNT_KEY", "RESPONSE").na.drop()

# %%
df.count()

# %%
# Spark NLP requires the input dataframe or column to be converted to document.
document_assembler = (
    DocumentAssembler()
    .setInputCol("RESPONSE")
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
# train the pipeline
nlp_model = nlp_pipeline.fit(df)

# %%
processed_df = nlp_model.transform(df)

# %%
tokens_df = processed_df.select("SFDC_ACCOUNT_KEY", "tokens")

# %%
tokens_df.show()

# %%
# SPark ML Pipeline

cv = CountVectorizer(
    inputCol="finished_chunk",
    outputCol="features",
    vocabSize=1000,
    minDF=10.0,
    minTF=10.0,
)
idf = IDF(inputCol="features", outputCol="idf")
lda = LDA(k=10, maxIter=5)
### Let's create Spark-NLP Pipeline
mlPipeline = Pipeline(stages=[cv, idf, lda])

# %%
model.show()

# %%
