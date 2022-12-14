# %%
import pandas as pd
import numpy as np
import json
from pyspark.ml import Pipeline
import pyspark.sql.functions as F
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline

# %%
! python --version

# %%
MODEL_NAME='sentimentdl_use_twitter'

# %%
text_list = [
             """We as a company used to us property boss and it was a nightmare. Appfolio makes it so much easier to navigate and keep track of everything that's important. Definitely need to improve on tech support. I can never speak to a live agent anymore. Its very hard to get ahold of someone and its very frustrating.""",
             """Your customer service and help is terrible. There is a great user group on facebook for Appfolio. It is more helpful than your customer service team. For what we pay for Appfolio you Customer Sucess people should be better.""",
             """Easy to use, constantly improving, listens to customer base.""",
             """Appfolio has almost all of the components of a comprehensive, efficient system to replace files and other forms of paperwork. I am very pro-documentation for correspondence between our team and our tenants. Appfolio allows us to see other conversations and know where we are at with all tenants."""
 ]

# %%
spark = sparknlp.start()

# %%
pipeline = PretrainedPipeline('explain_document_ml', lang='en')

# %%
pipeline.annotate("""We as a compfany used to us property boss and it was a nightmare. Appfolio makes it so much easier to navigate and keep track of everything that's important. Definitely need to improve on tech support. I can never speak to a live agent anymore. Its very hard to get ahold of someone and its very frustrating.""")

# %%


# %%


# %%


# %%


# %%
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
    
use = UniversalSentenceEncoder.pretrained(name="tfhub_use", lang="en")\
 .setInputCols(["document"])\
 .setOutputCol("sentence_embeddings")

# %%
sentimentdl = SentimentDLModel.pretrained(name=MODEL_NAME, lang="en")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("sentiment")

nlpPipeline = Pipeline(
      stages = [
          documentAssembler,
          use,
          sentimentdl
      ])


# %%
empty_df = spark.createDataFrame([['']]).toDF("text")

pipelineModel = nlpPipeline.fit(empty_df)

df = spark.createDataFrame(pd.DataFrame({"text":text_list}))
result = pipelineModel.transform(df)

# %%
help(result)

# %%
result.select(F.explode(F.arrays_zip('document.result', 'sentiment.result'))

# %%
result_df = result.select(F.explode(F.arrays_zip('document.result', 'sentiment.result')).alias("cols")) \
.select(F.expr("cols['0']").alias("document"),
        F.expr("cols['1']").alias("sentiment")).toPandas()

# %%
result_df.values

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%
import pandas as pd
import numpy as np
import json
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline

# %%
spark = sparknlp.start()

# %%
text_list = [
             """We are super sorry we ever signed up for AppFolio. There are FAR too many issuesthe support system is straight outta 1990 Get with the times or you will never survive.""",
             """The customer support I received after we signed up in February was nothing short of horrible. I try to do things myself anymore because the last time I tried I actually never got a call back on an important need Lots of improvement needed. System is good. CS is bad."""
             ]

# %%
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
    
use = UniversalSentenceEncoder.pretrained(name="tfhub_use", lang="en")\
 .setInputCols(["document"])\
 .setOutputCol("sentence_embeddings")


sentimentdl = SentimentDLModel.pretrained(name=MODEL_NAME, lang="en")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("sentiment")

nlpPipeline = Pipeline(
      stages = [
          documentAssembler,
          use,
          sentimentdl
      ])

# %%
empty_df = spark.createDataFrame([['']]).toDF("text")

pipelineModel = nlpPipeline.fit(empty_df)

df = spark.createDataFrame(pd.DataFrame({"text":text_list}))
result = pipelineModel.transform(df)

# %%
result.select(F.explode(F.arrays_zip('document.result', 'sentiment.result')).alias("cols")) \
.select(F.expr("cols['0']").alias("document"),
        F.expr("cols['1']").alias("sentiment")).show(truncate=False)

# %%
document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier = BertForSequenceClassification \
      .pretrained('bert_sequence_classifier_finbert_tone', 'en') \
      .setInputCols(['token', 'document']) \
      .setOutputCol('class')

pipeline = Pipeline(stages=[document_assembler, tokenizer, sequenceClassifier])

example = spark.createDataFrame([['Growth is strong and we have plenty of liquidity.']]).toDF("text")
result = pipeline.fit(example).transform(example)

# %%
3result.show()

# %%



