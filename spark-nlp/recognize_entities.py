# %%
import json
import pandas as pd
import numpy as np

from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline

spark = sparknlp.start()

# %%
# If you change the model, re-run all the cells below
# Other applicable models: ner_dl, ner_dl_bert
MODEL_NAME = "onto_100"

# %%
text_list = [
    """We as a company used to us property boss and it was a nightmare. Appfolio makes it so much easier to navigate and keep track of everything that's important. Definitely need to improve on tech support. I can never speak to a live agent anymore. Its very hard to get ahold of someone and its very frustrating.""",
    """Your customer service and help is terrible. There is a great user group on facebook for Appfolio. It is more helpful than your customer service team. For what we pay for Appfolio you Customer Sucess people should be better.""",
    """Easy to use, constantly improving, listens to customer base.""",
    """Appfolio has almost all of the components of a comprehensive, efficient system to replace files and other forms of paperwork. I am very pro-documentation for correspondence between our team and our tenants. Appfolio allows us to see other conversations and know where we are at with all tenants.""",
]

# %%
documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")

# ner_dl and onto_100 model are trained with glove_100d, so the embeddings in
# the pipeline should match
if (MODEL_NAME == "ner_dl") or (MODEL_NAME == "onto_100"):
    embeddings = (
        WordEmbeddingsModel.pretrained("glove_100d")
        .setInputCols(["document", "token"])
        .setOutputCol("embeddings")
    )

# Bert model uses Bert embeddings
elif MODEL_NAME == "ner_dl_bert":
    embeddings = (
        BertEmbeddings.pretrained(name="bert_base_cased", lang="en")
        .setInputCols(["document", "token"])
        .setOutputCol("embeddings")
    )

ner_model = (
    NerDLModel.pretrained(MODEL_NAME, "en")
    .setInputCols(["document", "token", "embeddings"])
    .setOutputCol("ner")
)

ner_converter = (
    NerConverter().setInputCols(["document", "token", "ner"]).setOutputCol("ner_chunk")
)

nlp_pipeline = Pipeline(
    stages=[documentAssembler, tokenizer, embeddings, ner_model, ner_converter]
)

# %%
empty_df = spark.createDataFrame([[""]]).toDF("text")
pipeline_model = nlp_pipeline.fit(empty_df)
df = spark.createDataFrame(pd.DataFrame({"text": text_list}))
result = pipeline_model.transform(df)

# %%
from sparknlp_display import NerVisualizer

NerVisualizer().display(
    result=result.collect()[2], label_col="ner_chunk", document_col="document"
)

# %%
