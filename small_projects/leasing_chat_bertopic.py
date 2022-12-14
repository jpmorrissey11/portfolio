import pandas as pd
import numpy as np
import bertopic
import spacy

nlp = spacy.load("en_core_web_sm")


import re
from dateutil.parser import parse
import pandas as pd
import html


def replace_html(text):
    text = re.sub("<.*?>", " ", text)
    # text = re.sub('\(.*?\)', ' ', text)
    text = re.sub(r"http\S+", "", text)
    return text.strip()


def clean(text):
    # convert html escapes like & to characters.
    text = html.unescape(text)
    # tags like
    text = re.sub(r"<[^<>]*>", " ", text)
    # markdown URLs like [Some text](https://....)
    text = re.sub(r"\[([^\[\]]*)\]\([^\(\)]*\)", r"\1", text)
    # text or code in brackets like [0]
    text = re.sub(r"\[[^\[\]]*\]", " ", text)
    # standalone sequences of specials, matches  but not #cool
    text = re.sub(r"(?:^|\s)[<>{}\[\]+|\\:-]{1,}(?:\s|$)", " ", text)
    # standalone sequences of hyphens like --- or ==
    text = re.sub(r"(?:^|\s)[\-=\+]{2,}(?:\s|$)", " ", text)
    # sequences of white spaces
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[0-9]{2}:[0-9]{2}:[0-9]{2}", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"^https?:\/\/.*[\r\n]*", "", text, flags=re.MULTILINE)
    return text.strip()


# @title Funcs
import re
from dateutil.parser import parse
import pandas as pd
import html


def replace_html(text):
    text = re.sub("<.*?>", " ", text)
    text = re.sub("\(.*?\)", " ", text)
    text = re.sub(r"http\S+", "", text)
    return text.strip()


def get_summaries(text, summarizer=s):
    try:
        summary = summarizer(text)
        return summary[0]["summary_text"]
    except:
        return "too many tokens"


def clean(text):
    # convert html escapes like & to characters.
    text = html.unescape(text)
    # tags like
    text = re.sub(r"<[^<>]*>", " ", text)
    # markdown URLs like [Some text](https://....)
    text = re.sub(r"\[([^\[\]]*)\]\([^\(\)]*\)", r"\1", text)
    # text or code in brackets like [0]
    text = re.sub(r"\[[^\[\]]*\]", " ", text)
    # standalone sequences of specials, matches  but not #cool
    text = re.sub(r"(?:^|\s)[<>{}\[\]+|\\:-]{1,}(?:\s|$)", " ", text)
    # standalone sequences of hyphens like --- or ==
    text = re.sub(r"(?:^|\s)[\-=\+]{2,}(?:\s|$)", " ", text)
    # sequences of white spaces
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[0-9]{2}:[0-9]{2}:[0-9]{2}", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"^https?:\/\/.*[\r\n]*", "", text, flags=re.MULTILINE)
    return text.strip()


def remove_patterns(text):
    patterns = [
        "Chat Started",
        "Chat Origin",
        "Agent",
        "Live Support Agent",
        "Enter a question to start a conversation with a Live Support Agent",
        "East Tier 1",
        "East Tier 2",
        "East Tier 3",
        "East Tier 4",
        "West Tier 1",
        "West Tier 2",
        "West Tier 3",
        "West Tier 4",
        "Enter a question to start a conversation with a Live Support",
        "How can we help?",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        ":",
        ",",
    ]
    for pattern in patterns:
        text = text.replace(pattern, "")
    return text


# @title Load and Preprocess Data
import pandas as pd

path = "/content/drive/MyDrive/Data/First Year 2022 _1500 Units Accounting DAS.csv"
df = pd.read_csv(path)
df.columns = [c.lower() for c in df.columns]

columns_to_keep = [
    "case_sub_reason",
    "chat_transcipt_body",
    "customer_adoption_category",
]
df = df[columns_to_keep]
df.columns = ["case_sub_reason", "text", "das"]

df["text"] = df["text"].apply(replace_html)
df["text"] = df["text"].apply(clean)
df["text"] = df["text"].apply(remove_patterns)

# @title Instantiate Summarization Pipeline
import pandas as pd
from transformers import pipeline

s = pipeline(
    "summarization",
    model="philschmid/bart-large-cnn-samsum",
    min_length=10,
    max_length=50,
)

summary_list = []
for i in range(len(df)):
    texts = df["text"].iloc[i : i + 5].values
    for chat in texts:
        temp_sum = get_summaries(chat)
        print(i, temp_sum)
        summary_list.append(temp_sum)

df = pd.DataFrame(pd.Series(summary_list), columns=["summary"])

df = df[df.summary.apply(lambda x: len(x) > 50)]
df = df[~df.summary.apply(lambda x: "chat" in x)]

# @title Generate embeddings, get keywords, build vocab
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

docs = df.dropna(subset=["text"]).summary.values.tolist()
embeddings = sentence_model.encode(docs, show_progress_bar=True)

kw_model = KeyBERT()
keywords = kw_model.extract_keywords(docs)

# Create our vocabulary
vocabulary = [k[0] for keyword in keywords for k in keyword]
vocabulary = list(set(vocabulary))

# @title Build topic model and fit to data
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN

ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
# ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)
vectorizer_model = CountVectorizer(stop_words="english", vocabulary=vocabulary)

model = BERTopic(
    language="english",
    vectorizer_model=vectorizer_model,
    diversity=0.2,
    min_topic_size=10,
    ctfidf_model=ctfidf_model,
    top_n_words=15,
)

topics, probs = model.fit_transform(docs, embeddings)
num_classified = model.get_topic_freq().Count.values[1:].sum()
print(f"classified {num_classified} out of {len(docs)}")

x = pd.DataFrame({"text": df.text, "summary": df.summary, "topic": model.topics_})

model.get_topic_info()

# @title Specify Custom Labels
topic_labels = {
    0: "renewal,lease,signed",
    1: "fees,application,pay",
    2: "zillow,listing,vacany",
    3: "guest,cards,virtual",
    4: "appfolio,website,html",
    5: "converted,saved,applications",
    6: "denied,cancelled,deleted",
    7: "expedite,case,expert",
    8: "manually,enter,cosigner",
}
model.set_topic_labels(topic_labels)

from umap import UMAP

# Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
reduced_embeddings = UMAP(
    n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine"
).fit_transform(embeddings)
model.visualize_documents(
    docs, reduced_embeddings=reduced_embeddings, width=1600, custom_labels=True
)

model.visualize_barchart(top_n_topics=8, custom_labels=True)
