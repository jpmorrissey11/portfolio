# Generate bootstrap training data

import pandas as pd
from transformers import pipeline
from utility import replace_html, clean

def get_summaries(text, summarizer):
    try:
        summary = summarizer(text)
        return summary[0]["summary_text"]
    except:
        return "too many tokens"


s = pipeline(
    "summarization",
    model="philschmid/bart-large-cnn-samsum",
    min_length=10,
    max_length=50,
)


path = "path/to/data"
df = pd.read_csv(path)
df.columns = [c.lower() for c in df.columns]

df["text"] = df["text"].apply(replace_html)
df["text"] = df["text"].apply(clean)
df["summary"] = df["summary"].apply(get_summaries, args=(s,))

summary_df = pd.DataFrame(pd.Series(summary_list), columns=["summary"])
summary_df["text"] = df["text"]

summary_df.to_csv("./data/generated_summaries.csv")
