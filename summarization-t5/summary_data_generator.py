import re
from dateutil.parser import parse
import pandas as pd
import html
import pandas as pd
from transformers import pipeline
from utility import replace_html, text


def get_summaries(text, summarizer=s):
    try:
        summary = summarizer(text)
        return summary[0]["summary_text"]
    except:
        return "too many tokens"


path = "path/to/data"
df = pd.read_csv(path)
df.columns = [c.lower() for c in df.columns]

df["text"] = df["text"].apply(replace_html)
df["text"] = df["text"].apply(clean)

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

x = pd.DataFrame(pd.Series(summary_list), columns=["summary"])
x["text"] = df["text"]
x.to_csv("/content/drive/MyDrive/Data/generated_summaries.csv")
