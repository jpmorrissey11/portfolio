import re
from dateutil.parser import parse
import pandas as pd
import html
import pandas as pd
from transformers import pipeline


def replace_html(text):
    text = re.sub("<.*?>", " ", text)
    text = re.sub("\(.*?\)", " ", text)
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


def get_summaries(text, summarizer=s):
    try:
        summary = summarizer(text)
        return summary[0]["summary_text"]
    except:
        return "too many tokens"


path = "/content/drive/MyDrive/Data/First Year 2022 _1500 Units Accounting DAS.csv"
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
