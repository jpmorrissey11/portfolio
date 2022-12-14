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


def remove_patterns(text, patterns):
    for pattern in patterns:
        text = text.replace(pattern, "")
    return text


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
