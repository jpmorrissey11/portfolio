# @title Funcs
import re
from dateutil.parser import parse
import html
import textacy
import textacy.preprocessing as tprep


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


RE_SUSPICIOUS = re.compile(r"[<>{}\[\]\\]")


def impurity(text, min_len=10):
    """returns the share of suspicious characters in a text"""
    if text == None or len(text) < min_len:
        return 0
    else:
        return len(RE_SUSPICIOUS.findall(text)) / len(text)


def normalize(text):
    text = tprep.normalize.hyphenated_words(text)
    text = tprep.normalize.quotation_marks(text)
    text = tprep.normalize.unicode(text)
    text = tprep.remove.accents(text)
    return text
