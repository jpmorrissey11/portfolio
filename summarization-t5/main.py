import pytorch_lightning as pl
import pandas as pd
from argparse import Namespace

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from utility import replace_html, clean, remove_patterns
from dm import SummaryDataModule
from model import SummaryModel

from transformers import (
    T5TokenizerFast as T5Tokenizer,
)


pl.seed_everything(42)

args = Namespace(
    path_to_data="path/to/data.csv",
    columns=["summary", "text"],
)

df = pd.read_csv(args.path_to_data)
df.columns = [c.lower() for c in df.columns]
df.columns = args.columns
df["text"] = df["text"].apply(replace_html)
df["text"] = df["text"].apply(clean)
df["text"] = df["text"].apply(remove_patterns, args=(["pattern_to_remove"],))
df["text"] = df["text"].apply(lambda x: x[50:])

df.summary = "summarize: " + df.summary

# @title Get Splits
train_df, test_df = train_test_split(df, test_size=0.1)

MODEL_NAME = "t5-base"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

N_EPOCHS = 2
BATCH_SIZE = 4

dm = SummaryDataModule(train_df, test_df, tokenizer, batch_size=BATCH_SIZE)
model = SummaryModel()

checkpoint_callback = ModelCheckpoint(
    dirpath="t5_checkpoints",
    filename="best_checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min",
)

logger = TensorBoardLogger("t5_lightning_logs", name="news-summary")

trainer = pl.Trainer(
    logger=logger, callbacks=[checkpoint_callback], max_epochs=N_EPOCHS
)

# @title Train
trainer.fit(model, dm)
