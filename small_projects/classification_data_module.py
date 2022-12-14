import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader


from datasets import Dataset
from transformers import AutoTokenizer
from typing import Optional
from sklearn.model_selection import train_test_split
import os
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


data_dir = "/content/drive/MyDrive/Data/"
df_name = "data.csv"


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path,
        data_dir,
        df_name,
        batch_size=8,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.df_name = df_name
        self.data_dir = data_dir
        self.batch_size = 8

    def read_csv(self) -> list:
        """Reads a comma separated value file.
        :param path: path to a csv file.
        :return: List of records as dictionaries
        """
        self.df = pd.read_csv(os.path.join(self.data_dir, df_name))
        self.df.columns = ["text", "labels"]

    def setup(self, stage: Optional[str] = None):
        test_size = 0.2
        val_size = test_size
        train_df, test_df = train_test_split(
            self.df, test_size=test_size, random_state=42, shuffle=True
        )
        train_df, val_df = train_test_split(
            train_df, test_size=val_size, random_state=42, shuffle=True
        )
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        val_dataset = Dataset.from_pandas(val_df)

        train_dataset = train_dataset.remove_columns(["__index_level_0__"])
        test_dataset = test_dataset.remove_columns(["__index_level_0__"])
        val_dataset = val_dataset.remove_columns(["__index_level_0__"])

        self.dataset_splits = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        for split in self.dataset_splits.keys():
            self.dataset_splits[split] = self.dataset_splits[split].map(
                self.tokenize_datasets, batched=True
            )

            self.dataset_splits[split] = self.dataset_splits[split].remove_columns(
                ["text"]
            )
            self.dataset_splits[split].set_format(type="torch")

    def tokenize_datasets(self, sample, indices=None):
        return self.tokenizer(sample["text"], padding="max_length", truncation=True)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_splits["train"], batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.dataset_splits["val"], batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_splits["test"], batch_size=self.batch_size)
