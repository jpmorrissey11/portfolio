import pandas as pd
from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch

data_dir = "path/to/data/"
dataset_name = "product_feedback.csv"

data = pd.read_csv(data_dir + dataset_name)

# @title Default title text
data.columns = ["text", "labels"]


train_df["labels"] = train_df["labels"].apply(lambda x: int(x))
test_df["labels"] = test_df["labels"].apply(lambda x: int(x))

train_df.to_csv(data_dir + "train_data.csv", index=False)
test_df.to_csv(data_dir + "test_data.csv", index=False)

Dataset.from_pandas(train_df, split="train")

path_to_train_data = data_dir + "train_data.csv"
path_to_test_data = data_dir + "test_data.csv"

from datasets import load_dataset

dataset = load_dataset(
    "csv", data_files={"train": path_to_train_data, "test": path_to_test_data}
)

train_dataset = dataset["train"]
test_dataset = dataset["test"]
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text"])
tokenized_test_dataset = tokenized_test_dataset.remove_columns(["text"])

tokenized_train_dataset.set_format("torch")
tokenized_test_dataset.set_format("torch")

tokenized_train_dataset["input_ids"].shape

from torch.utils.data import DataLoader

train_dataloader = DataLoader(tokenized_train_dataset, shuffle=True, batch_size=8)
test_dataloader = DataLoader(tokenized_test_dataset, batch_size=4)


def bar(**kwargs):
    for a in kwargs:
        print(a, kwargs[a])


batch = next(iter(train_dataloader))
#
print(batch["input_ids"])
# bar(**batch)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=4
)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

from datasets import load_metric

metric = load_metric("accuracy")
model.eval()
pred_list = []
for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    pred_list.append(predictions)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()

pred_list

"""# Output"""

sequences = [
    "It was really difficult trying to figure out how to use this for accounting. very clunky",
    "Im not able to generate the accounting reports I need",
    "Accounting leaves a lot to be desired",
    "This was clearly not designed with commercial properties in mind",
]
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
batch = {k: v.to(device) for k, v in tokens.items()}
output = model(**batch)

torch.argmax(output.logits, dim=-1)
