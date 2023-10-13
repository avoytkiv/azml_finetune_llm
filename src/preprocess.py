import dvc.api
from datasets import Dataset, DatasetDict
import pandas as pd
import transformers
from pathlib import Path
import sys
from logs import get_logger
import random
import torch
import numpy as np


src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))


def fix_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_data(path) -> pd.DataFrame:
    return pd.read_csv(path)


def labels_to_int(dataset) -> pd.DataFrame:
    labels = dataset["Label"].unique()
    label2id = {label: i for i, label in enumerate(labels)}
    dataset["Label"] = dataset["Label"].map(label2id)
    return dataset


def from_pd_to_hf(dataset: pd.DataFrame) -> Dataset:
    dataset = Dataset.from_pandas(dataset)
    dataset = dataset.rename_column("Label", "label")
    dataset = dataset.rename_column("Text", "text")
    return dataset


def split_dataset(dataset: Dataset, test_train_split: float, random_state: int) -> DatasetDict:
    dataset = dataset.train_test_split(test_size=test_train_split, shuffle=True, seed=random_state)
    return dataset


def get_model_tokenizer(model_name, num_labels):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model


def preprocess_batch(batch, tokenizer, max_length):
    """Process a batch of examples."""
    result = tokenizer(
        batch['text'],
        padding='max_length',
        max_length=max_length, 
        truncation=True,
        return_tensors="pt"  # Return PyTorch tensors
    )
    result['labels'] = batch['label']
    return result


def preprocess() -> DatasetDict:
    """Preprocess the datasets."""
    config = dvc.api.params_show()
    
    logger = get_logger("PREPROCESS", log_level=config["base"]["log_level"])

    random_state = config["base"]["random_state"]
    test_train_split = config["preprocess"]["test_train_split"]
    data_path = Path(src_path, config["data"]["data_path"])
    max_length = config["preprocess"]["max_length"]
    model_name = config["preprocess"]["model_name"]
    num_labels = config["data"]["num_labels"]

    fix_random_seeds(random_state)

    logger.info("Read data")
    dataset = read_data(data_path)
    
    logger.info("Convert labels to int")
    dataset = labels_to_int(dataset)
    
    logger.info("Convert to HuggingFace dataset")
    dataset = from_pd_to_hf(dataset)

    logger.info("Split the dataset")
    dataset = split_dataset(dataset, test_train_split=test_train_split, random_state=random_state)

    logger.info("Assert that the number of labels is correct")
    assert len(set(dataset["train"]["label"])) == num_labels
    assert len(set(dataset["test"]["label"])) == num_labels


    logger.info(f"Load tokenizer with {model_name} model")
    tokenizer, model = get_model_tokenizer(model_name, num_labels)

    logger.info("Preprocess the dataset")
    preprocessed_dataset = {}
    for split in dataset:
        logger.info(f"Preprocessing {split} split")
        preprocessed_dataset[split] = dataset[split].map(lambda x: preprocess_batch(x, tokenizer, max_length), batched=True)

    return preprocessed_dataset


if __name__ == "__main__":
    preprocess()