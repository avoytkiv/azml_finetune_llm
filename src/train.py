import pandas as pd
import torch
import transformers
from datasets import load_dataset
from logs import preprocess_function
import dvc.api
from logs import get_logger

def train():

    config = dvc.api.params_show()

    logger = get_logger("TRAIN", log_level=config["base"]["log_level"])

    model_name = config["data"]["model_name"]
    num_labels = config["train"]["num_labels"]
    data_train_path = config["data"]["data_train"]

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Load the dataset
    dataset = pd.read_csv(data_train_path)

    preprocessed_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Training configuration
    training_args = transformers.TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        save_total_limit=2,
        push_to_hub=False,
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=preprocessed_dataset["train"],
        eval_dataset=preprocessed_dataset["validation"],
        tokenizer=tokenizer
    )

    trainer.train()

    # Save the model
    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")

if __name__ == "__main__":
    train()
