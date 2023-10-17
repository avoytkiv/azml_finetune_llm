import os
import numpy as np
import torch
import transformers
from logs import get_logger
from preprocess import preprocess, get_model_tokenizer
import dvc.api
from logs import get_logger
from utils import CheckpointCallback, cleanup_incomplete_checkpoints, safe_save_model_for_hf_trainer, fix_random_seeds
from sklearn.metrics import accuracy_score


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}


def train():

    config = dvc.api.params_show()

    log_level = config["base"]["log_level"]
    random_state = config["base"]["random_state"]
    model_name = config["preprocess"]["model_name"]
    num_labels = config["data"]["num_labels"]
    trainer_args = config["train"]["trainer_args"]
    finetuned_model_out_path = config["train"]["finetuned_model_out_path"]

    if torch.cuda.is_available():
        trainer_args['fp16'] = True
    else:
        trainer_args['fp16'] = False
    os.environ["WANDB_PROJECT"] = "skypilot-test"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    logger = get_logger("TRAIN", log_level=log_level)

    fix_random_seeds(random_state)

    logger.info(f"Loading model {model_name}...")
    tokenizer, model = get_model_tokenizer(model_name, num_labels)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    logger.info("Loading dataset...")
    preprocessed_dataset = preprocess()

    logger.info("Training model...")
    training_args = transformers.TrainingArguments(**trainer_args)

    # train_subset = torch.utils.data.Subset(preprocessed_dataset["train"], indices=range(0, 1000))  # Use first 1000 samples
    # eval_subset = torch.utils.data.Subset(preprocessed_dataset["test"], indices=range(0, 200))  # Use first 200 samples for evaluation

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=preprocessed_dataset["train"],
        eval_dataset=preprocessed_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    cleanup_incomplete_checkpoints(training_args.output_dir)
    trainer.add_callback(CheckpointCallback())

    if not os.listdir(training_args.output_dir):
        trainer.train()
    else:
        logger.info("Resuming training from checkpoint")
        trainer.add_callback(CheckpointCallback)
        trainer.train(resume_from_checkpoint=True)
    
    logger.info("Saving model")
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=finetuned_model_out_path)


if __name__ == "__main__":
    train()
