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
import wandb


class WAndBEarlyStoppingLoggingCallback(transformers.TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Check if early stopping has been triggered
        if state.is_early_stopping:
            # Log a custom message or metric indicating that early stopping occurred
            wandb.log({"early_stopping": True, "best_metric": state.best_metric})


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
    trainer_args['run_name'] = os.environ.get("SKYPILOT_TASK_ID")
    run_name = trainer_args['run_name']
    project_name = config["train"]["project_name"]
    os.environ["WANDB_PROJECT"] = project_name
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    trainer_args['fp16'] = True if torch.cuda.is_available() else False

    logger = get_logger("TRAIN", log_level=log_level)

    fix_random_seeds(random_state)

    logger.info(f"Running with SKYPILOT_TASK_ID: {run_name}")

    logger.info("Initializing wandb run...")
    run = wandb.init(
        project=project_name,
        name=run_name,  # Skypilot task id is used to identify the same job
        resume="allow",  # "allow" will resume the run if it exists, otherwise it will start a new run
    )

    logger.info("Checking for previous checkpoints...")
    last_checkpoint = wandb.run.summary.get("last_checkpoint", None)
    checkpoint_dir = None

    if last_checkpoint:
        logger.info(f"Found checkpoint {last_checkpoint}, downloading...")
        artifact = run.use_artifact(last_checkpoint)
        checkpoint_dir = artifact.download()

    # Load the model, possibly from a checkpoint
    if checkpoint_dir:
        logger.info(f"Resuming training from checkpoint in {checkpoint_dir}...")
        tokenizer, model = get_model_tokenizer(checkpoint_dir, num_labels)
    else:
        logger.info(f"Loading model {model_name} from scratch...")
        tokenizer, model = get_model_tokenizer(model_name, num_labels)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    logger.info("Loading dataset...")
    preprocessed_dataset = preprocess()

    logger.info("Training model...")
    training_args = transformers.TrainingArguments(**trainer_args)

    train_subset = torch.utils.data.Subset(preprocessed_dataset["train"], indices=range(0, 100))  # Use first 1000 samples
    # eval_subset = torch.utils.data.Subset(preprocessed_dataset["test"], indices=range(0, 200))  # Use first 200 samples for evaluation

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_subset, # preprocessed_dataset["train"],
        eval_dataset=preprocessed_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[WAndBEarlyStoppingLoggingCallback()],
    )

    cleanup_incomplete_checkpoints(training_args.output_dir)
    trainer.add_callback(CheckpointCallback())

    trainer.train(resume_from_checkpoint=checkpoint_dir if checkpoint_dir else None)

    # Update the last checkpoint in wandb
    last_checkpoint = os.path.join(training_args.output_dir, "checkpoint-{}".format(training_args.logging_steps))
    run.summary["last_checkpoint"] = last_checkpoint
    
    logger.info("Saving model")
    logger.info("Don't save on virtual machine as far as model is daved to wandb")
    # trainer.save_state()
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=finetuned_model_out_path)


if __name__ == "__main__":
    train()
