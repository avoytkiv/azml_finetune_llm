import torch
import transformers
from preprocess import preprocess, get_model_tokenizer, fix_random_seeds
import dvc.api
from sklearn.metrics import accuracy_score
from logs import get_logger
from pathlib import Path
import sys
import json
from tqdm import tqdm


src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))


def evaluate():
    config = dvc.api.params_show()

    log_level = config["base"]["log_level"]
    random_state = config["base"]["random_state"]
    model_name = config["preprocess"]["model_name"]
    batch_size = config["evaluate"]["batch_size"]
    shuffle = config["evaluate"]["shuffle"]
    num_workers = config["evaluate"]["num_workers"]
    metrics_path = Path(src_path, config["evaluate"]["metrics_path"])
    num_labels = config["data"]["num_labels"]

    logger = get_logger("EVALUATE", log_level=log_level)

    fix_random_seeds(random_state)

    logger.info(f"Loading model {model_name}...")
    tokenizer, model = get_model_tokenizer(model_name, num_labels)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    logger.info("Loading dataset...")
    preprocessed_dataset = preprocess()

    val_loader = torch.utils.data.DataLoader(
        preprocessed_dataset["test"],
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=transformers.default_data_collator,
        num_workers=num_workers
    )

    all_preds = []
    all_labels = []

    logger.info("Evaluating...")
    max_batches = 10
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Evaluating", unit="batch")):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

            if i >= max_batches - 1:
                break

    accuracy = accuracy_score(all_labels, all_preds)

    logger.info(f"Accuracy: {accuracy}")

    json.dump({"accuracy": accuracy}, open(metrics_path, "w"))


if __name__ == "__main__":
    evaluate()
