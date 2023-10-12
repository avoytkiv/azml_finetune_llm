import torch
import transformers
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from logs import preprocess_function

model_name = "./saved_model"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name).to('cuda')
model.eval()

# Load the dataset
small_dataset = load_dataset("hotels-reviews-small")

preprocessed_small_dataset = small_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

val_loader = torch.utils.data.DataLoader(
    preprocessed_small_dataset["validation"], 
    batch_size=32, 
    shuffle=False, 
    collate_fn=transformers.default_data_collator, 
    num_workers=2
)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to('cuda') for k, v in batch.items()}
        outputs = model(**batch)
        all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)

print("Validation Accuracy:", accuracy)

if __name__ == "__main__":
    evaluate()
