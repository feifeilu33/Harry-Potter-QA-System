import torch
import json
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "BAAI/bge-reranker-v2-gemma"
NUM_LABELS = 3
LABELS = {"character": 0, "plot": 1, "other": 2}

class TextDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(item["text"], truncation=True, max_length=512, padding="max_length", return_tensors="pt")
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs["labels"] = torch.tensor(LABELS[item["label"]], dtype=torch.long)
        return inputs

def load_dataset(file_path, tokenizer):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TextDataset(data, tokenizer)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# Using LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,  
    lora_alpha=16,  
    lora_dropout=0.1,  
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("dataset_tune.json", tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir="./logs",
    save_strategy="epoch",
    fp16=True,  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# 训练模型
trainer.train()

# 保存微调后的模型
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Model fine-tuned with LoRA and saved successfully!")
