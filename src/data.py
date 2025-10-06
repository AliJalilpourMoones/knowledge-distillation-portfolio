# src/data.py

from datasets import load_dataset
from transformers import AutoTokenizer
from config import Config

def get_tokenized_dataset(tokenizer):
    """
    Loads the XSum dataset and tokenizes it.
    """
    config = Config()
    dataset = load_dataset(config.DATASET_NAME, split="train").train_test_split(test_size=0.1)
    
    def preprocess_function(examples):
        prefix = "summarize: "
        inputs = [prefix + doc for doc in examples["document"]]
        model_inputs = tokenizer(inputs, max_length=config.MAX_INPUT_LENGTH, truncation=True)
        labels = tokenizer(text_target=examples["summary"], max_length=config.MAX_TARGET_LENGTH, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset