# src/train.py

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# Import from our other project files
from config import Config
from data import get_tokenized_dataset

class DistillationTrainer(Seq2SeqTrainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # Move teacher to correct device
        if self.args.device is not None:
            self.teacher.to(self.args.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        # --- Standard "Hard" Loss ---
        # This is the normal loss from the student model against the true labels.
        outputs_student = model(**inputs)
        loss_hard = outputs_student.loss
        
        # --- Distillation "Soft" Loss ---
        # We get the logits from both the teacher and the student.
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)
        
        logits_student = outputs_student.logits
        logits_teacher = outputs_teacher.logits
        
        # Soften the probability distributions with temperature
        loss_soft = F.kl_div(
            F.log_softmax(logits_student / self.args.temperature, dim=-1),
            F.softmax(logits_teacher / self.args.temperature, dim=-1),
            reduction="batchmean"
        ) * (self.args.temperature ** 2)
        
        # --- Combine the losses ---
        loss = self.args.alpha * loss_hard + (1 - self.args.alpha) * loss_soft
        
        return (loss, outputs_student) if return_outputs else loss

def main():
    config = Config()
    
    # --- 1. Load Tokenizer and Models ---
    print("Loading models and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_MODEL_CHECKPOINT)
    student_model = AutoModelForSeq2SeqLM.from_pretrained(config.STUDENT_MODEL_CHECKPOINT)
    
    # For this script to run, a fine-tuned teacher model must be saved at this path.
    # teacher_model = AutoModelForSeq2SeqLM.from_pretrained(config.TEACHER_MODEL_PATH) 
    # Using the base model as a placeholder since we are not training.
    teacher_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    # --- 2. Load and prepare the dataset ---
    print("Loading and tokenizing dataset...")
    tokenized_dataset = get_tokenized_dataset(tokenizer)

    # --- 3. Set up training arguments ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        evaluation_strategy="epoch",
        # Custom args for our trainer
        alpha=config.ALPHA,
        temperature=config.TEMPERATURE,
    )

    # --- 4. Create the Custom Trainer ---
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=student_model)

    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- 5. Start Training ---
    print("Starting training...")
    # trainer.train() # This is the command to start the actual training
    print("\nSetup complete. The project is ready to be trained by running `trainer.train()`.")
    
if __name__ == "__main__":
    main()