# src/config.py

class Config:
    # --- Models & Dataset ---
    TEACHER_MODEL_PATH = "./models/teacher_model" # Path to the fine-tuned BART-large model
    STUDENT_MODEL_CHECKPOINT = "sshleifer/distilbart-cnn-12-6" # The smaller student model
    DATASET_NAME = "xsum"
    
    # --- Data Preprocessing ---
    MAX_INPUT_LENGTH = 1024
    MAX_TARGET_LENGTH = 128
    
    # --- Training Hyperparameters ---
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    NUM_TRAIN_EPOCHS = 3
    
    # --- Distillation Parameters ---
    ALPHA = 0.5  # Weight for the hard loss (standard training)
    TEMPERATURE = 2.0 # Softens the teacher's probability distribution
    
    # --- Directory for saving results ---
    OUTPUT_DIR = "./results"