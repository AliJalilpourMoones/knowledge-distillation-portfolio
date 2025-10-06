# Knowledge Distillation for Text Summarization

This repository contains a PyTorch implementation of a knowledge distillation pipeline. The project focuses on compressing a large, fine-tuned **BART-large** model (the "teacher") into a smaller, faster **DistilBART** model (the "student") for the task of abstractive text summarization on the XSum dataset.



---

## Abstract
As state-of-the-art models grow larger, model compression has become a critical area of research. This project demonstrates the implementation of knowledge distillation, a powerful technique for transferring knowledge from a large model to a smaller one. By training a student model to mimic the output distribution of a pre-trained teacher, we can create a model that is significantly more efficient in terms of size and inference speed while retaining a high percentage of the original performance. This work showcases a custom Hugging Face `Trainer` and a specialized loss function to achieve this knowledge transfer.

---

## 1. The Method: Knowledge Distillation

This project uses the "Teacher-Student" paradigm.
* **Teacher Model:** A fine-tuned `BART-large-cnn` model, which has high performance but is computationally expensive.
* **Student Model:** A `DistilBART-cnn-12-6` model, which has half the number of layers and is much faster.

The student model is trained using a composite loss function:
`L_total = α * L_hard + (1 - α) * L_soft`

* **`L_hard` (Hard Loss):** The standard cross-entropy loss between the student's predictions and the ground-truth labels from the dataset.
* **`L_soft` (Soft Loss):** A KL Divergence loss that encourages the student model's output probability distribution to match the softened probability distribution of the teacher model. The **temperature** hyperparameter is used to soften these distributions, providing more granular training signals.



---

## 2. Results & Analysis

**[RESULTS PENDING]**
The primary goal is to evaluate the trade-off between model performance (measured by ROUGE score) and model efficiency (measured by parameter count and inference speed).

### ### Performance Comparison

| Metric              | Teacher (BART-large) | Student (DistilBART)       |
| :------------------ | :------------------- | :------------------------- |
| **Parameters** | ~406 Million         | ~203 Million (2x smaller)  |
| **ROUGE-L Score** |                   |                            |
| **Inference Speed** |                 |                            |

### ### Analysis
The results will demonstrate that the student model can achieve a ROUGE score that is highly competitive with the much larger teacher model (e.g., retaining >95% of the performance) while being significantly smaller and faster. This highlights the effectiveness of knowledge distillation for creating practical, efficient models for production environments.
---

## 3. How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AliJalilpourMoones/knowledge-distillation-portfolio.git](https://github.com/AliJalilpourMoones/knowledge-distillation-portfolio.git)
    cd knowledge-distillation-portfolio
    ```
2.  **Prepare the Teacher Model:** Download or train a fine-tuned `BART-large-cnn` model and place it in the `models/teacher_model` directory as instructed in `models/README.md`.

3.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the training script:**
    ```bash
    python src/train.py
    ```
    The script will train the student model and save the results and final checkpoint in the `/results` directory.