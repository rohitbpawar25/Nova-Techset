# MCQ Generation

MCQ Generation using Unsloth Phi + Chain-of-Thought (CoT)

Purpose of This Document

This document explains both notebooks step by step so the team can:

Understand the training pipeline (fine-tuning Phi using Unsloth + CoT)

Understand the inference pipeline (MCQ generation)

Understand how the Streamlit UI is built on top of the trained model

The goal is that any team member can read this and reproduce or extend the system.

High-Level Architecture

Dataset Preparation (CoT-based MCQs)

Model Fine-Tuning (Unsloth + TRL)

Model Saving & Export

Inference Pipeline (MCQ Generation)

Streamlit Application (UI Layer)

Notebook 1: MCQ_Generation_Unsloth_phi_COT.ipynb

Step 1: Environment Setup

pip install unsloth accelerate bitsandbytes transformers datasets trl 

Why?

unsloth → Memory-efficient LLM fine-tuning

trl → Trainer utilities for instruction/CoT tuning

bitsandbytes → 4-bit / 8-bit quantization

Step 2: Dataset Folder Creation

os.mkdir('cot_dataset') 

Purpose:

Central directory for Chain-of-Thought MCQ training data

Step 3: Dataset Loading & Formatting

JSON-based MCQ samples

Each record contains:

instruction

input (skill + experience)

output (MCQs with reasoning)

Key Idea:

The model is trained to reason before answering (CoT)

Example (conceptual):

Instruction: Generate MCQsInput: Python, 3–5 yearsOutput:Reasoning: ...Q1: ... 

Step 4: Model Initialization (Unsloth)

FastLanguageModel.from_pretrained(    model_name="phi",    load_in_4bit=True,    max_seq_length=2048) 

Why Unsloth?

Faster training

Lower GPU memory usage

Optimized for instruction tuning

Step 5: LoRA Configuration

Low-Rank Adaptation (LoRA)

Trains only small adapter layers

Benefits:

Faster training

Less overfitting

Easy model versioning

Step 6: Trainer Setup (TRL)

Key parameters:

Epochs

Batch size

Learning rate

Gradient accumulation

trainer.train() 

Output:

Fine-tuned Phi model with MCQ + reasoning capability

Step 7: Model Saving

model.save_pretrained("final_model") 

Saved Artifacts:

Base model weights

LoRA adapters

Tokenizer

Step 8: Model Export (ZIP)

zip -r phi-mcq-skill-exp-cot.zip final_model 

Purpose:

Easy sharing

Deployment-ready package

Step 9: Inference Logic (MCQ Generation)

Core function:

generate_n_mcqs(n, skill, experience)

Flow:

Build structured prompt

Call model.generate()

Parse MCQs using regex

Return clean output

-----------------------------------------------------------------------------------------------------------------

This exacts all the process for the DPO 

Notebook 2: MCQ_Generation_Unsloth_phi_COT-Streamlit.ipynb

Step 1: Additional Dependencies

pip install streamlit pyngrok 

Why?

streamlit → Web UI

pyngrok → Public URL for Colab

Step 2: Dataset Directory

os.makedirs('cot_dataset', exist_ok=False) 

Ensures consistency with training pipeline

Step 3: Streamlit App Creation (app.py)

Key components inside app.py:

1. Model Loading

FastLanguageModel.from_pretrained(MODEL_PATH) 

Loads the trained MCQ model

2. User Inputs

Streamlit widgets:

Skill (text input)

Experience (dropdown or text)

Number of MCQs

3. Prompt Construction

Dynamic prompt built from UI inputs:

Skill

Experience

MCQ count

Ensures same prompt format as training

4. MCQ Generation

model.generate(...) 

Includes:

Max tokens

Temperature

Top-p sampling

5. Output Parsing

Regex-based cleaning

Separates reasoning & final MCQs

Step 4: Streamlit App Execution

streamlit run app.py 

Runs UI on port 8501

Step 5: Public URL via Cloudflared

cloudflared tunnel --url http://localhost:8501 

Purpose:

Share UI with team/stakeholders

End-to-End Flow Summary

Prepare CoT MCQ dataset

Fine-tune Phi using Unsloth

Save & export model

Load model in Streamlit app

Generate MCQs interactively

What the Team Should Remember

Training notebook = model brain

Streamlit notebook = user interface

Prompt format must remain consistent

CoT improves reasoning quality

LoRA keeps training efficient

Next Possible Extensions

DPO or RLAIF on MCQ quality

Skill-wise adapters

Difficulty-level conditioning

MCQ validation using rule-based checks

This document is designed to be shared directly with the team.