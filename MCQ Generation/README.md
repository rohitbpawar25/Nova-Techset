# **MCQ Generation using Unsloth Phi + Chain-of-Thought (CoT)**

## **Purpose of This Document**

This document explains **both notebooks step by step** so the team can:

* Understand the **training pipeline** (fine-tuning Phi using Unsloth + CoT)
* Understand the **inference pipeline** (MCQ generation)
* Understand how the **Streamlit UI** is built on top of the trained model

The goal is that **any team member can read this and reproduce or extend the system**.

## **High-Level Architecture**

1. **Dataset Preparation (CoT-based MCQs)**
2. **Model Fine-Tuning (Unsloth + TRL)**
3. **Model Saving & Export**
4. **Inference Pipeline (MCQ Generation)**
5. **Streamlit Application (UI Layer)**

## **Notebook 1: MCQ\_Generation\_Unsloth\_phi\_COT.ipynb**

### **Step 1: Environment Setup**

pip install unsloth accelerate bitsandbytes transformers datasets trl

**Why?**

* unsloth → Memory-efficient LLM fine-tuning
* trl → Trainer utilities for instruction/CoT tuning
* bitsandbytes → 4-bit / 8-bit quantization

### **Step 2: Dataset Folder Creation**

os.mkdir('cot\_dataset')

**Purpose:**

* Central directory for Chain-of-Thought MCQ training data

### **Step 3: Dataset Loading & Formatting**

* JSON-based MCQ samples
* Each record contains:
  + instruction
  + input (skill + experience)
  + output (MCQs with reasoning)

**Key Idea:**

* The model is trained to **reason before answering** (CoT)

Example (conceptual):

Instruction: Generate MCQs  
Input: Python, 3–5 years  
Output:  
Reasoning: ...  
Q1: ...

### **Step 4: Model Initialization (Unsloth)**

FastLanguageModel.from\_pretrained(  
 model\_name="phi",  
 load\_in\_4bit=True,  
 max\_seq\_length=2048  
)

**Why Unsloth?**

* Faster training
* Lower GPU memory usage
* Optimized for instruction tuning

### **Step 5: LoRA Configuration**

* Low-Rank Adaptation (LoRA)
* Trains only **small adapter layers**

**Benefits:**

* Faster training
* Less overfitting
* Easy model versioning

### **Step 6: Trainer Setup (TRL)**

Key parameters:

* Epochs
* Batch size
* Learning rate
* Gradient accumulation

trainer.train()

**Output:**

* Fine-tuned Phi model with MCQ + reasoning capability

### **Step 7: Model Saving**

model.save\_pretrained("final\_model")

**Saved Artifacts:**

* Base model weights
* LoRA adapters
* Tokenizer

### **Step 8: Model Export (ZIP)**

zip -r phi-mcq-skill-exp-cot.zip final\_model

**Purpose:**

* Easy sharing
* Deployment-ready package

### **Step 9: Inference Logic (MCQ Generation)**

Core function:

* generate\_n\_mcqs(n, skill, experience)

**Flow:**

1. Build structured prompt
2. Call model.generate()
3. Parse MCQs using regex
4. Return clean output

-----------------------------------------------------------------------------------------------------------------

## **This exacts all the process for the DPO**

## **Notebook 2: MCQ\_Generation\_Unsloth\_phi\_COT-Streamlit.ipynb**

### **Step 1: Additional Dependencies**

pip install streamlit pyngrok

**Why?**

* streamlit → Web UI
* pyngrok → Public URL for Colab

### **Step 2: Dataset Directory**

os.makedirs('cot\_dataset', exist\_ok=False)

Ensures consistency with training pipeline

### **Step 3: Streamlit App Creation (app.py)**

Key components inside app.py:

#### **1. Model Loading**

FastLanguageModel.from\_pretrained(MODEL\_PATH)

Loads the **trained MCQ model**

#### **2. User Inputs**

Streamlit widgets:

* Skill (text input)
* Experience (dropdown or text)
* Number of MCQs

#### **3. Prompt Construction**

Dynamic prompt built from UI inputs:

* Skill
* Experience
* MCQ count

Ensures **same prompt format as training**

#### **4. MCQ Generation**

model.generate(...)

Includes:

* Max tokens
* Temperature
* Top-p sampling

#### **5. Output Parsing**

* Regex-based cleaning
* Separates reasoning & final MCQs

### **Step 4: Streamlit App Execution**

streamlit run app.py

Runs UI on port 8501

### **Step 5: Public URL via Cloudflared**

cloudflared tunnel --url <http://localhost:8501>

**Purpose:**

* Share UI with team/stakeholders

## **End-to-End Flow Summary**

1. Prepare CoT MCQ dataset
2. Fine-tune Phi using Unsloth
3. Save & export model
4. Load model in Streamlit app
5. Generate MCQs interactively

## **What the Team Should Remember**

* **Training notebook = model brain**
* **Streamlit notebook = user interface**
* Prompt format **must remain consistent**
* CoT improves reasoning quality
* LoRA keeps training efficient

## **Next Possible Extensions**

* DPO or RLAIF on MCQ quality
* Skill-wise adapters
* Difficulty-level conditioning
* MCQ validation using rule-based checks

**This document is designed to be shared directly with the team.**