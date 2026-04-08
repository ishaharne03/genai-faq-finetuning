# OpenAI GPT-4o-mini Fine-Tuning Project

A hands-on project demonstrating how to fine-tune OpenAI's GPT-4o-mini model on a custom FAQ dataset using the OpenAI API.

**This project was created for TMLC Academy's GenAI Program FAQ chatbot**, enabling automated responses to frequently asked questions about the program.

## Overview

This project walks through the complete fine-tuning workflow:
1. Preparing a custom dataset (CSV to JSONL conversion)
2. Uploading training data to OpenAI
3. Creating and monitoring a fine-tuning job
4. Testing the fine-tuned model

## Dataset

The training data (`GenAI_Program_FAQ_dataset.csv`) contains 29 Q&A pairs about a Generative AI educational program, covering topics like:
- Program structure and duration
- Session timings and recordings
- Prerequisites and tools taught
- Projects and assignments
- Certificates and pricing

## Project Structure

```
fine-tuning-projects/
├── Finetune_OpenAI_A.ipynb       # Main Jupyter notebook with fine-tuning code
├── GenAI_Program_FAQ_dataset.csv # Training dataset (prompt-completion pairs)
└── README.md
```

## Prerequisites

- Python 3.8+
- OpenAI API key
- Basic understanding of Python and APIs

## Installation

```bash
pip install openai==1.55.3 httpx==0.27.2
```

## Usage

### 1. Set up your API key

```python
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
```

### 2. Prepare the dataset

Convert CSV to JSONL format required by OpenAI:

```python
import pandas as pd
import json

df = pd.read_csv('GenAI_Program_FAQ_dataset.csv')
with open('data.jsonl', 'w') as f:
    for _, row in df.iterrows():
        json_line = json.dumps({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": row['prompt']},
                {"role": "assistant", "content": row['completion']}
            ]
        })
        f.write(json_line + '\n')
```

### 3. Upload and fine-tune

```python
from openai import OpenAI
client = OpenAI()

# Upload file
file = client.files.create(file=open("data.jsonl", "rb"), purpose="fine-tune")

# Start fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini-2024-07-18"
)
```

### 4. Use the fine-tuned model

```python
completion = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:personal::YOUR_MODEL_ID",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How many projects are there in the program?"}
    ]
)
print(completion.choices[0].message.content)
```

## Key Concepts

- **Fine-tuning**: Customizing a pre-trained model on domain-specific data
- **JSONL Format**: Each line is a valid JSON object with a `messages` array
- **Supervised Fine-tuning**: Training using prompt-completion pairs

## Tools & Technologies

- OpenAI API
- GPT-4o-mini base model
- Python / Pandas
- Google Colab (optional)

## About TMLC Academy

This project was built for [TMLC Academy's Generative AI Program](https://www.tmlcacademy.in/genai) - a 6-week hands-on program covering LLMs, Fine-tuning, RAG, Agents, and LLMOps with tools like OpenAI, HuggingFace, LangChain, and more.

## License

This project is for educational purposes.
