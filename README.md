# AI-Fashion RAG

**AI-Fashion RAG** is a **Retrieval-Augmented Generation (RAG)** system for fashion products that supports **text and image queries** for searching items and generating styling recommendations.  

This project demonstrates how **RAG pipelines** can power interactive, context-aware fashion search and personalized outfit suggestions.

---

## Table of Contents
- [Overview](#overview)  
- [Features](#features)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Results](#results)  

---

## Overview
AI-Fashion RAG is a **multimodal search system** that combines **text and image embeddings** to retrieve fashion items and generate styling advice.  

Key technologies:
- Python, Streamlit  
- LangChain  
- FAISS for vector search  
- Hugging Face `all-MiniLM-L6-v2` embeddings  
- CLIP for image-text similarity  
- Groq LLM for style recommendation generation  

---

## Features
- Search fashion items using **text or image queries**  
- Retrieve items from a dataset of 20K+ fashion products  
- Generate **styling recommendations** and outfit pairings  
- Perform **fast similarity search** using FAISS  
- Interactive **Streamlit web interface**  

---

## Dataset
**H&M Fashion Caption Dataset**  
- 20K+ fashion items with images and text descriptions  
- Dataset URL: [Hugging Face Dataset](https://huggingface.co/datasets/tomytjandra/h-and-m-fashion-caption)  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Jaswanth113/AI-Fashion-RAG.git
cd AI-Fashion-RAG
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your API key in a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

## Usage

Run the Streamlit app:

```bash
streamlit run assignment_fashion_rag.py
```

Or run with a query directly:

```bash
python assignment_fashion_rag.py --query "Beige wool coat for evening"
```

---

## Results
- Retrieves fashion items based on **text or image queries**  
- Provides **context-aware styling recommendations**  
- Supports **real-time interaction** via the web interface  
