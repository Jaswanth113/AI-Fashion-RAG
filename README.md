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

## Results
- Retrieves fashion items based on **text or image queries**  
- Provides **context-aware styling recommendations**  
- Supports **real-time interaction** via the web interface  
