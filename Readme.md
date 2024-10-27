# MailGuard AI

## Overview
**MailGuard AI** is an intelligent spam detection system designed to identify and classify emails as either spam or not spam (ham). Utilizing a machine learning model with **BERT-tiny** for tokenization, the application helps users efficiently filter unwanted emails, ensuring only relevant communication is prioritized.

## Technologies Used
- **Language Model & Tokenization:** BERT-tiny
- **Machine Learning Library:** ONNX Runtime (for running the model)
- **Backend Framework:** FastAPI
- **Frontend Technologies:** 
  - React
  - HTML
  - CSS
  - JavaScript
  - Python
  - 

## Features
- **Spam Detection**: Uses a machine learning model to classify emails as "Spam" or "Ham".
- **Interactive User Interface**: A user-friendly interface to enter email content and view classification results.
- **Search History**: Stores previous search results for easy reference.
- **Real-Time Inference**: Fast processing of email content using ONNX Runtime for real-time feedback.
- **Accuracy**: The model achieved an accuracy of **0.92** during training.

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Arhaans/MailGuard-AI.git
   ```
2\) Navigate to the project directory
```bash
  cd hackathon
```
3\) Create a virtual environment
``` bash
conda create --name testenv(environment name)
```

4\) Activate the virtual environment
```bash
conda activate testenv
```
5\) Install the required dependecies mentioned in the requirements.txt

```bash
pip install -r requirements.txt
```


