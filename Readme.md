# MailGuard AI

## Overview
**MailGuard AI** is an intelligent spam detection system designed to identify and classify emails as either spam or not spam (ham). Utilizing a machine learning model with **BERT-tiny** for tokenization, the application helps users efficiently filter unwanted emails, ensuring only relevant communication is prioritized.

## Technologies Used
- **Language Model & Tokenization:** BERT-tiny
- **Machine Learning Library:** ONNX Runtime (for running the model)
- **Machine Learning Technologies:**
  - TensorFlow
  - PyTorch
  - Transformers
  - Scikit-learn


- **Backend Framework:** FastAPI
- **Frontend Technologies:** 
  - React
  - HTML
  - CSS
  - JavaScript

### Model Performance
Below is the model's training performance, showing training loss per epoch and accuracy metrics:

![PHOTO-2024-10-27-09-48-36](https://github.com/user-attachments/assets/d6d5d296-4f47-48e3-9350-2640fcb2338b)


## Features
- **Spam Detection**: Uses a machine learning model to classify emails as "Spam" or "Ham".
- **Interactive User Interface**: A user-friendly interface to enter email content and view classification results.
- **Search History**: Stores previous search results for easy reference.
- **Real-Time Inference**: Fast processing of email content using ONNX Runtime for real-time feedback.
- **Accuracy**: The model achieved an accuracy of **0.92** during training.

## Installation
1. **Clone the repository**:
     ```
     bash
     git clone https://github.com/Arhaans/MailGuard-AI.git
     ```
2. **Navigate to the project directory:**
    ```
    cd MailGuard-AI
    ```
3. **Create a virtual environment:**
    ```
    conda create --name mailguard-env
     ```
4. **Activate the virtual environment:**
    ```
    pip install -r requirements.txt
     ```
5. **Install the required dependencies:**
    ``` 
    pip install -r requirements.txt
    ```

## Usage
1. **Start the backend server:**

- Run the following command to start the FastAPI server:
bash
    ```
    uvicorn main:app --reload
    ```
- The backend will be accessible at http://127.0.0.1:8000.
2. **Start the frontend:**

- Ensure you are in the frontend directory (if separate).
- Run the frontend using a command like:

    ```
    npm start
    ```
- Open http://localhost:3000 in your web browser to interact with the MailGuard AI application.
3. **Using the Application:**

- Enter the email content in the provided input field.
- Click "Search" to classify the email as "Spam" or "Ham".
- View your search history and delete or clear entries as needed.
## Future Work
- Advanced Features:
  - Implement user authentication for personalized histories.
  - Add options to save and export chat histories.
- Enhanced Model Capabilities:
  - Incorporate a more complex language model for even more accurate spam detection.
  - Enable customization for individual spam preferences.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with any improvements or suggestions you may have.

## License
This project is licensed under the MIT License. See the LICENSE file for more information.

## Acknowledgements
- BERT-tiny for providing the foundation for tokenization and language processing.
- ONNX Runtime for enabling efficient model inference.
- React and FastAPI for the frontend and backend frameworks.
