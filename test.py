# Import the necessary library
import aivm_client as aic

# Define the model name and file path
MODEL_NAME = "bert_spam_detection"
onnx_file_path = "C:/Users/arhaa/Desktop/hackathon/bert_spam_detection.onnx"

# Upload the model to the AIVM server
aic.upload_bert_tiny_model(onnx_file_path, MODEL_NAME)
