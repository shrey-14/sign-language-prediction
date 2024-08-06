import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import pickle

# Load the saved model
model = load_model("ASL_CNN_model.h5")

with open('label_list.pkl', 'rb') as f:
    index_to_label = pickle.load(f)

IMG_HEIGHT = 128
IMG_WIDTH = 128

def predict_sign_language(img):
    # Preprocess the image
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if needed

    # Make predictions
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])  # Get the index with the highest probability
    predicted_label = index_to_label[predicted_index]

    return predicted_label

# Create a Gradio interface
iface = gr.Interface(
    fn=predict_sign_language,
    inputs=gr.Image(type="pil", label="Input Image"),  # Use 'pil' type to get a PIL image
    outputs=gr.Textbox(label="Predicted Label"),
    title="Sign Language Predictor",
    description="<div style='text-align: center;'>Upload an image and the model will predict the sign language.</div>",
    examples=[
        ["l-sign-language-fd31b4ca-e8746b6c-640w.webp"],
        ["lvb-cornuto5.webp"],
        ["sign-language-hand-showing-c-260nw-740041291.webp"]
    ]
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
