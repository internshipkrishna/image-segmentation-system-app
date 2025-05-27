import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("segmentation_model.h5")

def preprocess_image(img):
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predict(img):
    input_img = preprocess_image(img)
    mask = model.predict(input_img)[0]
    predicted_label = "flooded" if np.mean(mask) > 0.2 else "not flooded"

    actual_label = "flooded"  # Replace with ground truth source if needed

    result = f"🔮 Predicted Label: {predicted_label}\n"
    if predicted_label == actual_label:
        result += f"✅ Actual Label   : {actual_label}"
    else:
        result += f"❌ Actual Label   : {actual_label}"

    return mask, result

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="numpy", label="Segmentation Mask"),
        gr.Textbox(label="Flood Condition Output")
    ],
    title="Disaster Image Segmentation",
    description="Upload a satellite/drone image to detect and label flooded areas."
)

demo.launch()
