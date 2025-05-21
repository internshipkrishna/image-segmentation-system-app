
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your pre-trained model
model = tf.keras.models.load_model("segmentation_model.h5")  # or segmentation_model.keras

def segment(image):
    image = image.resize((256, 256))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, 0)
    
    prediction = model.predict(img_array)[0]
    mask = (prediction.squeeze() > 0.5).astype(np.uint8) * 255
    return Image.fromarray(mask)

# Define Gradio interface
interface = gr.Interface(
    fn=segment,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Disaster Image Segmentation",
    description="Upload satellite/drone image to visualize disaster segmentation."
)

interface.launch()
