# Image Segmentation for Disaster Resilience

This project uses a trained UNet model for segmenting satellite/drone images in disaster scenarios. The app is built using Gradio and TensorFlow.

## 🚀 Run on GitHub Codespaces

1. Click the green **"Code"** button and select **"Open in Codespaces"**.
2. In the terminal, install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

Then open the forwarded port to use the Gradio interface.

## 📂 Files

- `app.py` - Gradio app for image segmentation.
- `segmentation_model.h5` - Pre-trained UNet model.
- `requirements.txt` - Python dependencies.