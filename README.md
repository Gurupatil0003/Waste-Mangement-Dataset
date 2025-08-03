https://drive.google.com/file/d/1gTERu30UDDs7784VQ4YOXrNCStgJYfpI/view?usp=sharing






```python
import gradio as gr
import tensorflow.keras.utils as ku
import numpy as np
from tensorflow.keras.models import load_model

# Load your model
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Core classification logic
def classify_image(image):
    img = ku.array_to_img(image).resize((300, 300))
    img_array = ku.img_to_array(img, dtype=np.uint8) / 255.0
    prediction = model.predict(img_array[np.newaxis, ...])[0]

    predicted_class = labels[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    class_confidence = {label: float(prediction[i]) for i, label in enumerate(labels)}

    return predicted_class, f"{confidence * 100:.2f}%", class_confidence

# Fancy Gradio App
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div style='text-align: center;'>
            <h1 style='font-size: 2.5em; color: #4A90E2;'>🧠 Trash Classifier</h1>
            <p style='font-size: 1.2em;'>Upload an image of trash, and let our AI sort it into categories!</p>
            <p style='font-size: 1em; color: gray;'>Built with TensorFlow + Gradio</p>
        </div>
        """,
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="📸 Upload Trash Image", type="numpy", image_mode="RGB")
            submit_button = gr.Button("🚀 Classify", elem_id="submit-button")
        with gr.Column(scale=2):
            result_label = gr.Label(label="🔖 Predicted Class")
            confidence_text = gr.Text(label="📊 Confidence of Top Class")
            confidence_bar = gr.HighlightedText(label="📌 Confidence Breakdown")

    # Logic for updating UI
    def predict_and_format(image):
        predicted_class, confidence, class_confidence = classify_image(image)
        formatted = [(label, f"{conf:.2%}") for label, conf in class_confidence.items()]
        highlighted = [(label, conf, "highlight" if label == predicted_class else None) for label, conf in formatted]
        return predicted_class, confidence, highlighted

    submit_button.click(
        fn=predict_and_format,
        inputs=image_input,
        outputs=[result_label, confidence_text, confidence_bar]
    )

    gr.Markdown(
        """
        <div style='text-align: center; margin-top: 30px;'>
            <p style='font-size: 1em; color: #999;'>🛠️ Tip: Use clear images for better classification accuracy</p>
            <p style='font-size: 0.9em; color: #ccc;'>© 2025 Trash Classifier AI</p>
        </div>
        """,
    )

demo.launch()

```


```python
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os
from uuid import uuid4

app = FastAPI()

# Mount static directories
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

# Load the model
model = load_model("modelnew.h5")

# Labels
labels = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash"
}

# Ensure uploads folder exists
os.makedirs("uploads", exist_ok=True)

# Preprocessing function
def preprocess(img: Image.Image):
    img = img.resize((300, 300))
    img_array = np.array(img, dtype='uint8') / 255.0
    return img_array[np.newaxis, ...]

# Route to render the form using a Jinja2 template
@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction route
@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    processed_img = preprocess(img)

    prediction = model.predict(processed_img)
    pred_index = int(np.argmax(prediction[0]))
    pred_class = labels[pred_index]
    probability = round(float(np.max(prediction[0])) * 100, 2)

    # Save uploaded image
    filename = f"{uuid4().hex}_{file.filename}"
    filepath = os.path.join("uploads", filename)
    with open(filepath, "wb") as buffer:
        buffer.write(contents)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "predicted_class": pred_class,
        "probability": probability,
        "image_url": f"/uploads/{filename}"
    })


```
