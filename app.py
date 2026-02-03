from fastapi import FastAPI, UploadFile, File
from PIL import Image
import tensorflow as tf
import numpy as np
import io

# ======================================================
# LOAD CNN MODEL (GOOD / BREAKAGE) ✅
# ======================================================
CNN_MODEL_PATH = "o_ring_cnn_classifier (1).keras"
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
_, H, W, C = cnn_model.input_shape

CNN_CLASSES = ["breakage", "good"]

print("✅ CNN model loaded")

# ======================================================
# FASTAPI APP
# ======================================================
app = FastAPI(title="O-Ring CNN Inference API")

# ======================================================
# CNN PREDICTION
# ======================================================
def predict_cnn(image: Image.Image):
    if C == 1:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    image = image.resize((W, H))
    img = np.array(image, dtype=np.float32) / 255.0

    if C == 1:
        img = np.expand_dims(img, axis=-1)

    img = np.expand_dims(img, axis=0)

    preds = cnn_model.predict(img)[0]
    idx = int(np.argmax(preds))

    return {
        "model": "CNN Classifier",
        "prediction": CNN_CLASSES[idx],
        "confidence": float(np.max(preds)),
        "probabilities": {
            CNN_CLASSES[i]: float(preds[i])
            for i in range(len(CNN_CLASSES))
        }
    }

# ======================================================
# API ENDPOINT
# ======================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes))

    return predict_cnn(image)

# ======================================================
# HEALTH CHECK (IMPORTANT FOR RENDER)
# ======================================================
@app.get("/")
def health():
    return {"status": "ok"}
