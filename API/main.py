from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
from utils import (
    remove_shadow_opencv,
    convert_pdf_to_image,
    pil_to_bytes,
    extract_text_with_gemini
)

from PIL import Image

app = FastAPI()

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    filename = file.filename
    temp_path = f"temp_{filename}"

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    if filename.lower().endswith(".pdf"):
        image = convert_pdf_to_image(temp_path)
    else:
        image = remove_shadow_opencv(temp_path)

    image_bytes = pil_to_bytes(image)
    text = extract_text_with_gemini(image_bytes)

    os.remove(temp_path)
    return JSONResponse(content={"extracted_text": text})
