from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
import torch
from torchvision import transforms
from PIL import Image
import io
from utils import preprocess, load_model
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Load the trained model
model = load_model()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def main():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Save original image
    original_path = "static/original.png"
    image.save(original_path)

    # Apply transformations
    preprocessed = preprocess(image)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        enhanced_image, params_maps = model(preprocessed)

    # Convert the output tensor to an image
    output_image = transforms.ToPILImage()(enhanced_image.squeeze())

    # Save the enhanced image
    enhanced_path = "static/enhanced.png"
    output_image.save(enhanced_path)

    return {"original_filename": original_path, "enhanced_filename": enhanced_path}

@app.get("/pipeline", response_class=HTMLResponse)
async def show_pipeline():
    with open("static/pipeline.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/pipeline-image")
async def get_pipeline_image():
    return FileResponse("static/pipeline.png")

@app.get("/network-image")
async def get_network_image():
    return FileResponse("static/network.png")

@app.get("/thank-you", response_class=HTMLResponse)
async def thank_you():
    with open("static/thank_you.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/pedro-gif")
async def get_pedro_gif():
    return FileResponse("static/pedro.gif")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
