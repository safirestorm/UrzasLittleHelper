from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageOps
import io

app = FastAPI()

# Your requested processing function
def process_image_logic(image) -> str:
    # Imagine some AI or filtering logic happens here

    return {"format": image.format}


@app.post("/getId/")
async def upload_image(file: UploadFile = File(...)):
    # 1. Read the file content as bytes
    contents = await file.read()

    # Wrap bytes in BytesIO and open with PIL
    image = Image.open(io.BytesIO(contents))

    # 2. Run your processing function
    status = process_image_logic(image)

    return status

@app.post("/get-points/")
async def get_points(file: UploadFile = File(...)):
    # 1. Read and Open the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = ImageOps.exif_transpose(image)

    # 2. Transform the image:


    # 3. Save the result to an in-memory buffer
    buffer = io.BytesIO()
    # You must specify the format (PNG, JPEG, etc.)
    image.save(buffer, format="PNG")

    # 4. Seek to the start of the buffer so FastAPI can read it
    buffer.seek(0)

    # 5. Return the image stream
    return StreamingResponse(buffer, media_type="image/png")


@app.post("/transform-image/")
async def transform_image(file: UploadFile = File(...)):
    # 1. Read and Open the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = ImageOps.exif_transpose(image)

    # 2: transform the image

    # 3. Save the result to an in-memory buffer
    buffer = io.BytesIO()
    # You must specify the format (PNG, JPEG, etc.)
    image.save(buffer, format="PNG")

    # 4. Seek to the start of the buffer so FastAPI can read it
    buffer.seek(0)

    # 5. Return the image stream
    return StreamingResponse(buffer, media_type="image/png")