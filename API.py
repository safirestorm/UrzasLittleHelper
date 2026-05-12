from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageOps
#from qdrant_search import makeQuery
import io
import cv2

from image_recognizer import getWarpedImage, getAnnotatedImage

app = FastAPI()

# Your requested processing function
def process_image_logic(image) -> str:
    print("NOT READY")
    # Imagine some AI or filtering logic happens here
    #result = makeQuery(image)

    #return result


@app.post("/getId/")
async def upload_image(file: UploadFile = File(...)):
    # 1. Read the file content as bytes
    contents = await file.read()

    # Wrap bytes in BytesIO and open with PIL
    image = Image.open(io.BytesIO(contents))
    image = ImageOps.exif_transpose(image)


    # 2. Run your processing function
    result = process_image_logic(image)

    return {"ScryfallID": result}


@app.post("/annotate-image/")
async def annotate_image(file: UploadFile = File(...)):
    # 1. Read and Open the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = ImageOps.exif_transpose(image)

    # 2: transform the image
    warpedArray = getAnnotatedImage(image)

    if warpedArray is None:
        print("ERROR")

    rgb_image = cv2.cvtColor(warpedArray, cv2.COLOR_BGR2RGB)
    final_pil_image = Image.fromarray(rgb_image)

    # 3. Save the result to an in-memory buffer
    buffer = io.BytesIO()

    # You must specify the format (PNG, JPEG, etc.)
    final_pil_image.save(buffer, format="PNG")


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
    warpedArray = getWarpedImage(image)

    if warpedArray is None:
        print("ERROR")

    rgb_image = cv2.cvtColor(warpedArray, cv2.COLOR_BGR2RGB)
    final_pil_image = Image.fromarray(rgb_image)

    # 3. Save the result to an in-memory buffer
    buffer = io.BytesIO()

    # You must specify the format (PNG, JPEG, etc.)
    final_pil_image.save(buffer, format="PNG")


    # 4. Seek to the start of the buffer so FastAPI can read it
    buffer.seek(0)

    # 5. Return the image stream
    return StreamingResponse(buffer, media_type="image/png")