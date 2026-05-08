#Import pyTorch and TorchVision
from time import sleep

import torchvision.transforms as T
import torch

model = torch.hub.load('facebookresearch/dinov2','dinov2_vits14') #Loading this specific version of DinoV2, the small model
model.eval()  #Setting to inference mode - We are not training, no randomness. Deterministic outcomes ONLY

def pad_to_square(image): #function to add padding to the image, rather than crop
    width, height = image.size
    max_side = max(width, height)
    # Calculate how much padding we need for each side
    left = (max_side - width) // 2
    top = (max_side - height) // 2
    right = max_side - width - left
    bottom = max_side - height - top
    # Pad with 0 (black)
    return T.functional.pad(image, (left, top, right, bottom), fill=0)

transform = T.Compose([ #Creating a transform for the images, so they fit with what the model needs
    T.Lambda(pad_to_square), # Use our custom padding function first
    T.Resize((224, 224)),    # Now resize the square to the model's size
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def transformToVector(image): #Function to pack the image embedding into a simple package
    input_tensor = transform(img)[:3].unsqueeze(0)
    with torch.no_grad():
        embedding = model(input_tensor)
    return embedding


from PIL import Image #A way to import images
from pathlib import Path

database_path = Path("mtgDir")
valid_extensions = ('.jpg', '.png', 'jpeg')
import torch.nn.functional as F
import timeit

# Store vectors in a list to compare them
vectors = []
filenames = []

for file_path in database_path.glob('*'):
    if file_path.suffix.lower() in valid_extensions:
        start = timeit.default_timer()
        img = Image.open(file_path).convert('RGB')
        vec = transformToVector(img)
        end = timeit.default_timer()
        print(vec.shape)
        print(end - start)
        vectors.append(vec)
        filenames.append(file_path.name)

# Let's compare the first image (index 0) to all images in the list
first_vec = vectors[0]

print(f"Comparing everything to: {filenames[0]}\n")
for i in range(len(vectors)):
    # F.cosine_similarity returns a value between -1 and 1
    score = F.cosine_similarity(first_vec, vectors[i])
    print(f"{filenames[i]}: Similarity Score = {score.item():.4f}")




