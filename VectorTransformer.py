
import torch #This is equivalent to Keras on some levels, but different in how to use it
import torchvision.transforms as T #Part of the Torch package, used for vision ML

model = torch.hub.load('facebookresearch/dinov2','dinov2_vitb14') #Loading this specific version of DinoV2, the small model
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

def transformToVector(img): #Function to pack the image embedding into a simple package
    input_tensor = transform(img)[:3].unsqueeze(0)
    with torch.no_grad():
        embedding = model(input_tensor)
    return embedding.squeeze().tolist()

