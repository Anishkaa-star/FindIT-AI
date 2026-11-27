
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

model_name = "openai/clip-vit-base-patch32"

print("Loading CLIP...")

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()

print("CLIP is ready!")

# Load two images
image1 = Image.open("image1.jpeg").convert("RGB")
image2 = Image.open("image2.jpeg").convert("RGB")

# Prepare images
inputs = processor(
    images=[image1, image2],
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

# Compare images
with torch.no_grad():
    outputs = model.vision_model(
        **{
            "pixel_values": inputs["pixel_values"]
        }
    )

    features = model.visual_projection(
        outputs.pooler_output
    )

    features = torch.nn.functional.normalize(
        features, p=2, dim=-1
    )

similarity = torch.dot(
    features[0], features[1]
)

print("Image similarity:", round(similarity.item() * 100, 2), "%")
