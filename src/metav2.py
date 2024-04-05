import open_clip
model, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
import torch
from PIL import Image

image = Image.open("/peach_cutv2.jpg")
image = preprocess(image).unsqueeze(0)
text = tokenizer(["peach full", "peach half", "peach with core", "peach without core"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)