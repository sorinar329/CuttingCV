import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import matplotlib.pyplot as plt

processor = AutoProcessor.from_pretrained("facebook/metaclip-b32-400m")
model = AutoModel.from_pretrained("facebook/metaclip-b32-400m")

image = Image.open("/how-to-cut-a-peach-11.jpg")
inputs = processor(text=["peach with core", "peach core removed"], images=image, return_tensors="pt", padding=True)

with torch.no_grad():
  outputs = model(**inputs)
  logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
  text_probs = logits_per_image.softmax(dim=-1)

#print("Label probs:", text_probs)
# Convert probabilities to numpy array for plotting
probs = text_probs.cpu().numpy()

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plotting the image
ax1.imshow(image)
ax1.axis('off')
ax1.set_title('Image')

# Plotting the probabilities
labels = ["peach with core", "peach core removed"]
ax2.bar(labels, probs.flatten())
ax2.set_xlabel('Labels')
ax2.set_ylabel('Probabilities')
ax2.set_title('Label Probabilities')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()