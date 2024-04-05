from ultralytics import YOLO

# Initialize a YOLO-World model
model = YOLO('../weights/yolov8x-world.pt')  # or select yolov8m/l-world.pt

# Define custom classes
#model.set_classes(["dog", "cat", "peach"])

# Save the model with the defined offline vocabulary
# model.save("custom_yolov8s.pt")

# Load your custom model
# model = YOLO('custom_yolov8s.pt')

# Run inference to detect your custom classes
results = model.predict('C:\Dev\CuttingCV\watermelon-pineapple-oranges-cut-into-pieces-with-avocado-broccoli-apples-wood-table-top-view.jpg')

# Show results
results[0].show()