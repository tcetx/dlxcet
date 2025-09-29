export default {
  name: "EXP 3: Object Detection using Pre-trained Faster R-CNN",

  theory: `Object detection identifies and localizes multiple objects in an image. 
This experiment uses a pre-trained Faster R-CNN model from torchvision, trained on the COCO dataset with 80 object categories.
The model predicts bounding boxes, class labels, and confidence scores for each detected object.
Only high-confidence predictions are visualized on the image.`,

  code: `
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# COCO class labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Image transform
transform = transforms.Compose([transforms.ToTensor()])

# Load input image
image_path = "image_path.jpg"  # replace with your image path
img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0)  # add batch dimension

# Run detection
with torch.no_grad():
    predictions = model(img_tensor)

# Extract boxes, labels, and scores
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']

# Plot results
fig, ax = plt.subplots(1, figsize=(12,9))
ax.imshow(img)

threshold = 0.7
for i, box in enumerate(boxes):
    if scores[i] >= threshold:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if labels[i] < len(COCO_INSTANCE_CATEGORY_NAMES):
            label = f"{COCO_INSTANCE_CATEGORY_NAMES[labels[i]]}: {scores[i]:.2f}"
            ax.text(x1, y1, label, fontsize=10, color='white',
                    bbox=dict(facecolor='red', alpha=0.5))
        else:
            print(f"Warning: Predicted label index {labels[i]} is out of range.")

plt.axis("off")
plt.show()
  `,

  algorithm: `
1. Load pre-trained Faster R-CNN model
2. Prepare input image (convert to tensor and add batch dimension)
3. Run model inference to get bounding boxes, labels, and scores
4. Filter predictions using a confidence threshold
5. Visualize bounding boxes and labels on the image
  `,

  viva: [
    {
      q: "Why use a pre-trained model for object detection?",
      a: "It saves time and resources as the model has already learned features from a large dataset (COCO) and can generalize to new images."
    },
    {
      q: "What does the confidence score represent?",
      a: "It indicates how likely the predicted object belongs to the given class; higher scores mean higher certainty."
    },
    {
      q: "Why apply a threshold on confidence scores?",
      a: "To ignore low-confidence predictions and reduce false positives in the visualization."
    },
    {
      q: "What is Faster R-CNN?",
      a: "Faster R-CNN is an object detection model that uses a Region Proposal Network (RPN) to quickly propose object locations and then classifies them."
    }
  ]
};
