export default {
  name: "EXP 4: Image Segmentation using Pre-trained DeepLabV3 Model",
  
  theory: `Image segmentation partitions an image into meaningful regions at the pixel level. 
Unlike classification, which assigns a single label to the entire image, segmentation assigns a label to each pixel, identifying object boundaries and separating foreground from background.

This experiment uses DeepLabV3 with a ResNet50 backbone, pre-trained on the Pascal VOC dataset. 
Each pixel is classified into one of the pre-defined object classes. 
A color map is applied to visualize segmented regions. 
The pre-trained model allows segmentation without training from scratch, directly applying to new images for testing and visualization.`,

  code: `
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load pre-trained DeepLabV3 model
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
model.eval()

# Define transform to convert image to tensor
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load image
image_path = "img.jpg"  # Replace with your image path
img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Run segmentation
with torch.no_grad():
    output = model(img_tensor)['out'][0]  # output shape: [num_classes, H, W]

# Get predicted class per pixel
pred = torch.argmax(output, dim=0).byte().cpu().numpy()

# Create a color map for visualization
num_classes = 21  # DeepLabV3 trained on 21 classes (VOC)
colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
seg_image = colors[pred]

# Display original image and segmentation
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
ax1.imshow(img)
ax1.set_title("Original Image")
ax1.axis("off")

ax2.imshow(seg_image)
ax2.set_title("Segmented Image")
ax2.axis("off")

plt.show()
  `,

  algorithm: `
1. Load a pre-trained DeepLabV3 model with ResNet50 backbone
2. Define image transform to convert input image to tensor
3. Load and preprocess input image
4. Run model inference to get per-pixel class probabilities
5. Compute the predicted class for each pixel using argmax
6. Map predicted classes to colors for visualization
7. Display original and segmented images side by side
  `,

  viva: [
    {
      q: "What is the difference between image classification and image segmentation?",
      a: "Image classification assigns a single label to an entire image, indicating the main object or scene. In contrast, image segmentation assigns a label to every pixel, separating objects from the background and distinguishing multiple regions within the image. Segmentation provides spatial information and is critical in tasks such as autonomous driving, medical imaging, and object detection. Classification might tell you 'this is a cat', while segmentation tells you 'these pixels belong to the cat, these belong to the background', providing precise object localization."
    },
    {
      q: "Why do we use a pre-trained DeepLabV3 model instead of training from scratch?",
      a: "Pre-trained models like DeepLabV3 are trained on large datasets such as Pascal VOC, learning robust feature representations. Using a pre-trained model saves time and computational resources since training from scratch requires vast amounts of labeled data and processing power. It also improves accuracy, especially for small datasets. Transfer learning allows the model to adapt pre-learned knowledge to new images quickly, achieving good segmentation results without the need for extensive retraining."
    },
    {
      q: "What is the purpose of applying a color map in image segmentation?",
      a: "Applying a color map helps visualize the segmented regions clearly. Each class label is mapped to a unique color, making it easy to distinguish different objects or areas in the image. Without a color map, segmentation results would appear as numeric arrays, which are difficult to interpret. The color-coded visualization aids in evaluating model performance, debugging, and presenting results effectively, allowing humans to quickly understand which pixels belong to which object class."
    }
  ]
};
