import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

def generate_heatmap(model, input_tensor, original_image_rgb, target_class=1):
    """
    Generates a Grad-CAM heatmap over the spatial stream of our MultiStreamDetector.
    We target the last convolutional block of the EfficientNet backbone to see what parts 
    of the image influenced the fake detection probability.
    """
    # Note: Depending on the timm backbone, the target layer might vary. 
    # For tf_efficientnet_b4_ns, blocks[-1] is typically correct.
    target_layers = [model.spatial.backbone.blocks[-1]]
    
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Target the 'fake' class (index 1)
    targets = [ClassifierOutputTarget(target_class)]
    
    # Generate the heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    # Normalize original image for overlay
    original_np = np.array(original_image_rgb, dtype=np.float32) / 255.0
    
    # Resize if original image is not implicitly 224x224
    if original_np.shape[:2] != (224, 224):
        img_pil = Image.fromarray(np.uint8(original_image_rgb)).resize((224, 224))
        original_np = np.array(img_pil, dtype=np.float32) / 255.0
        
    visualization = show_cam_on_image(original_np, grayscale_cam, use_rgb=True)
    return visualization

def visualize_and_save(visualization, save_path="heatmap.jpg"):
    img = Image.fromarray(visualization)
    img.save(save_path)
    print(f"Saved Grad-CAM heatmap to {save_path}")
