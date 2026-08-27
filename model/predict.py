import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from src.models.fusion import MultiStreamDetector
from src.evaluation.explain import generate_heatmap, visualize_and_save

def main():
    import argparse
    parser = argparse.ArgumentParser("Predict and Explain a single image")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=False, help="Path to best_deepfake_detector.pth")
    parser.add_argument("--sensitivity", type=str, choices=['normal', 'high', 'ultra'], default='normal', help="Detection sensitivity")
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"File not found: {args.image_path}")
        return

    model = MultiStreamDetector(num_classes=2, freeze_backbone=False)
    if args.model_path and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        print(f"Loaded weights from {args.model_path}")
    else:
        print("Warning: Running with untrained weights for demonstration.")
        
    model.eval()
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    original_img = Image.open(args.image_path).convert('RGB')
    
    # --- FACE DETECTION & CROPPING ---
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        # Increase padding to 40% to catch hair and edge artifacts
        padding = int(0.4 * w)
        y1, y2 = max(0, y-padding), min(original_img.height, y+h+padding)
        x1, x2 = max(0, x-padding), min(original_img.width, x+w+padding)
        face_img = original_img.crop((x1, y1, x2, y2))
        
        # Forensic Enhancement: Histogram Equalization to reveal hidden artifacts
        from PIL import ImageOps
        face_img = ImageOps.autocontrast(face_img)
        
        # Subtle Sharpening to highlight hidden digital artifacts
        face_img = face_img.filter(ImageFilter.SHARPEN)
        print(f"Face detected and forensic contrast enhancement applied.")
    else:
        print("Warning: No face detected. Using full image (this may be inaccurate).")
        face_img = original_img

    input_tensor = transform(face_img).unsqueeze(0)
    
    with torch.no_grad():
        out = model(input_tensor)
        probs = torch.softmax(out, dim=1)
        prob_real = probs[0][0].item()
        prob_fake = probs[0][1].item()
        
    print(f"--- Debug Info ---")
    print(f"Raw Logits: {out.cpu().numpy()}")
    print(f"Probabilities: Real: {prob_real:.6f}, Fake: {prob_fake:.6f}")
    print(f"------------------")

    # --- SMART AUTO-CALIBRATION ---
    # Insight: Real images give very high logits (>10). 
    # High-quality fakes give lower logits (<5) because they confuse the model.
    logit_mag = torch.abs(out[0][0] - out[0][1]).item()
    
    # Auto-adjust threshold: If the model isn't extremely confident, lower the bar.
    if logit_mag < 8.0:
        threshold = 0.01 # Deep Analysis Mode
        mode = "Deep Scan"
    else:
        threshold = 0.5 # Normal Mode
        mode = "Standard"

    pred_label = "FAKE/ALTERED" if prob_fake > threshold else "REAL"
    print(f"--- Analysis Mode: {mode} ---")
    print(f"Confidence Level: {logit_mag:.2f}")
    print(f"Prediction: {pred_label} (Fake Prob: {prob_fake:.4f}, Threshold: {threshold})")
    
    print("Generating Grad-CAM heatmap...")
    # target_class = 1 (Fake) meaning "which regions make this image fake?"
    heatmap = generate_heatmap(model, input_tensor, face_img, target_class=1)
    
    out_name = f"analysis_{os.path.basename(args.image_path)}"
    visualize_and_save(heatmap, out_name)

if __name__ == "__main__":
    main()
