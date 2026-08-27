import os
import argparse
import torch
from torch.utils.data import DataLoader
from src.data.dataset import DeepfakeDataset
from src.models.fusion import MultiStreamDetector
from src.training.trainer import train_model

def main():
    parser = argparse.ArgumentParser(description="Deepfake Image Detector Training")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing 'real' and 'altered' subfolders")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--fine_tune", action="store_true", help="Unfreeze backbone and use lower LR")
    args = parser.parse_args()

    print("Initializing DeepfakeDatasets for train and valid...")
    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'valid')
    
    train_dataset = DeepfakeDataset(root_dir=train_dir, phase='train')
    valid_dataset = DeepfakeDataset(root_dir=valid_dir, phase='valid')
    
    if len(train_dataset) == 0:
        print(f"Error: Train Dataset is empty. Check your data_dir path ({train_dir}).")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Loaded {len(train_dataset)} train images and {len(valid_dataset)} valid images.")
    
    if args.fine_tune:
        print("Initializing MultiStreamDetector for Fine-Tuning (unfrozen)...")
        model = MultiStreamDetector(num_classes=2, freeze_backbone=False)
        # Load the best model weights before fine-tuning
        if os.path.exists("best_deepfake_detector.pth"):
            print("Loading best_deepfake_detector.pth for fine-tuning...")
            checkpoint = torch.load("best_deepfake_detector.pth", weights_only=False)
            # Handle if the file is a state_dict or a dict with metadata
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        # Override LR for fine-tuning if it was still at default
        if args.lr == 1e-4:
            args.lr = 1e-5
            print(f"Fine-tuning mode: Automatically lowered LR to {args.lr}")
    else:
        print("Initializing MultiStreamDetector with PEFT (spatial frozen)...")
        model = MultiStreamDetector(num_classes=2, freeze_backbone=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Starting training...")
    train_model(model, train_loader, valid_loader, num_epochs=args.epochs, device=device, lr=args.lr)

if __name__ == "__main__":
    main()
