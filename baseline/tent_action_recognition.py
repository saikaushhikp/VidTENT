import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.video import r3d_18, R3D_18_Weights

# ==========================================
# 1. TENT Core Functions
# ==========================================

def configure_tent_model(model):
    """
    Configures the model for TENT:
    1. Sets model to train mode (crucial for updating BatchNorm stats).
    2. Freezes all parameters.
    3. Unfreezes ONLY the affine parameters of normalization layers.
    """
    model.train() # TENT requires train mode to track batch statistics
    model.requires_grad_(False) # Freeze everything first
    
    # Identify and unfreeze 3D BatchNorm layers
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.requires_grad_(True)
            # Ensure weight (scale) and bias (shift) are updatable
            if m.weight is not None:
                m.weight.requires_grad_(True)
            if m.bias is not None:
                m.bias.requires_grad_(True)
                
    return model

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Calculates the Shannon entropy of the model's predictions."""
    # Entropy formula: H(y) = - sum(y * log(y))
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

# ==========================================
# 2. UCF50 DataLoader (Skeleton)
# ==========================================

def get_ucf50_dataloader(data_dir, batch_size=8):
    """
    Dummy dataloader setup. In a real scenario, you would use 
    torchvision.datasets.VideoClips or a custom PyTorch Dataset 
    to read the .avi files from the UCF50 directory structure.
    """
    print(f"Loading UCF50 data from {data_dir}...")
    
    # Placeholder for a real video dataset
    # Expects tensors of shape: (Batch, Channels, Frames, Height, Width)
    dummy_data = torch.randn(batch_size, 3, 16, 112, 112) 
    dummy_labels = torch.randint(0, 50, (batch_size,))
    
    # Yielding a single batch for demonstration
    yield dummy_data, dummy_labels

# ==========================================
# 3. Main Evaluation Loop
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Action Recognition with/without TENT")
    parser.add_argument('--data_dir', type=str, default='./UCF50', help='Path to UCF50 dataset')
    parser.add_argument('--use_tent', action='store_true', help='Flag to enable TENT adaptation')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for TENT optimizer')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained video model (e.g., Kinetics-400 pre-trained)
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    
    # Adjust final classification layer for UCF50 (50 classes instead of 400)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 50)
    model = model.to(device)

    # Initialize TENT if flag is passed
    if args.use_tent:
        print("--- TENT Adaptation is ENABLED ---")
        model = configure_tent_model(model)
        
        # Collect only the trainable parameters (the un-frozen BatchNorm parameters)
        params, param_names = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params.append(param)
                param_names.append(name)
        
        # Use a small learning rate for adaptation
        optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    else:
        print("--- Standard Inference (TENT DISABLED) ---")
        model.eval() # Standard static inference mode

    # Dataloader
    dataloader = get_ucf50_dataloader(args.data_dir)

    correct = 0
    total = 0

    # Inference Loop
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        if args.use_tent:
            # 1. Forward pass
            outputs = model(inputs)
            
            # 2. Calculate Entropy Loss
            loss = softmax_entropy(outputs).mean(0) 
            
            # 3. Backward pass & Update Normalization parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Note: In standard TENT, the prediction used is the one generated 
            # *during* the forward pass that computed the entropy. 
            predictions = outputs.argmax(dim=1)
            
        else:
            # Standard forward pass without gradients
            with torch.no_grad():
                outputs = model(inputs)
                predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()
        
        print(f"Batch Processed. Current Accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    main()