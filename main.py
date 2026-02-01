import torch
import cv2
import numpy as np
import sys

def main():
    print(f"Python Version: {sys.version}")
    print("-" * 30)
    
    # Check OpenCV
    print(f"OpenCV Version: {cv2.__version__}")
    
    # Check PyTorch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    # Simple Tensor Operation
    x = torch.rand(5, 3)
    print("\nRandom Tensor (5x3):")
    print(x)
    
    print("\nSetup successful! You are ready for Deep Learning Computer Vision.")

if __name__ == "__main__":
    main()
