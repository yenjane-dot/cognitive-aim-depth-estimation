# Model Checkpoints

This directory contains the pre-trained model weights for the Cognitive-Aim Depth Estimation model.

## Download Pre-trained Model

The pre-trained model file `cognitive_aim_model.pth` (568MB) is available from:

### Hugging Face Hub
**Link:** [Download cognitive_aim_model.pth](https://huggingface.co/yenjane123/cognitive_aim1.0/tree/main)

## Installation

1. Download the model file from any of the above sources
2. Place the downloaded `cognitive_aim_model.pth` file in this directory
3. The final structure should be:
   ```
   checkpoints/
   ├── README.md
   └── cognitive_aim_model.pth
   ```

## Model Information

- **File Name:** `cognitive_aim_model.pth`
- **File Size:** 568MB
- **Model Type:** Cognitive-Aim Depth Estimation
- **Framework:** PyTorch
- **Input Resolution:** Variable (supports multiple resolutions)

## Usage

Once the model is downloaded and placed in this directory, you can use it for inference:

```bash
# Single image inference
python demo.py --image your_image.jpg --instruction center

# Guided inference with spatial attention
python demo.py --image your_image.jpg --instruction top-left
```

## Note for International Users

If you have difficulty accessing Baidu Netdisk from outside China, please:
1. Try using a VPN service
2. Contact the repository maintainer for alternative download links
3. Check back later for Google Drive or Hugging Face links