# PIV-LiteFlowNet

A streamlined PyTorch implementation of LiteFlowNet for Particle Image Velocimetry (PIV) velocity field extraction.

Based on the PIV-LiteFlowNet architecture from [Cai et al., 2019](https://ieeexplore.ieee.org/document/8793167) and the original LiteFlowNet from [Hui et al., 2018](https://arxiv.org/abs/1805.07036).

## Features

- Train on synthetic PIV data with ground truth flow fields
- Run inference on experimental PIV image pairs
- Supports both LiteFlowNet and LiteFlowNet2 architectures
- Handles arbitrary image sizes via adaptive resizing
- Outputs standard `.flo` flow files and color visualizations

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/piv-liteflownet.git
cd piv-liteflownet

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.7+
- PyTorch >= 1.4.0
- CUDA-compatible GPU (required for correlation layer)
- cupy (match your CUDA version, e.g., `cupy-cuda11x`)

## Usage

### Training

```bash
# Basic training
python train.py --mode train --data /path/to/piv_data --epochs 100

# With custom parameters
python train.py --mode train \
    --data /path/to/piv_data \
    --epochs 200 \
    --batch-size 16 \
    --lr 1e-4 \
    --crop-size 256 256 \
    --save-dir ./checkpoints

# Resume from checkpoint
python train.py --mode train --data /path/to/piv_data --resume checkpoint.pth

# Train on specific subsets
python train.py --mode train --data /path/to/piv_data --subsets DNS_turbulence JHTDB_channel
```

### Inference

```bash
# Single image pair
python train.py --mode infer \
    --img1 image_a.tif \
    --img2 image_b.tif \
    --weights model_best.pth \
    --output ./results

# Directory of image pairs
python train.py --mode infer \
    --data /path/to/images \
    --weights model_best.pth \
    --output ./results
```

## Data Format

### Training Data Structure

```
data_root/
├── DNS_turbulence/
│   ├── DNS_turbulence_00001_img1.tif
│   ├── DNS_turbulence_00001_img2.tif
│   ├── DNS_turbulence_00001_flow.flo
│   ├── DNS_turbulence_00002_img1.tif
│   ├── DNS_turbulence_00002_img2.tif
│   ├── DNS_turbulence_00002_flow.flo
│   └── ...
├── JHTDB_channel/
│   └── ...
└── SQG/
    └── ...
```

**Naming convention:** `{name}_img1.{ext}`, `{name}_img2.{ext}`, `{name}_flow.flo`

### Inference Data

For inference, only image pairs are needed (no ground truth flow):
```
images/
├── sample_001_img1.tif
├── sample_001_img2.tif
├── sample_002_img1.tif
├── sample_002_img2.tif
└── ...
```

### Flow File Format

Flow files use the standard Middlebury `.flo` format:
- 4-byte magic number (202021.25 as float)
- 4-byte width (int)
- 4-byte height (int)
- width × height × 2 float32 values (u, v components)

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | *required* | `train` or `infer` |
| `--data` | `""` | Data root directory |
| `--subsets` | all | Specific subdirectories to use |
| `--ext` | `tif` | Image file extension |
| `--model` | `piv` | Model type: `piv` or `hui` |
| `--version` | `1` | LiteFlowNet version: `1` or `2` |
| `--epochs` | `100` | Number of training epochs |
| `--batch-size` | `8` | Training batch size |
| `--lr` | `1e-4` | Learning rate |
| `--crop-size` | `256 256` | Training crop size (H W) |
| `--save-dir` | `./checkpoints` | Checkpoint save directory |
| `--weights` | `""` | Pretrained weights for inference |
| `--resume` | `""` | Resume training from checkpoint |
| `--img1` | `""` | First image (single pair inference) |
| `--img2` | `""` | Second image (single pair inference) |
| `--output` | `./output` | Inference output directory |
| `--device` | `cuda` | Device: `cuda` or `cpu` |
| `--workers` | `4` | DataLoader workers |

## Project Structure

```
piv-simple/
├── train.py                    # Training & inference CLI
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── src/
    ├── __init__.py
    ├── piv_dataset_simple.py   # Dataset classes
    ├── models.py               # LiteFlowNet architectures
    ├── loss.py                 # Multi-scale loss functions
    ├── correlation.py          # CUDA cost volume layer
    ├── flow_transforms.py      # Data augmentation
    ├── utils_plot.py           # Flow I/O and visualization
    └── utils_color.py          # Flow color wheel
```

## Output

### Training
- `checkpoints/config.json` - Training configuration
- `checkpoints/checkpoint_last.pth` - Latest checkpoint
- `checkpoints/model_best.pth` - Best validation EPE
- `checkpoints/checkpoint_XXX.pth` - Periodic checkpoints

### Inference
- `{name}_flow.flo` - Flow field in Middlebury format
- `{name}_vis.png` - Color visualization of flow

## Citation

If you use this code, please cite:

```bibtex
@article{cai2019particle,
  title={Particle image velocimetry based on a deep learning motion estimator},
  author={Cai, Shengze and Zhou, Shichao and Xu, Chao and Gao, Qi},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={69},
  number={6},
  pages={3538--3554},
  year={2019},
  publisher={IEEE}
}

@inproceedings{hui2018liteflownet,
  title={LiteFlowNet: A lightweight convolutional neural network for optical flow estimation},
  author={Hui, Tak-Wai and Tang, Xiaoou and Loy, Chen Change},
  booktitle={CVPR},
  pages={8981--8989},
  year={2018}
}
```

## License

This project is for research purposes. Please refer to the original repositories for licensing information.
