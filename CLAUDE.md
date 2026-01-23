# PIV-LiteFlowNet (Simplified)

A streamlined implementation of LiteFlowNet for Particle Image Velocimetry (PIV) velocity field extraction.

## Project Structure

```
piv-simple/
├── train.py                    # Training & inference CLI (main entry point)
├── requirements.txt            # Minimal dependencies
└── src/
    ├── __init__.py
    ├── piv_dataset_simple.py   # Dataset: PIVDataset, PIVInference, make_splits
    ├── models.py               # piv_liteflownet, hui_liteflownet
    ├── loss.py                 # piv_loss, hui_loss
    ├── correlation.py          # CUDA cost volume correlation layer
    ├── flow_transforms.py      # Data augmentation transforms
    ├── utils_plot.py           # Flow I/O (read_flow, write_flow, motion_to_color)
    └── utils_color.py          # Color wheel for flow visualization
```

## Usage

### Training
```bash
python train.py --mode train --data /path/to/piv_data --epochs 100
python train.py --mode train --data /path/to/piv_data --resume checkpoint.pth
```

### Inference
```bash
# Directory of image pairs
python train.py --mode infer --data /path/to/images --weights model_best.pth --output ./results

# Single image pair
python train.py --mode infer --img1 a.tif --img2 b.tif --weights model_best.pth
```

## Data Format

Expected directory structure:
```
data_root/
├── DNS_turbulence/
│   ├── DNS_turbulence_00500_img1.tif
│   ├── DNS_turbulence_00500_img2.tif
│   └── DNS_turbulence_00500_flow.flo
├── JHTDB_channel/
│   └── ...
```

Naming convention: `{name}_img1.{ext}`, `{name}_img2.{ext}`, `{name}_flow.flo`

## Key CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | required | `train` or `infer` |
| `--data` | "" | Data root directory |
| `--model` | "piv" | Model type: `piv` or `hui` |
| `--version` | 1 | LiteFlowNet version: 1 or 2 |
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 8 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--crop-size` | [256, 256] | Training crop size |
| `--weights` | "" | Pretrained weights for inference |
| `--resume` | "" | Resume training from checkpoint |
| `--save-dir` | "./checkpoints" | Checkpoint save directory |
| `--output` | "./output" | Inference output directory |

## Architecture

- **Models**: `piv_liteflownet` (optimized for PIV) and `hui_liteflownet` (original)
- **Loss**: Multi-scale EPE loss with configurable scaling
- **Correlation**: Custom CUDA kernel for cost volume computation (requires cupy)

## Notes for Development

- The `correlation.py` module uses cupy for GPU acceleration - ensure cupy version matches CUDA
- Training outputs checkpoints to `./checkpoints/` including `model_best.pth`
- Inference uses adaptive sizing (multiples of 32) to handle arbitrary image sizes
- Flow files use `.flo` format (standard optical flow format)
