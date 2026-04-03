# MS-Road Examples

This directory contains example scripts for working with the MS-Road dataset.

## Files

- `load_data.py` - Basic data loading and visualization
- `load_annotations.py` - COCO format annotation loading and processing
- `train_example.py` - Training examples
- `requirements.txt` - Python dependencies

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run examples:
```bash
# Load and visualize multispectral data
python load_data.py

# Work with annotations
python load_annotations.py
```

## Important Notes

### Data Shape
MS-Road images have shape `(8, 1200, 900)` which is **(Channels, Width, Height)**, NOT (C, H, W).

```python
import numpy as np

data = np.load('image.npy')
print(data.shape)  # (8, 1200, 900)
# Channel 0-7: 395, 450, 525, 590, 650, 730, 850, 950 nm
```

### Loading NPY Images

```python
import numpy as np
import matplotlib.pyplot as plt

# Load multispectral image
data = np.load('image.npy')  # Shape: (8, 1200, 900)
print(f"Data shape: {data.shape}")

# Access individual bands
band_395nm = data[0]  # Near-UV
band_450nm = data[1]  # Blue
band_525nm = data[2]  # Green
band_590nm = data[3]  # Yellow
band_650nm = data[4]  # Red
band_730nm = data[5]  # Red Edge
band_850nm = data[6]  # NIR-1
band_950nm = data[7]  # NIR-2
```

### Pseudo-RGB Creation

```python
import numpy as np

# Extract bands for pseudo-RGB
r = data[4]  # 650 nm (Red)
g = data[2]  # 525 nm (Green)
b = data[1]  # 450 nm (Blue)

# Create RGB image
rgb = np.stack([r, g, b], axis=-1)
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
```

## Citation

If you use these examples in your research, please cite:

```bibtex
@inproceedings{gao2026msroad,
  title={MS-Road: A Large-Scale Challenging Benchmark for Multispectral Road Object Detection},
  author={Gao, Huawei and Xu, Tingfa and Liu, Peifu and Li, Tianhao and Chen, Huan and Li, Jianan},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
  year={2026}
}
```
