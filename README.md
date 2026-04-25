# EDFusion
This is the official PyTorch implementation of: EDFusion: Uncertainty-Aware Infrared and Visible Image Fusion via Evidential Deep Learning

📌 Framework

EDFusion formulates infrared and visible image fusion as a pixel-wise uncertainty-aware inference problem.
Each modality is modeled with a Normal–Inverse-Gamma (NIG) distribution, enabling joint estimation of intensity and uncertainty.
Unlike conventional deterministic fusion methods, EDFusion explicitly models reliability and uncertainty, allowing adaptive fusion under noise, illumination variation, and cross-modal inconsistency.

🧠 Method

EDFusion consists of three key components:

1. Evidential Regression
Each modality (IR / VIS) is independently mapped to NIG parameters:
Mean: μ
Evidence: ν, α, β
This enables pixel-wise uncertainty estimation.

2. NIG-based Uncertainty-Aware Fusion
A probabilistic fusion module is designed to:
Aggregate cross-modal evidence
Model modality conflicts
Propagate uncertainty
The final fused image is generated in an uncertainty-aware manner, rather than simple averaging.

3. Uncertainty-Guided Learning
Uncertainty is further used to guide optimization via:
Evidence-weighted reconstruction
Evidence regularization
Structure-preserving constraints (SSIM, TV)

📂 Code Structure
EDFusion/
├── train.py        # training script
├── model.py        # evidential fusion network
├── utils.py        # loss functions and utilities
├── test.py         # testing / inference
├── data/
│   ├── train/
│   │   ├── ir/
│   │   └── vi/
│   └── test/
│       ├── ir/
│       └── vi/
└── output/

⚙️ Environment
Recommended environment:
python >= 3.8
torch >= 1.10
numpy
opencv-python
matplotlib
h5py
Pillow
tensorboard

Install dependencies:
pip install torch torchvision numpy pillow matplotlib h5py opencv-python tensorboard

📊 Dataset
Organize dataset as:
data/
├── train/
│   ├── ir/
│   └── vi/
└── test/
    ├── ir/
    └── vi/
Notes:
IR images are converted to grayscale
VIS images can be RGB
Image pairs are matched by filename order

🚀 Training
Run training:
python train.py \
  --ir-dir ./data/train/ir \
  --vi-dir ./data/train/vi \
  --output-dir ./output \
  --batch-size 8 \
  --epochs 100
  
⚡ Faster Training (optional)
--use-h5 True
This enables HDF5 patch caching for faster training.

The framework automatically saves:
best_model.pth
checkpoint_epoch_x.pth
final_model.pth
🧪 Testing
python test.py \
  --ir-dir ./data/test/ir \
  --vis-dir ./data/test/vi \
  --model-path ./output/models/final_model.pth \
  --output-dir ./test_output
🎨 Color Fusion Strategy
Convert VIS images to YCbCr space
Perform fusion on Y channel
Preserve CbCr channels
Reconstruct final RGB image

📤 Outputs
test_output/
├── fused_rgb/
├── f_uncertainty_maps/
└── fused_with_uncertainty/
🔑 Key Features
Pixel-wise evidential learning (NIG modeling)
Uncertainty-aware fusion mechanism
Conflict-aware evidence aggregation
Uncertainty-guided reconstruction
End-to-end trainable framework

📈 Loss Function
The total loss includes:
NLL loss (evidential regression)
Evidence regularization
Uncertainty-aware reconstruction loss
SSIM loss
Total variation (TV) loss

📌 Citation
@article{edfusion2026,
  title={EDFusion: Uncertainty-Aware Infrared and Visible Image Fusion via Evidential Deep Learning},
  author={Your Name},
  year={2026}
}

✨ Acknowledgement
This work is inspired by evidential deep learning and multimodal image fusion research.
