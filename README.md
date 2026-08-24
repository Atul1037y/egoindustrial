# EgoIndustrial

Scalable Egocentric Action Recognition Pipeline for multi-domain video understanding.

## Target Metrics
- **78.3% top-1** on EPIC-KITCHENS-100 verb+noun
- **120 FPS** TensorRT INT8 inference on T4
- **500+ hours** multi-domain video processing

## Stack
- **Training**: PyTorch Lightning, Hydra, Weights & Biases
- **Models**: VideoMAEv2, MViTv2, SlowFast, InternVideo2
- **Data**: EPIC-KITCHENS-100, Assembly101, HoloAssist
- **Inference**: ONNX → TensorRT INT8 → FastAPI + Prometheus
- **Weak Supervision**: Pseudo-labeling + Streamlit Human-in-loop UI
- **CI/CD**: GitHub Actions → Docker → Modal/GCP Cloud Run

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Train
python -m egoindustrial.training.train dataset=epic_kitchens model=videomaev2

# Export to ONNX
python -m egoindustrial.inference.export_onnx checkpoint=path/to/ckpt.ckpt

# Build TensorRT engine
python -m egoindustrial.inference.tensorrt_engine onnx=model.onnx

# Run inference server
python -m egoindustrial.inference.server
Project Structure
egoindustrial/
├── configs/           # Hydra configs (dataset, model, train, inference)
├── src/egoindustrial/ # Main package
│   ├── data/          # Datasets & dataloaders
│   ├── models/        # Model zoo
│   ├── training/      # Lightning modules
│   ├── weak_supervision/  # Pseudo-labeling + UI
│   └── inference/     # ONNX, TensorRT, FastAPI
├── scripts/           # Utility scripts
├── tests/             # Unit tests
├── docker/            # Dockerfiles
└── notebooks/         # Exploration notebooks
License
MIT
