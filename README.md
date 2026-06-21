# MS-PIGNN

This repository contains the implementation of the MS-PIGNN framework.

## Environment

Recommended environment:

- Python >= 3.9
- PyTorch
- NumPy
- Pandas
- SciPy
- scikit-learn

Install the required packages according to your local environment.

---

## Project Structure

```text
MS-PIGNN/
├── data/               # Dataset
├── ieee118new/         # IEEE 118-bus system related files
├── layers/             # Network layers
├── test_model/         # Saved models and testing utilities
├── utils/              # Utility functions
├── train.py            # Stage 1: Data-driven model training
├── train_pinn.py       # Stage 2: Physics-informed fine-tuning
└── README.md
```

---

## Usage

The training procedure consists of two stages.

### Stage 1: Train the baseline model

Run:

```bash
python train.py
```

This step performs the initial data-driven training and generates the pretrained model required for the next stage.

---

### Stage 2: Physics-informed training

After completing Stage 1, run:

```bash
python train_pinn.py
```

This step incorporates physics-informed constraints to further optimize the pretrained model.

---

## Training Workflow

```text
train.py
    ↓
Generate pretrained model
    ↓
train_pinn.py
    ↓
Obtain the final MS-PIGNN model
```

---

## Notes

- `train.py` must be executed before `train_pinn.py`.
- Ensure that the datasets are correctly placed under the `data/` directory before training.
- Check the corresponding scripts for configurable hyperparameters and paths.

---

## Citation

If you find this repository useful in your research, please cite the corresponding paper.
