# QTB Scan: Hybrid Quantum-Classical CNN for Tuberculosis Detection on Chest X-rays

A hybrid model that pairs a frozen ImageNet ResNet18 with a 4-qubit quantum convolutional neural network (QCNN) built in [PennyLane](https://pennylane.ai/). It classifies chest radiographs as **Normal** or **Tuberculosis (TB)**, and includes a Streamlit app that shows the prediction with a lung-masked activation map.

> **Research prototype.** This code is for research only. It is not a medical device and must not be used for clinical diagnosis, treatment, or coding decisions.

---

## Results

Test set: DB2 test split, 1,936 images. Single run, single seed (42).

| | Predicted Normal | Predicted TB |
|---|---:|---:|
| **True Normal** | 1,122 (TN) | 154 (FP) |
| **True TB** | 48 (FN) | 612 (TP) |

| Metric | Value |
|---|---:|
| Accuracy | 89.6% (1,734 / 1,936) |
| Sensitivity (TB recall) | 92.7% (612 / 660) |
| Specificity (Normal recall) | 87.9% (1,122 / 1,276) |
| Precision (TB) | 79.9% (612 / 766) |

Validation accuracy peaked at 90.7% during the final phase of training.

---

## Model

```
Chest X-ray (224×224 RGB)
  │
  ├─ ResNet18 backbone (ImageNet weights, frozen)        → 512-d features
  ├─ Projector: Linear(512 → 4) + Tanh, scaled by π       → 4 rotation angles
  ├─ QCNN (PennyLane, 4 qubits, 16 trainable parameters)  → ⟨Z⟩ on qubit 3
  └─ Linear(1 → 2) head                                   → Normal / TB
```

The 16 quantum parameters are split into three layers:

| Layer | Gates | Parameters |
|---|---|---:|
| HEAD | Two-qubit unitaries on (q0, q1) and (q2, q3) | 8 |
| MIDDLE | Pooling (CRZ) and second convolution on q1, q3 | 6 |
| BODY | Final pooling (CRZ) and rotation on q3 | 2 |

**Training.** The backbone is frozen, so its features are extracted once and cached. The quantum and head layers are then trained in three progressive phases (HEAD → HEAD + MIDDLE → full QCNN), 5 epochs each, with Adam, cosine learning-rate decay and a weighted sampler for class balance.

**Explainability.** The app uses a lung mask and a 3-step CAM pipeline. The notebook produces true Grad-CAM on DB2 test images (`pytorch-grad-cam`, ResNet18 layer 4). The app uses a linear approximation of the quantum layer, so its maps are heuristic.

---

## Repository layout

```
.
├── app/
│   └── app.py                          Streamlit inference app
├── notebooks/
│   ├── 01_qiskit_baseline.ipynb        Original Qiskit QCNN, progressive 3-phase training
│   ├── 02_pennylane_optimised.ipynb    PennyLane version: AdamW, OneCycle LR, early stopping
│   ├── 03_final_model.ipynb            Final model, produces qtb_model_final.pth
│   ├── 04_icmr_finetune.ipynb          Experiment: fine-tuning on ICMR DICOMs only
│   └── validation_notebook.ipynb       TODO: describe
├── requirements.txt
└── README.md
```

---

## Data

| Dataset | Source | Images | Use |
|---|---|---|---|
| DB1: TB Chest Radiography Database | [kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) | 4,200 (3,500 Normal, 700 TB) | Training |
| DB2: TB dataset with train / valid / test splits | [kaggle.com/datasets/tirthachhetri/tuberclosis-dataset](https://www.kaggle.com/datasets/tirthachhetri/tuberclosis-dataset) | 9,537 train, 1,288 val, 1,936 test | Training, validation, test |
| ICMR DICOM collection | **Not included** (data-use agreement) | 162 (all TB) | Training only |

The ICMR DICOM images are subject to a data-use agreement and are **not redistributed** in this repository. The notebooks expect them in a folder named `intbtr254` (or whatever you set `ICMR_ROOT` to). The DB1 and DB2 notebooks download the public datasets through `kagglehub` automatically.

Training pool after merging: 13,899 images (9,728 Normal, 4,171 TB).

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` should include: `torch`, `torchvision`, `pennylane`, `pennylane-lightning`, `kagglehub`, `scikit-learn`, `seaborn`, `matplotlib`, `grad-cam`, `pydicom`, `pillow`, `numpy`, `scipy`, `streamlit`. The app uses `st.html`, so use a recent Streamlit release.

---

## Reproducing the final model

1. Open `notebooks/03_final_model.ipynb`, locally or on Kaggle/Colab. The notebook detects the environment and selects the save directory.
2. Set `ICMR_ROOT` in the path cell to your local copy of the ICMR DICOM folder.
3. Run all cells. The DB1 and DB2 datasets download automatically. The notebook saves `qtb_model_final.pth` to the save directory.

Training uses a fixed seed (42) but has not been run across multiple seeds, so expect small run-to-run variation.

---

## Running the app

1. Put `qtb_model_final.pth` somewhere on disk.
2. In `app/app.py`, set `PTH_PATH` to that path. It is currently a hard-coded Windows path.
3. Start the app:

   ```bash
   streamlit run app/app.py
   ```

4. Upload a chest X-ray (PNG, JPG, BMP, TIFF, or DICOM).

If the filename contains a label keyword (for example `_Abnormal`, `_Normal`, `tb`, `healthy`), the app compares the prediction against it. This is a convenience for evaluation only and is not a source of ground truth.

---

## Model weights

| File | Description | Status |
|---|---|---|
| `qtb_model_final.pth` | Final model used by the app (`model_state_dict`, `classes`, `history`) | Link pending: `TODO: Kaggle model URL or GitHub Release URL` |
| `qcnn_phase3_best.pth` | Qiskit-era QCNN checkpoint from `01_qiskit_baseline` | Starting point for the fine-tuning experiment |
| `icmr_finetuned_best.pth` | Fine-tuned on ICMR DICOMs only | Experimental, not validated (see Limitations) |

Load the final model like this:

```python
ckpt = torch.load("qtb_model_final.pth", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"], strict=True)
```

---

## Limitations

- **Single split, single seed.** The results come from one run on one test split. There are no confidence intervals and no repeated runs.
- **No external validation.** DB1 and DB2 are both public Kaggle collections, and the ICMR data is used only for training. The model has not been tested on data from another hospital or scanner. Overlap between DB1 and DB2 has not been checked.
- **Input resolution mismatch.** The final training notebook extracts features at 160×160. `app.py` preprocesses at 224×224. Confirm which size the final checkpoint was trained with before relying on app predictions.
- **ICMR fine-tuning is not validated.** The fine-tuning notebook trains on ICMR DICOMs only, which are all TB-positive. Its "validation accuracy" is therefore always 100% and says nothing about performance. It also initialises from a Qiskit checkpoint whose quantum weights were remapped to the PennyLane layout, and some layers were re-initialised randomly.
- **Heuristic explanations in the app.** The app's activation maps use a linear approximation, not true Grad-CAM.
- **Filename labels.** The app's ground-truth comparison relies on filenames and is for evaluation only.

---

## Citation

```bibtex
% TODO: add once the paper is accepted or preprinted
```

## Acknowledgements

- Public datasets: DB1 and DB2, via Kaggle. See the Data section for links. `TODO: add the dataset citations used in the paper`
- ICMR DICOM collection: used under a data-use agreement; not redistributed.
- Built with PennyLane, PyTorch, torchvision, and pytorch-grad-cam.

## License

`TODO: choose a license for code (e.g. MIT). Note that the datasets and model weights carry their own terms.`
