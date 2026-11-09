import os, io, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter
from torchvision import models, transforms
import pennylane as qml
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pydicom

# ─────────────────────────────────────────────────────────────
#  CONFIG  — updated to qtb_model_final.pth (PennyLane trained)
# ─────────────────────────────────────────────────────────────
PTH_PATH  = r"C:\Users\rames\Downloads\qtb_model_final.pth"
CLASSES   = ["Normal", "Tuberculosis"]
N_QUBITS  = 4
N_WEIGHTS = 16          # 8 (HEAD) + 6 (MIDDLE) + 2 (BODY)
MEAN      = np.array([0.485, 0.456, 0.406])
STD       = np.array([0.229, 0.224, 0.225])

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG  — must be first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QTB Scan · TB Diagnostic AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────
st.html("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg:       #050c14;
  --bg2:      #0a1622;
  --bg3:      #0f1e2e;
  --border:   #1a3048;
  --cyan:     #00d4ff;
  --green:    #00ff9d;
  --red:      #ff4560;
  --amber:    #ffa040;
  --text:     #c8dce8;
  --text-dim: #5a7a8a;
  --mono:     'DM Mono', monospace;
  --display:  'Syne', sans-serif;
}

html, body,
[data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text);
  font-family: var(--display);
}
[data-testid="stAppViewContainer"] { background: var(--bg) !important; }
[data-testid="stHeader"]           { background: transparent !important; }
section[data-testid="stSidebar"]   { display: none; }
[data-testid="stMainBlockContainer"] { padding: 0 2rem; max-width: 1400px; }

::-webkit-scrollbar       { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.hero {
  position: relative; padding: 3.5rem 3rem 2.5rem;
  margin-bottom: 2rem; border-bottom: 1px solid var(--border); overflow: hidden;
}
.hero::before {
  content: ''; position: absolute; inset: 0;
  background: radial-gradient(ellipse 80% 60% at 60% 0%,
    rgba(0,212,255,0.07) 0%, transparent 70%);
  pointer-events: none;
}
.hero-tag {
  font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.25em;
  color: var(--cyan); text-transform: uppercase; border: 1px solid var(--cyan);
  display: inline-block; padding: 0.25rem 0.75rem; border-radius: 2px; margin-bottom: 1rem;
}
.hero h1 {
  font-size: 3.2rem; font-weight: 800; line-height: 1.05;
  letter-spacing: -0.03em; margin: 0 0 0.6rem;
  background: linear-gradient(120deg, #ffffff 30%, var(--cyan));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p {
  color: var(--text-dim); font-family: var(--mono);
  font-size: 0.82rem; letter-spacing: 0.05em; margin: 0;
}
.hero-badge {
  position: absolute; top: 3rem; right: 3rem;
  font-family: var(--mono); font-size: 0.62rem;
  color: var(--text-dim); text-align: right; line-height: 1.8; letter-spacing: 0.08em;
}

[data-testid="stFileUploader"] {
  background: var(--bg2) !important; border: 1px dashed var(--border) !important;
  border-radius: 8px !important; padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--cyan) !important; }
[data-testid="stFileUploader"] label,
[data-testid="stFileDropzoneInstructions"] { color: var(--text-dim) !important; }

.panel {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 10px; padding: 1.6rem 1.8rem;
  margin-bottom: 1.2rem; position: relative; overflow: hidden;
}
.panel::after {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--cyan), transparent);
}
.panel-title {
  font-family: var(--mono); font-size: 0.65rem; letter-spacing: 0.22em;
  text-transform: uppercase; color: var(--text-dim); margin-bottom: 1.2rem;
}

.diag-normal {
  background: linear-gradient(135deg, #001a0d, #002a14);
  border: 1px solid #00ff9d44; border-radius: 10px; padding: 2rem;
  position: relative; overflow: hidden;
}
.diag-normal::after {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--green), transparent);
}
.diag-tb {
  background: linear-gradient(135deg, #1a0005, #2a000a);
  border: 1px solid #ff456044; border-radius: 10px; padding: 2rem;
  position: relative; overflow: hidden;
}
.diag-tb::after {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--red), transparent);
}
.diag-label {
  font-size: 0.65rem; font-family: var(--mono); letter-spacing: 0.2em;
  text-transform: uppercase; color: var(--text-dim); margin-bottom: 0.5rem;
}
.diag-result-normal { font-size: 2.6rem; font-weight: 800; color: var(--green); letter-spacing: -0.02em; }
.diag-result-tb     { font-size: 2.6rem; font-weight: 800; color: var(--red);   letter-spacing: -0.02em; }
.conf-value { font-family: var(--mono); font-size: 1.8rem; font-weight: 500; margin-top: 0.3rem; }

.prob-row   { margin-bottom: 1rem; }
.prob-label {
  display: flex; justify-content: space-between;
  font-family: var(--mono); font-size: 0.75rem;
  color: var(--text-dim); margin-bottom: 0.4rem;
}
.prob-track       { height: 6px; background: var(--bg3); border-radius: 3px; overflow: hidden; }
.prob-fill-normal { height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--green), #00ffaa); }
.prob-fill-tb     { height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--red), #ff8080); }

.chip-row { display: flex; gap: 0.7rem; flex-wrap: wrap; margin-top: 1.2rem; }
.chip {
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 4px; padding: 0.35rem 0.8rem;
  font-family: var(--mono); font-size: 0.68rem;
  color: var(--text-dim); letter-spacing: 0.08em;
}
.chip span { color: var(--cyan); }

.step-badge {
  display: inline-block; font-family: var(--mono); font-size: 0.58rem;
  letter-spacing: 0.15em; padding: 0.15rem 0.6rem; border-radius: 2px;
  margin-bottom: 0.4rem; text-transform: uppercase;
}
.step-1 { background: #00d4ff18; border: 1px solid #00d4ff44; color: var(--cyan); }
.step-2 { background: #ffa04018; border: 1px solid #ffa04044; color: var(--amber); }
.step-3 { background: #ff456018; border: 1px solid #ff456044; color: var(--red); }

.feat-bar-wrap  { margin: 0.35rem 0; }
.feat-bar-label {
  font-family: var(--mono); font-size: 0.65rem; color: var(--text-dim);
  display: flex; justify-content: space-between; margin-bottom: 0.2rem;
}
.feat-bar-track { height: 4px; background: var(--bg3); border-radius: 2px; overflow: hidden; }

.gcam-caption {
  font-family: var(--mono); font-size: 0.68rem; color: var(--text-dim);
  text-align: center; letter-spacing: 0.05em; margin-top: 0.5rem;
}

[data-testid="stImage"] > img { border-radius: 8px; border: 1px solid var(--border); }

.footer {
  text-align: center; padding: 2.5rem 0 1.5rem;
  font-family: var(--mono); font-size: 0.62rem;
  color: var(--text-dim); letter-spacing: 0.12em;
  border-top: 1px solid var(--border); margin-top: 3rem;
}

/* ── ICD-10-CM Panel ── */
.icd-panel {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 10px; padding: 1.4rem 1.6rem;
  margin-bottom: 1.2rem; position: relative; overflow: hidden;
}
.icd-panel::after {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, #e53935, #ff7043, transparent);
}
.icd-panel-normal::after {
  background: linear-gradient(90deg, var(--green), #69f0ae, transparent);
}
.icd-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 0.7rem; margin-top: 0.9rem;
}
.icd-card {
  background: var(--bg3); border-radius: 7px;
  padding: 0.75rem 1rem; border: 1px solid var(--border);
  transition: border-color .15s;
}
.icd-card:hover { border-color: #ff456066; }
.icd-card-normal:hover { border-color: #00ff9d66; }
.icd-code {
  font-family: var(--mono); font-size: 1rem; font-weight: 700;
  letter-spacing: 0.06em; margin-bottom: 0.3rem;
}
.icd-desc {
  font-family: var(--mono); font-size: 0.68rem;
  color: var(--text-dim); line-height: 1.6;
}
.icd-tier {
  display: inline-block; font-family: var(--mono); font-size: 0.55rem;
  letter-spacing: 0.15em; text-transform: uppercase;
  padding: 0.1rem 0.5rem; border-radius: 2px; margin-bottom: 0.4rem;
}
.icd-primary   { background: #ff456020; border: 1px solid #ff456066; color: #ff7043; }
.icd-range     { background: #ffa04020; border: 1px solid #ffa04066; color: var(--amber); }
.icd-screening { background: #00d4ff20; border: 1px solid #00d4ff66; color: var(--cyan); }
.icd-normal-p  { background: #00ff9d20; border: 1px solid #00ff9d66; color: var(--green); }
.icd-followup  { background: #a855f720; border: 1px solid #a855f766; color: #ce93d8; }
.icd-disclaimer {
  font-family: var(--mono); font-size: 0.62rem; color: var(--text-dim);
  margin-top: 0.9rem; padding-top: 0.7rem;
  border-top: 1px solid var(--border); line-height: 1.7;
}
.sens-spec-row {
  display: flex; gap: 1.2rem; flex-wrap: wrap; margin-top: 0.8rem;
}
.ss-card {
  background: var(--bg3); border-radius: 7px;
  padding: 0.7rem 1.1rem; border: 1px solid var(--border);
  text-align: center; flex: 1; min-width: 120px;
}
.ss-value { font-family: var(--mono); font-size: 1.5rem; font-weight: 700; }
.ss-label { font-family: var(--mono); font-size: 0.6rem; color: var(--text-dim);
            letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.2rem; }
.ss-bar-track { height: 5px; background: var(--bg2); border-radius: 3px;
                overflow: hidden; margin-top: 0.5rem; }

/* ── Ground-truth label banner ── */
.gt-match {
  background: #001a0d; border: 1px solid #00ff9d55;
  border-left: 4px solid var(--green);
  border-radius: 8px; padding: 1rem 1.4rem; margin-bottom: 1.2rem;
}
.gt-mismatch {
  background: #220500; border: 1px solid #ff456077;
  border-left: 4px solid var(--red);
  border-radius: 8px; padding: 1rem 1.4rem; margin-bottom: 1.2rem;
}
.gt-unknown {
  background: var(--bg2); border: 1px solid var(--border);
  border-left: 4px solid var(--text-dim);
  border-radius: 8px; padding: 1rem 1.4rem; margin-bottom: 1.2rem;
}
.gt-header { font-family: var(--mono); font-size: 0.6rem; letter-spacing: 0.22em;
             text-transform: uppercase; margin-bottom: 0.6rem; }
.gt-row    { display: flex; gap: 2.5rem; flex-wrap: wrap; margin-top: 0.4rem; }
.gt-cell   { font-family: var(--mono); font-size: 0.78rem; line-height: 1.9; }
.gt-cell .lbl { color: var(--text-dim); font-size: 0.6rem;
                letter-spacing: 0.15em; text-transform: uppercase; display: block; }
</style>
""")


# ─────────────────────────────────────────────────────────────
#  LUNG MASK
# ─────────────────────────────────────────────────────────────
def _make_lung_mask(h=224, w=224):
    y, x = np.ogrid[:h, :w]
    cy   = h * 0.46
    cx_r, ry_r, rx_r = w * 0.30, h * 0.295, w * 0.195
    cx_l, ry_l, rx_l = w * 0.70, h * 0.295, w * 0.195
    right = ((x - cx_r)/rx_r)**2 + ((y - cy)/ry_r)**2 <= 1.0
    left  = ((x - cx_l)/rx_l)**2 + ((y - cy)/ry_l)**2 <= 1.0
    mask  = (right | left).astype(np.float32)
    mask  = gaussian_filter(mask, sigma=13)
    return np.clip(mask / mask.max(), 0, 1)

LUNG_MASK = _make_lung_mask()


# ─────────────────────────────────────────────────────────────
#  MODEL  — PennyLane HybridQCNN matching training code
#
#  The checkpoint was trained with PennyLane lightning.qubit.
#  We rebuild the EXACT same architecture here so state_dict
#  keys match perfectly.
#
#  Key differences from the old Qiskit-based app:
#    • qml.device("default.qubit") for CPU inference (no lightning needed)
#    • qcnn_qnode mirrors the training qnode exactly
#    • TorchLayer weight_shapes = {"weights": 16}
#    • forward_cached() path used for inference (same as training)
# ─────────────────────────────────────────────────────────────

# Use default.qubit for local inference (no C++ install required).
# Switch to "lightning.qubit" if pennylane-lightning is installed locally.
_dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(_dev, interface="torch", diff_method="backprop")
def _qcnn_qnode(inputs, weights):
    h = weights[:8]
    m = weights[8:14]
    b = weights[14:]

    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')

    # HEAD: Conv Layer 1
    qml.RY(h[0], wires=0);  qml.RY(h[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(h[2], wires=0);  qml.RY(h[3], wires=1)
    qml.RY(h[4], wires=2);  qml.RY(h[5], wires=3)
    qml.CNOT(wires=[2, 3])
    qml.RY(h[6], wires=2);  qml.RY(h[7], wires=3)

    # MIDDLE: Pool1 + Conv Layer 2
    qml.CRZ(m[0], wires=[0, 1])
    qml.CRZ(m[1], wires=[2, 3])
    qml.RY(m[2], wires=1);  qml.RY(m[3], wires=3)
    qml.CNOT(wires=[1, 3])
    qml.RY(m[4], wires=1);  qml.RY(m[5], wires=3)

    # BODY: Pool2 + final rotation
    qml.CRZ(b[0], wires=[1, 3])
    qml.RY(b[1], wires=3)

    return qml.expval(qml.PauliZ(3))


class HybridQCNN(nn.Module):
    """
    Mirrors the training HybridQCNN exactly so state_dict loads cleanly.
    Keys expected in checkpoint:
        backbone.*  projector.*  qlayer.weights  out_head.*
    """
    def __init__(self):
        super().__init__()
        res = models.resnet18(weights=None)
        for p in res.parameters():
            p.requires_grad = False
        self.backbone  = nn.Sequential(*list(res.children())[:-1])
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, N_QUBITS),
            nn.Tanh(),
        )
        self.qlayer   = qml.qnn.TorchLayer(_qcnn_qnode, {"weights": N_WEIGHTS})
        self.out_head = nn.Linear(1, 2)

    def forward_cached(self, feats):
        """Inference path — feats are (B, 512) backbone outputs."""
        x = self.projector(feats) * np.pi     # (B, N_QUBITS)
        x = self.qlayer(x).unsqueeze(-1)      # (B, 1)
        return self.out_head(x)               # (B, 2)

    def forward(self, x):
        """Full path — raw image tensor (B, 3, H, W)."""
        x = self.backbone(x).squeeze(-1).squeeze(-1)   # (B, 512)
        return self.forward_cached(x)


# CAM wrapper — linear approximation of quantum layer for gradient maps
class CAMWrapper(nn.Module):
    def __init__(self, trained: HybridQCNN):
        super().__init__()
        self.backbone  = trained.backbone
        self.projector = trained.projector
        # Approximate quantum→output mapping with a direct linear layer
        self.head = nn.Linear(N_QUBITS, 2, bias=True)
        with torch.no_grad():
            self.head.weight.data = torch.tensor(
                [[-1., -1., -1., -1.],
                 [ 1.,  1.,  1.,  1.]])
            self.head.bias.data = trained.out_head.bias.data.clone()

    def forward(self, x):
        x = self.backbone(x)
        x = self.projector(x)
        return self.head(x)


@st.cache_resource(show_spinner=False)
def load_model():
    ckpt  = torch.load(PTH_PATH, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt
    model = HybridQCNN()
    model.load_state_dict(state, strict=True)
    model.eval()
    cam_model = CAMWrapper(model)
    cam_model.eval()
    return model, cam_model


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def predict(model: HybridQCNN, tensor: torch.Tensor):
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0))
        probs  = torch.softmax(logits, dim=1)[0].numpy()
    return int(np.argmax(probs)), probs


# ─────────────────────────────────────────────────────────────
#  FILENAME LABEL PARSER
#  Reads ground-truth label embedded in DICOM filenames.
#  e.g.  *_Abnormal.dicom  → 1 (TB)
#        *_Normal.dicom    → 0 (Normal)
#        anything else     → None (unknown)
# ─────────────────────────────────────────────────────────────
_FN_TB_KEYS     = {"abnormal", "tb", "tuberculosis", "positive", "infected"}
_FN_NORMAL_KEYS = {"normal", "healthy", "negative"}

def parse_filename_label(filename: str):
    """
    Returns (label_int, label_str) or (None, None) if not detectable.
    Checks each underscore/hyphen-separated token in the filename stem.
    """
    stem   = os.path.splitext(os.path.basename(filename))[0].lower()
    tokens = set(stem.replace("-", "_").split("_"))
    if tokens & _FN_TB_KEYS:
        return 1, "TB / Abnormal"
    if tokens & _FN_NORMAL_KEYS:
        return 0, "Normal / Healthy"
    # fallback: check full stem as substring
    if any(k in stem for k in _FN_TB_KEYS):
        return 1, "TB / Abnormal"
    if any(k in stem for k in _FN_NORMAL_KEYS):
        return 0, "Normal / Healthy"
    return None, None


# ─────────────────────────────────────────────────────────────
#  IMAGE LOADING  — supports PNG / JPG / DICOM
# ─────────────────────────────────────────────────────────────
def load_and_preprocess(uploaded_file):
    """
    Returns: (pil_img_for_display, preprocessed_tensor, info_string)
    DICOM files are converted to grayscale PIL before preprocessing.
    """
    ext = uploaded_file.name.lower().rsplit('.', 1)[-1]

    if ext in ('dcm', 'dicom'):
        dcm = pydicom.dcmread(io.BytesIO(uploaded_file.getvalue()), force=True)
        arr = dcm.pixel_array.astype(np.float32)

        photometric = getattr(dcm, 'PhotometricInterpretation', 'MONOCHROME2')
        if isinstance(photometric, str) and 'MONOCHROME1' in photometric.upper():
            arr = arr.max() - arr

        arr     = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        pil_img = Image.fromarray(arr.astype(np.uint8)).convert('L')
        tensor  = preprocess(pil_img.convert('RGB'))
        rows    = getattr(dcm, 'Rows', '?')
        cols    = getattr(dcm, 'Columns', '?')
        info    = f"DICOM · {photometric} · {rows}×{cols}"
        return pil_img, tensor, info

    pil_img = Image.open(uploaded_file).convert("RGB")
    tensor  = preprocess(pil_img)
    info    = f"{pil_img.mode} · {pil_img.size[0]}×{pil_img.size[1]}"
    return pil_img, tensor, info


# ─────────────────────────────────────────────────────────────
#  3-STEP CAM
# ─────────────────────────────────────────────────────────────
def compute_cam(cam_model: CAMWrapper, tensor: torch.Tensor, pred_class: int):
    spatial, proj_store = {}, {}

    h1 = cam_model.backbone[7].register_forward_hook(
        lambda m, i, o: spatial.update({"feat": o}))
    h2 = cam_model.projector.register_forward_hook(
        lambda m, i, o: proj_store.update({"out": o}))

    with torch.no_grad():
        _ = cam_model(tensor.unsqueeze(0))

    h1.remove(); h2.remove()

    feat             = spatial["feat"].squeeze(0)          # (512, 7, 7)
    proj_out         = proj_store["out"].squeeze(0)        # (N_QUBITS,)
    proj_w           = cam_model.projector[1].weight.data  # (N_QUBITS, 512)
    head_w           = cam_model.head.weight.data          # (2, N_QUBITS)
    class_w          = head_w[pred_class]                  # (N_QUBITS,)

    feature_scores   = (class_w * proj_out).numpy()
    channel_import   = ((class_w * proj_out).unsqueeze(1) * proj_w).sum(0)  # (512,)

    cam_7x7 = (channel_import.view(512, 1, 1) * feat).sum(0).numpy().astype(np.float32)
    cam_7x7 = np.maximum(cam_7x7, 0)
    if cam_7x7.max() > 1e-8:
        cam_7x7 /= cam_7x7.max()

    cam_224 = torch.nn.functional.interpolate(
        torch.tensor(cam_7x7).unsqueeze(0).unsqueeze(0),
        size=(224, 224), mode="bilinear", align_corners=False,
    ).squeeze().numpy().astype(np.float32)

    cam_224 = gaussian_filter(cam_224, sigma=5)
    cam_224 = cam_224 * LUNG_MASK
    if cam_224.max() > 1e-8:
        cam_224 /= cam_224.max()

    cam_224 = np.power(cam_224, 0.6)
    cam_224 = gaussian_filter(cam_224, sigma=4)
    if cam_224.max() > 1e-8:
        cam_224 /= cam_224.max()

    in_lung = cam_224[LUNG_MASK > 0.5]
    thresh  = np.percentile(in_lung, 65) if len(in_lung) > 0 else 0.5
    roi_bin  = ((cam_224 >= thresh) & (LUNG_MASK > 0.3)).astype(np.float32)
    roi_soft = gaussian_filter(roi_bin, sigma=8)
    if roi_soft.max() > 1e-8:
        roi_soft /= roi_soft.max()

    return cam_224, roi_soft, feature_scores


# ─────────────────────────────────────────────────────────────
#  RENDERING HELPERS
# ─────────────────────────────────────────────────────────────
def tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
    return np.clip(
        t.numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1
    ).astype(np.float32)


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    buf.seek(0); plt.close(fig)
    return Image.open(buf)


def render_plain(rgb):
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor("#0a1622"); ax.set_facecolor("#0a1622")
    ax.imshow(rgb, aspect="equal"); ax.axis("off")
    ax.set_title("PREPROCESSED INPUT", color="#5a7a8a", fontsize=7,
                 fontfamily="monospace", pad=8, loc="left")
    fig.text(0.5, 0.01, "224×224 · normalised · grayscale→RGB",
             ha="center", color="#5a7a8a", fontsize=6.5, fontfamily="monospace")
    plt.tight_layout(pad=0.4)
    return _fig_to_pil(fig)


def render_heatmap(cam_224):
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor("#050c14"); ax.set_facecolor("#050c14")
    ax.imshow(np.zeros((224, 224, 3)), aspect="equal")
    ax.imshow(cam_224, cmap="inferno", vmin=0, vmax=1,
              aspect="equal", alpha=np.clip(LUNG_MASK * 1.1, 0, 1))
    ax.axis("off")
    ax.set_title("ACTIVATION HEATMAP", color="#5a7a8a", fontsize=7,
                 fontfamily="monospace", pad=8, loc="left")
    fig.text(0.5, 0.01, "lung-masked · inferno · layer4",
             ha="center", color="#5a7a8a", fontsize=6.5, fontfamily="monospace")
    plt.tight_layout(pad=0.4)
    return _fig_to_pil(fig)


def render_roi_overlay(rgb, cam_224, roi_soft, pred_class):
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor("#0a1622"); ax.set_facecolor("#050c14")

    lung_3c = LUNG_MASK[:, :, np.newaxis]
    roi_3c  = roi_soft[:, :, np.newaxis]
    heat    = cm.inferno(cam_224)[:, :, :3].astype(np.float32)

    dimmed     = rgb * 0.30
    inside     = rgb * 0.55 + heat * 0.45
    inside_roi = rgb * 0.30 + heat * 0.70
    result     = (dimmed * (1 - lung_3c) +
                  (inside * (1 - roi_3c) + inside_roi * roi_3c) * lung_3c)
    result     = np.clip(result, 0, 1)

    ax.imshow(result, aspect="equal")
    border_color = "#ff4560" if pred_class == 1 else "#00d4ff"
    try:
        ax.contour(roi_soft, levels=[0.35],
                   colors=[border_color], linewidths=1.8, alpha=0.95)
    except Exception:
        pass

    ax.axis("off")
    ax.set_title("ROI OVERLAY", color="#5a7a8a", fontsize=7,
                 fontfamily="monospace", pad=8, loc="left")
    label = "TB region of interest" if pred_class == 1 else "Normal lung field"
    fig.text(0.5, 0.01, label, ha="center",
             color=border_color, fontsize=6.5, fontfamily="monospace")
    plt.tight_layout(pad=0.4)
    return _fig_to_pil(fig)


def feature_bar_html(feature_scores, pred_class):
    color = "#ff4560" if pred_class == 1 else "#00ff9d"
    norms = np.abs(feature_scores)
    if norms.max() > 0:
        norms = norms / norms.max()
    html = '<div style="margin-top:0.8rem">'
    for i, (raw, norm) in enumerate(zip(feature_scores, norms)):
        sign = "+" if raw >= 0 else "−"
        pct  = int(norm * 100)
        html += f"""
        <div class="feat-bar-wrap">
          <div class="feat-bar-label">
            <span>Quantum neuron Q{i+1}</span>
            <span style="color:{color}">{sign} {abs(raw):.3f}</span>
          </div>
          <div class="feat-bar-track">
            <div style="width:{pct}%;height:100%;border-radius:2px;
              background:linear-gradient(90deg,{color}88,{color});"></div>
          </div>
        </div>"""
    html += "</div>"
    return html


# ─────────────────────────────────────────────────────────────
#  ICD-10-CM PANEL
#
#  Confidence tiers (TB):
#    > 85%  → A15.0  primary confirmed code
#    60–85% → A15-A19 range + Z11.1 screening
#    < 60%  → R76.11 nonspecific / borderline
#
#  Normal prediction always shows Z11.1 (screened, no TB found)
#  plus Z86.15 (personal history flag if applicable).
#
#  Model performance stats (from test set confusion matrix):
#    TP=612  FP=154  FN=48  TN=1122
#    Sensitivity = 612/(612+48)  = 92.7%
#    Specificity = 1122/(1122+154) = 88.0%
# ─────────────────────────────────────────────────────────────
def icd_panel_html(pred_class: int, conf_pct: float) -> str:
    is_tb = pred_class == 1

    # ── Sensitivity / Specificity stats (fixed from test results) ──
    SENSITIVITY = 92.7
    SPECIFICITY = 88.0

    ss_html = f"""
    <div class="sens-spec-row">
      <div class="ss-card">
        <div class="ss-value" style="color:#00ff9d">{SENSITIVITY}%</div>
        <div class="ss-label">Sensitivity<br>(TB Recall)</div>
        <div class="ss-bar-track">
          <div style="width:{SENSITIVITY}%;height:100%;background:linear-gradient(90deg,#00ff9d88,#00ff9d);border-radius:3px"></div>
        </div>
      </div>
      <div class="ss-card">
        <div class="ss-value" style="color:#00d4ff">{SPECIFICITY}%</div>
        <div class="ss-label">Specificity<br>(Normal Recall)</div>
        <div class="ss-bar-track">
          <div style="width:{SPECIFICITY}%;height:100%;background:linear-gradient(90deg,#00d4ff88,#00d4ff);border-radius:3px"></div>
        </div>
      </div>
      <div class="ss-card">
        <div class="ss-value" style="color:#ffa040">90.0%</div>
        <div class="ss-label">Overall<br>Accuracy</div>
        <div class="ss-bar-track">
          <div style="width:90%;height:100%;background:linear-gradient(90deg,#ffa04088,#ffa040);border-radius:3px"></div>
        </div>
      </div>
      <div class="ss-card">
        <div class="ss-value" style="color:#a855f7">1,936</div>
        <div class="ss-label">Test Set<br>Samples</div>
        <div class="ss-bar-track">
          <div style="width:100%;height:100%;background:linear-gradient(90deg,#a855f788,#a855f7);border-radius:3px"></div>
        </div>
      </div>
    </div>"""

    # ── ICD cards for TB prediction ─────────────────────────────────
    if is_tb:
        if conf_pct > 85:
            cards = [
                ("primary",   "A15.0",  "Tuberculosis of lung, confirmed",
                 "PRIMARY · HIGH CONFIDENCE",
                 "Confirmed pulmonary TB. Active infection of the lung parenchyma by Mycobacterium tuberculosis. Requires immediate initiation of standard DOTS therapy."),
                ("range",     "A15–A19","Tuberculosis — Full ICD-10 Range",
                 "RANGE · RESPIRATORY TO MILIARY",
                 "A15: Respiratory TB · A16: TB of lung without confirmation · A17: TB of nervous system · A18: TB of other organs · A19: Miliary TB."),
                ("followup",  "Z11.1",  "Encounter for screening for respiratory TB",
                 "FOLLOW-UP · CONFIRM WITH CULTURE",
                 "Recommended next step: sputum smear microscopy + culture (Löwenstein–Jensen or BACTEC MGIT). Await bacteriological confirmation before treatment."),
                ("screening", "R76.11", "Nonspecific reaction to tuberculin skin test (TST/Mantoux) without active TB",
                 "DIFFERENTIAL · IF CULTURE NEGATIVE",
                 "If sputum culture returns negative, consider R76.11 — prior BCG vaccination or non-tuberculous mycobacteria (NTM) cross-reaction."),
            ]
        else:
            cards = [
                ("range",     "A15–A19","Tuberculosis — Range (Moderate Confidence)",
                 "RANGE · MODERATE CONFIDENCE",
                 "Moderate TB probability. Cannot confirm A15.0 at this confidence level. Recommend confirmatory workup before coding as active TB."),
                ("screening", "Z11.1",  "Encounter for screening for TB",
                 "SCREENING · RECOMMENDED ACTION",
                 "Initiate confirmatory screening: TST / IGRA (QuantiFERON-TB Gold) + sputum culture. Do not treat until bacteriological confirmation."),
                ("followup",  "R76.11", "Nonspecific reaction to TST without active TB",
                 "DIFFERENTIAL · BORDERLINE RESULT",
                 "Consider if TST positive but culture negative. BCG vaccination history and NTM exposure should be evaluated."),
            ]
        panel_extra_cls = ""
        header_color    = "#ff4560"
        header_icon     = "⚠️"
        header_text     = f"ICD-10-CM Codes &nbsp;·&nbsp; TB Prediction ({conf_pct:.1f}% confidence)"
        card_hover_cls  = ""

    # ── ICD cards for Normal prediction ────────────────────────────
    else:
        cards = [
            ("normal-p",  "Z03.89", "Encounter for observation for other suspected diseases ruled out",
             "PRIMARY · NO DISEASE FOUND",
             "Patient examined and tuberculosis ruled out at this time. No active TB infection detected by AI-assisted chest radiograph analysis."),
            ("screening", "Z11.1",  "Encounter for screening for respiratory TB",
             "SCREENING · ROUTINE",
             "This study constitutes a TB screening encounter. Result: negative. Recommend repeat screening per local TB control guidelines if patient is high-risk (contact tracing, endemic region)."),
            ("screening", "Z86.15", "Personal history of latent tuberculosis infection",
             "HISTORY · IF APPLICABLE",
             "Apply if patient has documented prior LTBI or positive IGRA. Latent TB does not appear on chest X-ray — requires clinical history review."),
            ("followup",  "Z11.7",  "Encounter for testing for latent tuberculosis infection",
             "LATENT TB TEST",
             "If clinical risk factors present despite negative CXR, order IGRA (QuantiFERON-TB Gold Plus) or TST to rule out latent TB infection (LTBI)."),
        ]
        panel_extra_cls = "icd-panel-normal"
        header_color    = "#00ff9d"
        header_icon     = "✅"
        header_text     = f"ICD-10-CM Codes &nbsp;·&nbsp; Normal Prediction ({conf_pct:.1f}% confidence)"
        card_hover_cls  = "icd-card-normal"

    # ── Build cards HTML ────────────────────────────────────────────
    cards_html = ""
    for tier_cls, code, short_desc, tier_label, long_desc in cards:
        code_color = ("#ff7043" if tier_cls in ("primary","range","followup") and is_tb
                      else "#00ff9d" if tier_cls == "normal-p"
                      else "#00d4ff" if tier_cls == "screening"
                      else "#ce93d8" if tier_cls == "followup"
                      else "#ffa040")
        cards_html += f"""
        <div class="icd-card {card_hover_cls}">
          <div class="icd-tier icd-{tier_cls}">{tier_label}</div>
          <div class="icd-code" style="color:{code_color}">{code}</div>
          <div class="icd-desc" style="color:#c8dce8;margin-bottom:0.4rem">{short_desc}</div>
          <div class="icd-desc">{long_desc}</div>
        </div>"""

    return f"""
    <div class="icd-panel {panel_extra_cls}">
      <div class="panel-title" style="color:{header_color}">
        {header_icon} &nbsp; {header_text}
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#5a7a8a;
                  line-height:1.7;margin-bottom:0.6rem;">
        ICD-10-CM classification per WHO coding standards &nbsp;·&nbsp;
        For billing, reporting, and clinical documentation.
        Sensitivity: <span style="color:#00ff9d">{SENSITIVITY}%</span> &nbsp;|&nbsp;
        Specificity: <span style="color:#00d4ff">{SPECIFICITY}%</span>
      </div>
      {ss_html}
      <div class="icd-grid" style="margin-top:1rem">{cards_html}</div>
      <div class="icd-disclaimer">
        ⚕ These ICD codes are AI-suggested for reference only and must be reviewed and confirmed
        by a licensed clinician or radiologist before use in clinical documentation, billing, or
        treatment decisions. This tool does not replace bacteriological confirmation (sputum
        culture, GeneXpert MTB/RIF, or IGRA). &nbsp;·&nbsp; ICD-10-CM source: WHO / CDC / CMS 2024.
      </div>
    </div>"""


# ─────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">Quantum · Medical AI</div>
  <h1>QTB Scan</h1>
  <p>Tuberculosis detection via Hybrid Quantum-Classical Neural Network · ResNet18 + QCNN (PennyLane)</p>
  <div class="hero-badge">
    MODEL &nbsp;·&nbsp; qtb_model_final.pth<br>
    BACKEND &nbsp;·&nbsp; PennyLane default.qubit<br>
    QUBITS &nbsp;·&nbsp; 4<br>
    LAYERS &nbsp;·&nbsp; 3 quantum (HEAD·MID·BODY)<br>
    CLASSES &nbsp;·&nbsp; Normal / TB
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.8], gap="large")

with left_col:
    st.markdown('<div class="panel-title">01 · Upload Chest X-Ray</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a chest X-ray (PNG, JPG, or DICOM)",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "dcm", "dicom"],
        label_visibility="collapsed",
    )

    pil_img = tensor = None

    if uploaded:
        pil_img, tensor, file_info = load_and_preprocess(uploaded)
        st.image(
            pil_img,
            use_container_width=True,
            channels="GRAY" if pil_img.mode == 'L' else "RGB",
        )
        st.markdown(
            f'<div class="chip-row">'
            f'<div class="chip">TYPE &nbsp;<span>{file_info}</span></div>'
            f'<div class="chip">FILE &nbsp;<span>{uploaded.name}</span></div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel-title">02 · Model Status</div>',
                unsafe_allow_html=True)

    model, cam_model = None, None
    try:
        with st.spinner("Building PennyLane quantum circuit & loading weights…"):
            model, cam_model = load_model()
        st.markdown(
            '<div class="chip-row">'
            '<div class="chip">STATUS &nbsp;<span style="color:#00ff9d">● LOADED</span></div>'
            '<div class="chip">BACKEND &nbsp;<span>PennyLane</span></div>'
            '<div class="chip">PHASE &nbsp;<span>3 (full · 16 params)</span></div>'
            '<div class="chip">CAM &nbsp;<span style="color:#00ff9d">lung-masked ✓</span></div>'
            '</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Model load failed: {e}")


# ─────────────────────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────────────────────
with right_col:
    if pil_img is not None and model is not None:

        with st.spinner("Running quantum inference…"):
            pred_class, probs = predict(model, tensor)

        # ── Ground-truth label from filename ──────────────────────
        gt_label, gt_str = parse_filename_label(uploaded.name)
        pred_str         = "TB / Abnormal" if pred_class == 1 else "Normal / Healthy"

        if gt_label is None:
            # No label hint in filename — neutral info chip
            st.markdown(f"""
            <div class="gt-unknown">
              <div class="gt-header" style="color:var(--text-dim)">
                🏷 Filename Ground Truth
              </div>
              <div class="gt-row">
                <div class="gt-cell">
                  <span class="lbl">Filename Label</span>
                  <span style="color:var(--text-dim)">Not detectable from filename</span>
                </div>
                <div class="gt-cell">
                  <span class="lbl">Model Prediction</span>
                  <span style="color:{'#ff4560' if pred_class==1 else '#00ff9d'}">{pred_str}</span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        elif gt_label == pred_class:
            # ✓ Model agrees with filename label
            st.markdown(f"""
            <div class="gt-match">
              <div class="gt-header" style="color:var(--green)">
                ✓ Model agrees with filename label
              </div>
              <div class="gt-row">
                <div class="gt-cell">
                  <span class="lbl">Filename Label (Ground Truth)</span>
                  <span style="color:var(--green)">{gt_str}</span>
                </div>
                <div class="gt-cell">
                  <span class="lbl">Model Prediction</span>
                  <span style="color:var(--green)">{pred_str}</span>
                </div>
                <div class="gt-cell">
                  <span class="lbl">Confidence</span>
                  <span style="color:var(--green)">{probs[pred_class]*100:.1f}%</span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        else:
            # ✗ Mismatch — model predicted wrong class vs filename
            gt_conf   = probs[gt_label] * 100
            pred_conf = probs[pred_class] * 100
            margin    = pred_conf - gt_conf
            st.markdown(f"""
            <div class="gt-mismatch">
              <div class="gt-header" style="color:var(--red)">
                ✗ FALSE NEGATIVE — Model disagrees with filename label
              </div>
              <div class="gt-row">
                <div class="gt-cell">
                  <span class="lbl">Filename Label (Ground Truth)</span>
                  <span style="color:var(--red);font-weight:700">{gt_str}</span>
                </div>
                <div class="gt-cell">
                  <span class="lbl">Model Predicted</span>
                  <span style="color:var(--amber)">{pred_str} ({pred_conf:.1f}%)</span>
                </div>
                <div class="gt-cell">
                  <span class="lbl">Correct class score</span>
                  <span style="color:var(--text-dim)">{CLASSES[gt_label]}: {gt_conf:.1f}%</span>
                </div>
                <div class="gt-cell">
                  <span class="lbl">Decision margin</span>
                  <span style="color:var(--amber)">+{margin:.1f}% toward wrong class</span>
                </div>
              </div>
              <div style="font-family:var(--mono);font-size:0.7rem;color:#ff9070;
                          margin-top:0.8rem;line-height:1.7;border-top:1px solid #ff456033;padding-top:0.7rem;">
                ⚠ This is a known model limitation — 7.3% of TB cases are missed (48/660 in test set).
                The ICMR DICOM images were used as training data only; the model may not generalise
                perfectly to all DICOM acquisition settings. The filename label takes precedence
                for research evaluation purposes.
              </div>
            </div>""", unsafe_allow_html=True)

        is_tb     = pred_class == 1
        card_cls  = "diag-tb" if is_tb else "diag-normal"
        lbl_cls   = "diag-result-tb" if is_tb else "diag-result-normal"
        result    = "TUBERCULOSIS DETECTED" if is_tb else "NORMAL — NO TB"
        icon      = "⚠️" if is_tb else "✅"
        conf_pct  = probs[pred_class] * 100
        certainty = ("Very High" if conf_pct > 90 else
                     "High"      if conf_pct > 75 else
                     "Moderate"  if conf_pct > 60 else "Low")

        st.markdown(f"""
        <div class="panel {card_cls}">
          <div class="diag-label">03 · Diagnosis Result</div>
          <div class="{lbl_cls}">{icon} {result}</div>
          <div class="conf-value" style="color:{'#ff4560' if is_tb else '#00ff9d'}">
            {conf_pct:.1f}% confidence &nbsp;·&nbsp; {certainty} certainty
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(
            '<div class="panel-title" style="margin-top:1rem">04 · Class Probabilities</div>',
            unsafe_allow_html=True)
        for i, (cls, fill) in enumerate(zip(CLASSES, ["prob-fill-normal", "prob-fill-tb"])):
            pct = probs[i] * 100
            st.markdown(f"""
            <div class="prob-row">
              <div class="prob-label"><span>{cls}</span><span>{pct:.2f}%</span></div>
              <div class="prob-track">
                <div class="{fill}" style="width:{pct:.1f}%"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        # ── ICD-10-CM Classification Panel ────────────────────────
        st.markdown(
            '<div class="panel-title" style="margin-top:1.4rem">'
            '05 · ICD-10-CM Clinical Classification</div>',
            unsafe_allow_html=True)
        st.markdown(icd_panel_html(pred_class, conf_pct), unsafe_allow_html=True)

        with st.spinner("Running 3-step CAM pipeline…"):
            try:
                cam_224, roi_soft, feature_scores = compute_cam(
                    cam_model, tensor, pred_class)
                cam_ok = True
            except Exception as e:
                st.warning(f"CAM failed: {e}")
                cam_ok = False

        if cam_ok:
            rgb_img = tensor_to_rgb(tensor)

            st.markdown("""
            <div style="margin-top:1.5rem;border-top:1px solid var(--border);padding-top:1.2rem;">
              <div class="panel-title">06 · Explainability — 3-Step CAM Pipeline</div>
              <div class="step-badge step-1">Step 1 · How the model decides</div>
              <div style="font-family:var(--mono);font-size:0.72rem;color:var(--text-dim);
                          line-height:1.8;margin-top:0.4rem;">
                Projector neuron activations fed into the quantum circuit.
                Bars show which neurons drove the classification and by how much.
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown(feature_bar_html(feature_scores, pred_class),
                        unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:1.4rem;">
              <div class="step-badge step-2">Step 2 · Where in the image</div>
              &nbsp;
              <div class="step-badge step-3">Step 3 · Region of Interest</div>
            </div>
            <div style="font-family:var(--mono);font-size:0.72rem;color:var(--text-dim);
                        line-height:1.8;margin:0.5rem 0 1rem;">
              Layer4 maps × channel importance → upsample →
              <span style="color:#ffa040">lung-masked</span> →
              65th-percentile ROI →
              <span style="color:{'#ff4560' if is_tb else '#00d4ff'}">contour overlay</span>
            </div>""", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3, gap="small")
            with c1:
                st.image(render_plain(rgb_img), use_container_width=True)
                st.markdown(
                    '<div class="gcam-caption">Preprocessed input<br>224×224 · normalised</div>',
                    unsafe_allow_html=True)
            with c2:
                st.image(render_heatmap(cam_224), use_container_width=True)
                st.markdown(
                    '<div class="gcam-caption">Raw activation heatmap<br>lung region only</div>',
                    unsafe_allow_html=True)
            with c3:
                st.image(render_roi_overlay(rgb_img, cam_224, roi_soft, pred_class),
                         use_container_width=True)
                st.markdown(
                    '<div class="gcam-caption">ROI overlay<br>contour = region of interest</div>',
                    unsafe_allow_html=True)

            dom_q   = int(np.argmax(np.abs(feature_scores))) + 1
            dom_val = feature_scores[dom_q - 1]
            dom_dir = "positive" if dom_val > 0 else "negative"
            interp  = (
                f"🔴 Quantum neuron <b>Q{dom_q}</b> was the dominant driver "
                f"({dom_dir}: {dom_val:+.3f}). The ROI contour marks the lung sub-region "
                f"where backbone channels activate most strongly — consistent with TB "
                f"consolidation, opacity, or cavitation patterns."
                if is_tb else
                f"🟢 Quantum neuron <b>Q{dom_q}</b> was the dominant driver "
                f"({dom_dir}: {dom_val:+.3f}). Activations are distributed across both lung "
                f"fields without focal consolidation — consistent with a normal chest X-ray."
            )
            st.markdown(f"""
            <div class="panel" style="margin-top:1rem;
              border-color:{'#ff456033' if is_tb else '#00ff9d33'}">
              <div class="panel-title">CAM Interpretation — Step 06</div>
              <div style="font-family:'DM Mono',monospace;font-size:0.78rem;
                          color:#c8dce8;line-height:1.8;">{interp}</div>
            </div>""", unsafe_allow_html=True)

    elif pil_img is None:
        st.markdown("""
        <div class="panel" style="text-align:center;padding:4rem 2rem;border-style:dashed;">
          <div style="font-size:3rem;margin-bottom:1rem;">🫁</div>
          <div style="font-family:'DM Mono',monospace;font-size:0.8rem;
                      color:#5a7a8a;line-height:2;letter-spacing:0.1em;">
            UPLOAD A CHEST X-RAY<br>TO BEGIN ANALYSIS
          </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  QUANTUM CIRCUIT EXPLORER  — full-width interactive section
# ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top:1px solid #1a3048;padding-top:2rem;margin-bottom:0.5rem;">
  <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.22em;
              text-transform:uppercase;color:#5a7a8a;margin-bottom:0.4rem;">
    07 · Quantum Circuit Explorer
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#c8dce8;
              line-height:1.8;margin-bottom:1rem;">
    Interactive diagram of the 4-qubit QCNN. Hover gates for details · click a layer
    to highlight it · toggle parameter labels · zoom with scroll.
  </div>
</div>""", unsafe_allow_html=True)

# Load trained weights so the circuit shows real trained values
_weights_for_circuit = None
try:
    _ckpt = torch.load(PTH_PATH, map_location="cpu", weights_only=False)
    _state = _ckpt["model_state_dict"] if isinstance(_ckpt, dict) else _ckpt
    _w = _state.get("qlayer.weights", None)
    if _w is not None:
        _weights_for_circuit = _w.detach().numpy().tolist()
except Exception:
    pass

# Fallback: zeros if weights not loaded
if _weights_for_circuit is None:
    _weights_for_circuit = [0.0] * 16

_w = _weights_for_circuit  # shorthand

import streamlit.components.v1 as components

_CIRCUIT_HTML = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #050c14; color: #c8dce8;
    font-family: 'Courier New', monospace; overflow-x: hidden;
  }}

  /* ── toolbar ── */
  #toolbar {{
    display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
    padding: 12px 16px; background: #0a1622;
    border-bottom: 1px solid #1a3048;
  }}
  .tb-btn {{
    background: #0f1e2e; border: 1px solid #1a3048; color: #5a7a8a;
    padding: 5px 14px; border-radius: 4px; font-size: 11px;
    letter-spacing: .1em; cursor: pointer; transition: all .15s;
    font-family: 'Courier New', monospace; text-transform: uppercase;
  }}
  .tb-btn:hover, .tb-btn.active {{ border-color: #00d4ff; color: #00d4ff; background: #00d4ff11; }}
  .tb-sep {{ width: 1px; height: 20px; background: #1a3048; }}
  .tb-label {{ font-size: 10px; color: #5a7a8a; letter-spacing: .1em; }}

  /* ── legend ── */
  #legend {{
    display: flex; gap: 16px; flex-wrap: wrap; padding: 8px 16px;
    background: #0a1622; border-bottom: 1px solid #1a3048; font-size: 10px;
  }}
  .leg-item {{ display: flex; align-items: center; gap: 6px; color: #5a7a8a; }}
  .leg-box  {{ width: 14px; height: 14px; border-radius: 3px; }}

  /* ── main svg area ── */
  #wrap {{
    overflow: auto; background: #050c14; padding: 20px 10px;
    cursor: grab; user-select: none;
  }}
  #wrap:active {{ cursor: grabbing; }}
  #circuit-svg {{ display: block; }}

  /* ── tooltip ── */
  #tip {{
    position: fixed; pointer-events: none;
    background: #0a1622; border: 1px solid #00d4ff55;
    border-radius: 6px; padding: 10px 14px;
    font-size: 11px; line-height: 1.7; color: #c8dce8;
    max-width: 260px; display: none; z-index: 999;
    box-shadow: 0 4px 20px #00000088;
  }}
  #tip .tip-title {{ color: #00d4ff; font-size: 12px; margin-bottom: 4px; font-weight: bold; }}
  #tip .tip-val   {{ color: #00ff9d; }}
  #tip .tip-dim   {{ color: #5a7a8a; font-size: 10px; }}

  /* ── phase info panel ── */
  #phaseinfo {{
    padding: 10px 16px; background: #0a1622;
    border-top: 1px solid #1a3048; font-size: 11px;
    min-height: 38px; color: #5a7a8a; letter-spacing: .05em;
    transition: all .2s;
  }}
  #phaseinfo span {{ color: #c8dce8; }}

  /* ── param table ── */
  #paramtable {{
    padding: 12px 16px; background: #050c14;
    border-top: 1px solid #1a3048; display: none;
  }}
  #paramtable table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
  #paramtable th {{ color: #5a7a8a; text-align: left; padding: 4px 8px;
                    border-bottom: 1px solid #1a3048; letter-spacing: .1em; font-size: 10px; }}
  #paramtable td {{ padding: 4px 8px; border-bottom: 1px solid #0f1e2e; }}
  #paramtable tr:hover td {{ background: #0a1622; }}
  .val-pos {{ color: #00ff9d; }}
  .val-neg {{ color: #ff4560; }}
</style>
</head>
<body>

<div id="toolbar">
  <span class="tb-label">HIGHLIGHT LAYER →</span>
  <button class="tb-btn active" onclick="setLayer('all')"  id="btn-all">All</button>
  <button class="tb-btn"        onclick="setLayer('embed')" id="btn-embed">Embedding</button>
  <button class="tb-btn"        onclick="setLayer('head')"  id="btn-head">HEAD</button>
  <button class="tb-btn"        onclick="setLayer('mid')"   id="btn-mid">MIDDLE</button>
  <button class="tb-btn"        onclick="setLayer('body')"  id="btn-body">BODY</button>
  <div class="tb-sep"></div>
  <button class="tb-btn" onclick="toggleLabels()" id="btn-lbl">Labels ON</button>
  <button class="tb-btn" onclick="toggleParams()" id="btn-par">Param Table</button>
  <button class="tb-btn" onclick="resetZoom()">Reset Zoom</button>
</div>

<div id="legend">
  <div class="leg-item"><div class="leg-box" style="background:#00d4ff33;border:1px solid #00d4ff88"></div>AngleEmbedding (RY)</div>
  <div class="leg-item"><div class="leg-box" style="background:#ffa04033;border:1px solid #ffa04088"></div>HEAD conv (RY + CNOT)</div>
  <div class="leg-item"><div class="leg-box" style="background:#a855f733;border:1px solid #a855f788"></div>MIDDLE pool+conv (CRZ + RY + CNOT)</div>
  <div class="leg-item"><div class="leg-box" style="background:#ff456033;border:1px solid #ff456088"></div>BODY pool+readout (CRZ + RY)</div>
  <div class="leg-item"><div class="leg-box" style="background:#00ff9d33;border:1px solid #00ff9d88"></div>Observable ⟨Z₃⟩</div>
</div>

<div id="wrap">
<svg id="circuit-svg" xmlns="http://www.w3.org/2000/svg"></svg>
</div>

<div id="phaseinfo">Hover a gate or click a layer button for details.</div>
<div id="paramtable"></div>
<div id="tip"></div>

<script>
// ── Trained weights (real values from checkpoint) ─────────────────
const W = {_w};
// HEAD: W[0..7]  MIDDLE: W[8..13]  BODY: W[14..15]

const H = W.slice(0,8);
const M = W.slice(8,14);
const B = W.slice(14,16);

function fmt(v) {{ return (v >= 0 ? '+' : '') + v.toFixed(4); }}
function fmtPi(v) {{ return fmt(v/Math.PI) + 'π  (' + fmt(v) + ' rad)'; }}

// ── Layout constants ──────────────────────────────────────────────
const WIRE_Y    = [70, 150, 230, 310];   // y for q0,q1,q2,q3
const WIRE_CLR  = ['#00d4ff','#ffa040','#a855f7','#00ff9d'];
const WIRE_LBLS = ['q₀','q₁','q₂','q₃'];
const SVG_H     = 390;
const LPAD      = 60;

// Gate column x-positions
const COL = {{
  embed : [110, 150, 190, 230],   // 4 angle-embed RY gates
  h0    : 300, h1: 340,           // HEAD q0,q1 first RY pair
  cx01  : 390,                    // CNOT (0→1)
  h2    : 440, h3: 480,           // HEAD q0,q1 second RY pair
  h4    : 540, h5: 580,           // HEAD q2,q3 first RY pair
  cx23  : 630,                    // CNOT (2→3)
  h6    : 680, h7: 720,           // HEAD q2,q3 second RY pair
  crz01 : 790,                    // MIDDLE CRZ(0→1)
  crz23 : 860,                    // MIDDLE CRZ(2→3)
  m2    : 930, m3: 970,           // MIDDLE RY on q1,q3
  cx13  : 1020,                   // CNOT (1→3)
  m4    : 1070, m5: 1110,         // MIDDLE RY on q1,q3
  crz13b: 1180,                   // BODY CRZ(1→3)
  b1    : 1250,                   // BODY RY on q3
  obs   : 1320,                   // Observable
}};
const SVG_W = 1410;

// ── Gate data ─────────────────────────────────────────────────────
// Each gate: {{ id, layer, type, x, qubits, label, tooltip_html }}
const GATES = [
  // EMBEDDING
  {{ id:'e0', layer:'embed', type:'ry', x:COL.embed[0], q:[0], label:'RY',
     tip:`<div class='tip-title'>AngleEmbedding — q₀</div>Input feature x₀ encoded as RY rotation<div class='tip-dim'>Rotation = x₀ (from projector·π)</div>` }},
  {{ id:'e1', layer:'embed', type:'ry', x:COL.embed[1], q:[1], label:'RY',
     tip:`<div class='tip-title'>AngleEmbedding — q₁</div>Input feature x₁<div class='tip-dim'>Rotation = x₁</div>` }},
  {{ id:'e2', layer:'embed', type:'ry', x:COL.embed[2], q:[2], label:'RY',
     tip:`<div class='tip-title'>AngleEmbedding — q₂</div>Input feature x₂<div class='tip-dim'>Rotation = x₂</div>` }},
  {{ id:'e3', layer:'embed', type:'ry', x:COL.embed[3], q:[3], label:'RY',
     tip:`<div class='tip-title'>AngleEmbedding — q₃</div>Input feature x₃<div class='tip-dim'>Rotation = x₃</div>` }},

  // HEAD
  {{ id:'h0', layer:'head', type:'ry', x:COL.h0, q:[0], label:'RY',
     tip:`<div class='tip-title'>HEAD Conv — RY(h₀) on q₀</div><div class='tip-val'>${{fmtPi(H[0])}}</div><div class='tip-dim'>Trainable · Phase 1+</div>` }},
  {{ id:'h1', layer:'head', type:'ry', x:COL.h1, q:[1], label:'RY',
     tip:`<div class='tip-title'>HEAD Conv — RY(h₁) on q₁</div><div class='tip-val'>${{fmtPi(H[1])}}</div><div class='tip-dim'>Trainable · Phase 1+</div>` }},
  {{ id:'cx01a', layer:'head', type:'cnot', x:COL.cx01, q:[0,1], label:'',
     tip:`<div class='tip-title'>CNOT q₀→q₁</div>Entangles q₀ and q₁<div class='tip-dim'>HEAD conv layer — creates superposition</div>` }},
  {{ id:'h2', layer:'head', type:'ry', x:COL.h2, q:[0], label:'RY',
     tip:`<div class='tip-title'>HEAD Conv — RY(h₂) on q₀</div><div class='tip-val'>${{fmtPi(H[2])}}</div><div class='tip-dim'>Trainable · Phase 1+</div>` }},
  {{ id:'h3', layer:'head', type:'ry', x:COL.h3, q:[1], label:'RY',
     tip:`<div class='tip-title'>HEAD Conv — RY(h₃) on q₁</div><div class='tip-val'>${{fmtPi(H[3])}}</div><div class='tip-dim'>Trainable · Phase 1+</div>` }},
  {{ id:'h4', layer:'head', type:'ry', x:COL.h4, q:[2], label:'RY',
     tip:`<div class='tip-title'>HEAD Conv — RY(h₄) on q₂</div><div class='tip-val'>${{fmtPi(H[4])}}</div><div class='tip-dim'>Trainable · Phase 1+</div>` }},
  {{ id:'h5', layer:'head', type:'ry', x:COL.h5, q:[3], label:'RY',
     tip:`<div class='tip-title'>HEAD Conv — RY(h₅) on q₃</div><div class='tip-val'>${{fmtPi(H[5])}}</div><div class='tip-dim'>Trainable · Phase 1+</div>` }},
  {{ id:'cx23a', layer:'head', type:'cnot', x:COL.cx23, q:[2,3], label:'',
     tip:`<div class='tip-title'>CNOT q₂→q₃</div>Entangles q₂ and q₃<div class='tip-dim'>HEAD conv layer</div>` }},
  {{ id:'h6', layer:'head', type:'ry', x:COL.h6, q:[2], label:'RY',
     tip:`<div class='tip-title'>HEAD Conv — RY(h₆) on q₂</div><div class='tip-val'>${{fmtPi(H[6])}}</div><div class='tip-dim'>Trainable · Phase 1+</div>` }},
  {{ id:'h7', layer:'head', type:'ry', x:COL.h7, q:[3], label:'RY',
     tip:`<div class='tip-title'>HEAD Conv — RY(h₇) on q₃</div><div class='tip-val'>${{fmtPi(H[7])}}</div><div class='tip-dim'>Trainable · Phase 1+</div>` }},

  // MIDDLE
  {{ id:'crz01', layer:'mid', type:'crz', x:COL.crz01, q:[0,1], label:'CRZ',
     tip:`<div class='tip-title'>Pool1 — CRZ(m₀) q₀→q₁</div><div class='tip-val'>${{fmtPi(M[0])}}</div><div class='tip-dim'>Pooling: collapses q₀ info into q₁ · Phase 2+</div>` }},
  {{ id:'crz23', layer:'mid', type:'crz', x:COL.crz23, q:[2,3], label:'CRZ',
     tip:`<div class='tip-title'>Pool1 — CRZ(m₁) q₂→q₃</div><div class='tip-val'>${{fmtPi(M[1])}}</div><div class='tip-dim'>Pooling: collapses q₂ info into q₃ · Phase 2+</div>` }},
  {{ id:'m2', layer:'mid', type:'ry', x:COL.m2, q:[1], label:'RY',
     tip:`<div class='tip-title'>MIDDLE Conv — RY(m₂) on q₁</div><div class='tip-val'>${{fmtPi(M[2])}}</div><div class='tip-dim'>Active qubit after Pool1 · Phase 2+</div>` }},
  {{ id:'m3', layer:'mid', type:'ry', x:COL.m3, q:[3], label:'RY',
     tip:`<div class='tip-title'>MIDDLE Conv — RY(m₃) on q₃</div><div class='tip-val'>${{fmtPi(M[3])}}</div><div class='tip-dim'>Active qubit after Pool1 · Phase 2+</div>` }},
  {{ id:'cx13', layer:'mid', type:'cnot', x:COL.cx13, q:[1,3], label:'',
     tip:`<div class='tip-title'>CNOT q₁→q₃</div>Entangles surviving qubits after pooling<div class='tip-dim'>MIDDLE conv layer</div>` }},
  {{ id:'m4', layer:'mid', type:'ry', x:COL.m4, q:[1], label:'RY',
     tip:`<div class='tip-title'>MIDDLE Conv — RY(m₄) on q₁</div><div class='tip-val'>${{fmtPi(M[4])}}</div><div class='tip-dim'>Phase 2+</div>` }},
  {{ id:'m5', layer:'mid', type:'ry', x:COL.m5, q:[3], label:'RY',
     tip:`<div class='tip-title'>MIDDLE Conv — RY(m₅) on q₃</div><div class='tip-val'>${{fmtPi(M[5])}}</div><div class='tip-dim'>Phase 2+</div>` }},

  // BODY
  {{ id:'crz13b', layer:'body', type:'crz', x:COL.crz13b, q:[1,3], label:'CRZ',
     tip:`<div class='tip-title'>Pool2 — CRZ(b₀) q₁→q₃</div><div class='tip-val'>${{fmtPi(B[0])}}</div><div class='tip-dim'>Final pooling — all info collapsed to q₃ · Phase 3</div>` }},
  {{ id:'b1', layer:'body', type:'ry', x:COL.b1, q:[3], label:'RY',
     tip:`<div class='tip-title'>BODY Readout — RY(b₁) on q₃</div><div class='tip-val'>${{fmtPi(B[1])}}</div><div class='tip-dim'>Final rotation before measurement · Phase 3</div>` }},

  // OBSERVABLE
  {{ id:'obs', layer:'obs', type:'meas', x:COL.obs, q:[3], label:'⟨Z⟩',
     tip:`<div class='tip-title'>Observable ⟨PauliZ⟩ on q₃</div>Expectation value ∈ [−1, +1]<div class='tip-dim'>→ Linear head → 2-class logits</div>` }},
];

// ── Layer background regions ──────────────────────────────────────
const REGIONS = [
  {{ id:'embed', x1:85,   x2:260,  color:'#00d4ff', label:'ANGLE EMBEDDING' }},
  {{ id:'head',  x1:275,  x2:745,  color:'#ffa040', label:'HEAD  (Conv Layer 1)  ·  h₀…h₇' }},
  {{ id:'mid',   x1:765,  x2:1145, color:'#a855f7', label:'MIDDLE  (Pool1 + Conv Layer 2)  ·  m₀…m₅' }},
  {{ id:'body',  x1:1160, x2:1295, color:'#ff4560', label:'BODY  (Pool2 + Readout)  ·  b₀,b₁' }},
  {{ id:'obs',   x1:1300, x2:1400, color:'#00ff9d', label:'OBSERVABLE  ⟨Z₃⟩' }},
];

// ── State ─────────────────────────────────────────────────────────
let activeLayer = 'all';
let showLabels  = true;
let isDragging  = false, dragStartX, dragStartY, scrollLeft, scrollTop;

// ── Build SVG ─────────────────────────────────────────────────────
const svg = document.getElementById('circuit-svg');
svg.setAttribute('width',  SVG_W);
svg.setAttribute('height', SVG_H);
svg.setAttribute('viewBox', `0 0 ${{SVG_W}} ${{SVG_H}}`);

function mkEl(tag, attrs, parent) {{
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}}

function buildSVG() {{
  svg.innerHTML = '';

  // Background
  mkEl('rect', {{x:0,y:0,width:SVG_W,height:SVG_H,fill:'#050c14'}}, svg);

  // Layer regions (background bands)
  REGIONS.forEach(r => {{
    const dim = (activeLayer !== 'all' && activeLayer !== r.id) ? 0.06 : 0.12;
    const g = mkEl('g', {{}}, svg);
    mkEl('rect', {{
      x:r.x1, y:20, width:r.x2-r.x1, height:SVG_H-40,
      fill:r.color, opacity:dim, rx:6
    }}, g);
    // Region label
    const lbl = mkEl('text', {{
      x:(r.x1+r.x2)/2, y:16,
      fill: (activeLayer==='all'||activeLayer===r.id) ? r.color : '#2a3a4a',
      'font-size':9, 'text-anchor':'middle',
      'font-family':'Courier New,monospace', 'letter-spacing':'0.08em',
      opacity: (activeLayer==='all'||activeLayer===r.id) ? 0.9 : 0.3,
    }}, g);
    lbl.textContent = r.label;
  }});

  // Wire lines
  WIRE_Y.forEach((y, qi) => {{
    const wireGroup = mkEl('g', {{}}, svg);
    // Label
    const wlbl = mkEl('text', {{
      x:LPAD-8, y:y+4, fill:WIRE_CLR[qi],
      'font-size':12, 'text-anchor':'end',
      'font-family':'Courier New,monospace', 'font-weight':'bold'
    }}, wireGroup);
    wlbl.textContent = WIRE_LBLS[qi];

    // Wire itself (dashed where qubit is "pooled away")
    // q0: pooled after MIDDLE CRZ → x > crz01
    // q2: pooled after MIDDLE CRZ → x > crz23
    // q1: pooled after BODY CRZ  → x > crz13b
    const fullEnd = (qi===0) ? COL.crz01+10
                  : (qi===2) ? COL.crz23+10
                  : (qi===1) ? COL.crz13b+10
                  : SVG_W - 30;

    mkEl('line', {{
      x1:LPAD, y1:y, x2:fullEnd, y2:y,
      stroke:WIRE_CLR[qi], 'stroke-width':1.5, opacity:0.5
    }}, wireGroup);

    // Faded continuation after pooling (except q3 which goes all the way)
    if (qi !== 3) {{
      mkEl('line', {{
        x1:fullEnd, y1:y, x2:SVG_W-30, y2:y,
        stroke:WIRE_CLR[qi], 'stroke-width':0.5,
        'stroke-dasharray':'4,6', opacity:0.18
      }}, wireGroup);
    }}
  }});

  // ── Draw gates ──────────────────────────────────────────────────
  GATES.forEach(g => {{
    const active = (activeLayer === 'all' || activeLayer === g.layer);
    const gEl    = mkEl('g', {{'class':'gate', 'data-id':g.id, opacity: active?1:0.18}}, svg);
    gEl.style.cursor = 'pointer';

    if (g.type === 'ry') {{
      const y = WIRE_Y[g.q[0]];
      // Box
      mkEl('rect', {{
        x:g.x-16, y:y-16, width:32, height:32, rx:5,
        fill: layerFill(g.layer), stroke: layerStroke(g.layer),
        'stroke-width':1.5
      }}, gEl);
      // Gate label
      mkEl('text', {{
        x:g.x, y:y+4, fill:'#ffffff', 'font-size':10,
        'text-anchor':'middle', 'font-family':'Courier New,monospace',
        'font-weight':'bold'
      }}, gEl).textContent = g.label;
      // Param label below gate
      if (showLabels && g.layer !== 'embed') {{
        const paramIdx = getParamLabel(g.id);
        const valText  = mkEl('text', {{
          x:g.x, y:y+36, fill: layerStroke(g.layer),
          'font-size':7.5, 'text-anchor':'middle',
          'font-family':'Courier New,monospace', opacity:0.8
        }}, gEl);
        valText.textContent = paramIdx;
      }}

    }} else if (g.type === 'cnot') {{
      const [qc, qt] = g.q;
      const yc = WIRE_Y[qc], yt = WIRE_Y[qt];
      // Vertical line
      mkEl('line', {{x1:g.x,y1:yc,x2:g.x,y2:yt,
        stroke:'#ffffff', 'stroke-width':1.5, opacity:0.6}}, gEl);
      // Control dot
      mkEl('circle', {{cx:g.x, cy:yc, r:5,
        fill:'#ffffff', opacity:0.85}}, gEl);
      // Target circle
      mkEl('circle', {{cx:g.x, cy:yt, r:14,
        fill:'none', stroke:'#ffffff', 'stroke-width':1.5, opacity:0.85}}, gEl);
      mkEl('line', {{x1:g.x,y1:yt-14,x2:g.x,y2:yt+14,
        stroke:'#ffffff', 'stroke-width':1.2, opacity:0.7}}, gEl);
      mkEl('line', {{x1:g.x-14,y1:yt,x2:g.x+14,y2:yt,
        stroke:'#ffffff', 'stroke-width':1.2, opacity:0.7}}, gEl);

    }} else if (g.type === 'crz') {{
      const [qc, qt] = g.q;
      const yc = WIRE_Y[qc], yt = WIRE_Y[qt];
      const col = layerStroke(g.layer);
      // Vertical line
      mkEl('line', {{x1:g.x,y1:yc,x2:g.x,y2:yt,
        stroke:col, 'stroke-width':1.5, opacity:0.7}}, gEl);
      // Control dot
      mkEl('circle', {{cx:g.x, cy:yc, r:5, fill:col, opacity:0.9}}, gEl);
      // Target box
      mkEl('rect', {{x:g.x-18,y:yt-16,width:36,height:32,rx:5,
        fill:layerFill(g.layer), stroke:col, 'stroke-width':1.5}}, gEl);
      mkEl('text', {{x:g.x, y:yt+4, fill:'#ffffff', 'font-size':9,
        'text-anchor':'middle','font-family':'Courier New,monospace',
        'font-weight':'bold'}}, gEl).textContent = 'CRZ';
      // Param label
      if (showLabels) {{
        const paramIdx = getParamLabel(g.id);
        mkEl('text', {{x:g.x,y:yt+36, fill:col,
          'font-size':7.5,'text-anchor':'middle',
          'font-family':'Courier New,monospace',opacity:0.8}}, gEl
        ).textContent = paramIdx;
      }}

    }} else if (g.type === 'meas') {{
      const y = WIRE_Y[g.q[0]];
      // Double border measurement box
      mkEl('rect', {{x:g.x-22,y:y-20,width:44,height:40,rx:6,
        fill:'#00ff9d22', stroke:'#00ff9d', 'stroke-width':2}}, gEl);
      mkEl('rect', {{x:g.x-18,y:y-16,width:36,height:32,rx:4,
        fill:'none', stroke:'#00ff9d55', 'stroke-width':1}}, gEl);
      mkEl('text', {{x:g.x,y:y+4, fill:'#00ff9d','font-size':12,
        'text-anchor':'middle','font-family':'Courier New,monospace',
        'font-weight':'bold'}}, gEl).textContent = '⟨Z⟩';
    }}

    // Hover handlers
    gEl.addEventListener('mouseenter', e => showTip(e, g.tip));
    gEl.addEventListener('mousemove',  e => moveTip(e));
    gEl.addEventListener('mouseleave', hideTip);
  }});

  // Phase separators
  const SEPS = [
    {{ x:COL.crz01-30, label:'Pool 1', color:'#a855f7' }},
    {{ x:COL.crz13b-30, label:'Pool 2', color:'#ff4560' }},
  ];
  SEPS.forEach(s => {{
    mkEl('line', {{x1:s.x,y1:30,x2:s.x,y2:SVG_H-30,
      stroke:s.color,'stroke-width':0.8,'stroke-dasharray':'5,4',opacity:0.4}}, svg);
    mkEl('text', {{x:s.x+3,y:SVG_H-18, fill:s.color,'font-size':8,
      'font-family':'Courier New,monospace',opacity:0.7}}, svg
    ).textContent = s.label;
  }});
}}

function layerFill(l) {{
  return {{embed:'#00d4ff22',head:'#ffa04022',mid:'#a855f722',body:'#ff456022',obs:'#00ff9d22'}}[l] || '#ffffff11';
}}
function layerStroke(l) {{
  return {{embed:'#00d4ff',head:'#ffa040',mid:'#a855f7',body:'#ff4560',obs:'#00ff9d'}}[l] || '#ffffff';
}}
function getParamLabel(id) {{
  const map = {{
    h0:'h₀',h1:'h₁',h2:'h₂',h3:'h₃',h4:'h₄',h5:'h₅',h6:'h₆',h7:'h₇',
    crz01:'m₀',crz23:'m₁',m2:'m₂',m3:'m₃',m4:'m₄',m5:'m₅',
    crz13b:'b₀',b1:'b₁'
  }};
  return map[id] || '';
}}

// ── Tooltip ───────────────────────────────────────────────────────
const tip = document.getElementById('tip');
function showTip(e, html) {{
  tip.innerHTML = html; tip.style.display = 'block'; moveTip(e);
}}
function moveTip(e) {{
  let x = e.clientX + 14, y = e.clientY - 10;
  if (x + 280 > window.innerWidth) x = e.clientX - 290;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}}
function hideTip()  {{ tip.style.display = 'none'; }}

// ── Toolbar actions ───────────────────────────────────────────────
function setLayer(l) {{
  activeLayer = l;
  ['all','embed','head','mid','body'].forEach(id => {{
    const b = document.getElementById('btn-' + id);
    if (b) b.classList.toggle('active', id === l);
  }});
  const infos = {{
    all:   'Showing all layers. Click a layer to highlight.',
    embed: '<span>ANGLE EMBEDDING</span> — 4 RY gates encode the 4 projector outputs (×π) as qubit rotation angles. No trainable parameters.',
    head:  '<span>HEAD — Conv Layer 1</span> — 8 trainable params (h₀…h₇). Two 2-qubit convolutional unitaries: RY·CNOT·RY on (q₀,q₁) and (q₂,q₃). Trained in Phase 1.',
    mid:   '<span>MIDDLE — Pool1 + Conv Layer 2</span> — 6 trainable params (m₀…m₅). CRZ gates pool q₀→q₁ and q₂→q₃, then a second conv on surviving qubits q₁,q₃. Unlocked in Phase 2.',
    body:  '<span>BODY — Pool2 + Readout</span> — 2 trainable params (b₀,b₁). Final CRZ pools q₁→q₃, RY fine-tunes the readout qubit. Unlocked in Phase 3.',
  }};
  document.getElementById('phaseinfo').innerHTML = infos[l] || '';
  buildSVG();
}}

function toggleLabels() {{
  showLabels = !showLabels;
  document.getElementById('btn-lbl').textContent = showLabels ? 'Labels ON' : 'Labels OFF';
  buildSVG();
}}

function toggleParams() {{
  const el = document.getElementById('paramtable');
  const show = el.style.display === 'none' || el.style.display === '';
  el.style.display = show ? 'block' : 'none';
  document.getElementById('btn-par').classList.toggle('active', show);
  if (show) buildParamTable();
}}

function buildParamTable() {{
  const rows = [
    ['HEAD',   'h₀','q₀ RY 1', H[0]],['HEAD','h₁','q₁ RY 1', H[1]],
    ['HEAD',   'h₂','q₀ RY 2', H[2]],['HEAD','h₃','q₁ RY 2', H[3]],
    ['HEAD',   'h₄','q₂ RY 1', H[4]],['HEAD','h₅','q₃ RY 1', H[5]],
    ['HEAD',   'h₆','q₂ RY 2', H[6]],['HEAD','h₇','q₃ RY 2', H[7]],
    ['MIDDLE', 'm₀','CRZ q₀→q₁',M[0]],['MIDDLE','m₁','CRZ q₂→q₃',M[1]],
    ['MIDDLE', 'm₂','q₁ RY 1',  M[2]],['MIDDLE','m₃','q₃ RY 1',  M[3]],
    ['MIDDLE', 'm₄','q₁ RY 2',  M[4]],['MIDDLE','m₅','q₃ RY 2',  M[5]],
    ['BODY',   'b₀','CRZ q₁→q₃',B[0]],['BODY','b₁','q₃ RY',   B[1]],
  ];
  const lc = {{HEAD:'#ffa040',MIDDLE:'#a855f7',BODY:'#ff4560'}};
  let html = `<table><tr><th>Layer</th><th>Param</th><th>Gate / Qubit</th>
    <th>Value (rad)</th><th>Value (×π)</th><th>Bar</th></tr>`;
  rows.forEach(([layer,param,desc,val]) => {{
    const col  = lc[layer] || '#c8dce8';
    const vcls = val >= 0 ? 'val-pos' : 'val-neg';
    const pct  = Math.abs(val) / Math.PI * 100;
    const bar  = `<div style="width:${{Math.min(pct,100).toFixed(1)}}%;height:6px;
      background:${{col}};border-radius:3px;min-width:2px"></div>`;
    html += `<tr>
      <td style="color:${{col}}">${{layer}}</td>
      <td style="color:${{col}}">${{param}}</td>
      <td style="color:#8899aa">${{desc}}</td>
      <td class="${{vcls}}">${{fmt(val)}}</td>
      <td class="${{vcls}}">${{fmt(val/Math.PI)}}π</td>
      <td style="min-width:80px">${{bar}}</td>
    </tr>`;
  }});
  html += '</table>';
  document.getElementById('paramtable').innerHTML = html;
}}

// ── Pan/drag on the SVG wrap ──────────────────────────────────────
const wrap = document.getElementById('wrap');
wrap.addEventListener('mousedown', e => {{
  isDragging = true;
  dragStartX = e.pageX - wrap.offsetLeft;
  dragStartY = e.pageY - wrap.offsetTop;
  scrollLeft = wrap.scrollLeft;
  scrollTop  = wrap.scrollTop;
}});
wrap.addEventListener('mousemove', e => {{
  if (!isDragging) return;
  e.preventDefault();
  wrap.scrollLeft = scrollLeft - (e.pageX - wrap.offsetLeft - dragStartX);
  wrap.scrollTop  = scrollTop  - (e.pageY - wrap.offsetTop  - dragStartY);
}});
['mouseup','mouseleave'].forEach(ev =>
  wrap.addEventListener(ev, () => isDragging = false));

function resetZoom() {{
  wrap.scrollLeft = 0; wrap.scrollTop = 0;
}}

// ── Init ──────────────────────────────────────────────────────────
buildSVG();
</script>
</body>
</html>
"""

components.html(_CIRCUIT_HTML, height=620, scrolling=False)


st.markdown("""
<div class="footer">
  QTB SCAN &nbsp;·&nbsp; HYBRID QUANTUM-CLASSICAL AI &nbsp;·&nbsp;
  ResNet18 + QCNN (PennyLane · 4 qubits · 3 layers) &nbsp;·&nbsp;
  FOR RESEARCH USE ONLY — NOT A CLINICAL DIAGNOSTIC TOOL
</div>""", unsafe_allow_html=True)