"""
Configuration StyleGAN2 — Auto-détection de l'environnement.

Aucun changement manuel nécessaire :
  - En local sur Mac  -> détecte automatiquement
  - Sur RunPod        -> détecte automatiquement
"""

import os

# ── Auto-détection de l'environnement ────────────────────────────────────────
# RunPod : les données sont dans /workspace/Data/ffhq256
# Local  : les données sont dans data/ffhq256_sample

RUNPOD_DATA = "/workspace/Data/ffhq256"
MODE = "runpod" if os.path.exists(RUNPOD_DATA) else "local"

# ── Chemins ───────────────────────────────────────────────────────────────────
if MODE == "runpod":
    DATA_PATH   = RUNPOD_DATA
    OUTPUT_PATH = "/workspace/outputs/stylegan2"
    NUM_EPOCHS  = 200
    BATCH_SIZE  = 32
    NUM_WORKERS = 4
else:
    # Chemin local relatif au dossier du projet
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH     = os.path.join(_PROJECT_ROOT, "data", "ffhq256_sample")
    OUTPUT_PATH   = os.path.join(_PROJECT_ROOT, "stylegan2", "outputs")
    NUM_EPOCHS    = 2
    BATCH_SIZE    = 4
    NUM_WORKERS   = 0

# ── Sous-dossiers outputs ─────────────────────────────────────────────────────
CHECKPOINTS_DIR    = os.path.join(OUTPUT_PATH, "checkpoints")
GENERATED_IMGS_DIR = os.path.join(OUTPUT_PATH, "generated_images")
METRICS_DIR        = os.path.join(OUTPUT_PATH, "metrics")

# ── Hyperparamètres modèle ────────────────────────────────────────────────────
Z_DIM      = 512
W_DIM      = 512
IMG_SIZE   = 256
CHANNELS   = 3

# ── Hyperparamètres entraînement ──────────────────────────────────────────────
LR_G       = 0.002
LR_D       = 0.002
BETA1      = 0.0
BETA2      = 0.99
R1_GAMMA   = 10.0
R1_EVERY   = 16
SAVE_EVERY = 10
