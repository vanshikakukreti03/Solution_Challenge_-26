import os
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data', 'elliptic')
EGO_FEATURE_COUNT = 94
STRUCTURAL_FEATURE_COUNT = 72
TOTAL_FEATURES = EGO_FEATURE_COUNT + STRUCTURAL_FEATURE_COUNT

# Model
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.3
NUM_CLASSES = 2

# Training
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 100
PATIENCE = 15

# Audit
IG_STEPS = 50
AUDIT_SAMPLE_SIZE = 200
STRUCTURAL_BIAS_THRESHOLD = 0.65
EGO_DRIVEN_THRESHOLD = 0.35

# Dashboard
FLASK_HOST = '127.0.0.1'
FLASK_PORT = 5000

# Results
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'audit_results.json')
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', 'fraud_gcn.pt')

# Google Cloud (optional)
USE_VERTEX_AI = os.environ.get('USE_VERTEX_AI', 'false').lower() == 'true'
USE_BIGQUERY = os.environ.get('USE_BIGQUERY', 'false').lower() == 'true'
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID', '')
