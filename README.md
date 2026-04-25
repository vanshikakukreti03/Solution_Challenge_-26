# FairGraph-Audit

**Bridging the Feature Reliance Audit Gap in GNN-Based Fraud Detection**

A forensic auditing tool for Graph Neural Networks that quantifies whether a fraud flag was triggered by a user's own behavior (**Ego-features**) or purely by their neighborhood connections (**Structural context**). Ensures compliance with **RBI FREE-AI "Seven Sutras"** and **EU AI Act Article 86**.

---

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│   Elliptic    │───▶│  FraudGCN    │───▶│ Feature          │
│   Dataset     │    │  (3-layer    │    │ Attribution      │
│   (PyG)       │    │   GCN)       │    │ (Integrated      │
└──────────────┘    └──────────────┘    │  Gradients)      │
                                        └────────┬─────────┘
                                                 │
                    ┌──────────────┐    ┌────────▼─────────┐
                    │ Remediation  │◀───│ Bias Detector    │
                    │ Engine       │    │ (GbA, Disparate  │
                    └──────┬───────┘    │  Impact, Leakage)│
                           │            └──────────────────┘
                    ┌──────▼───────┐
                    │  Dashboard   │
                    │  (Flask +    │
                    │   Plotly)    │
                    └──────────────┘
```

## Quick Start

### Demo Mode (no ML dependencies needed)
```bash
pip install flask numpy plotly
python run.py --demo
```

### Full Pipeline
```bash
pip install -r requirements.txt
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
python run.py
```

Open **http://127.0.0.1:5000** in your browser.

## Project Structure

```
├── config.py                  # Central configuration
├── run.py                     # Main entry point
├── data/
│   └── loader.py              # Elliptic dataset loader (PyG + synthetic fallback)
├── models/
│   ├── gnn.py                 # FraudGCN (3-layer GCN with BatchNorm)
│   └── trainer.py             # Training loop with class-weighted loss
├── audit/
│   ├── attributor.py          # Integrated Gradients + neighborhood perturbation
│   ├── bias_detector.py       # Guilt-by-association, disparate impact detection
│   └── remediator.py          # Actionable fix recommendations
├── dashboard/
│   ├── app.py                 # Flask application
│   ├── static/css/style.css   # Premium dark theme
│   ├── static/js/dashboard.js # Interactive Plotly charts
│   └── templates/index.html   # Dashboard UI
└── results/
    └── audit_results.json     # Generated audit output
```

## Key Metrics

| Metric | Description |
|--------|-------------|
| **Reliance Ratio** | `structural_attribution / (ego + structural)` — higher = more neighborhood-driven |
| **Guilt-by-Association** | Nodes flagged as fraud with >65% structural reliance |
| **Disparate Impact** | FPR ratio between high-structural and low-structural groups |

## Regulatory Compliance

- **RBI FREE-AI "Understandable by Design"**: Per-feature Integrated Gradients attribution
- **RBI FREE-AI "Fairness"**: Disparate impact and equalized odds monitoring
- **EU AI Act Article 86**: Right to explanation via ego/structural decomposition

## Google Cloud Integration

Set environment variables for optional cloud deployment:
```bash
export USE_VERTEX_AI=true
export USE_BIGQUERY=true
export GCP_PROJECT_ID=your-project-id
```

## Tech Stack

- **PyTorch Geometric** — GNN framework
- **Integrated Gradients** — Feature attribution
- **Flask + Plotly.js** — Interactive dashboard
- **Google Vertex AI** — Model hosting (optional)
- **Google BigQuery** — Audit storage (optional)
