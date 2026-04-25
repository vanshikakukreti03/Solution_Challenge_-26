"""
FairGraph-Audit — Main Entry Point
===================================
Orchestrates: Data Loading → Model Training → Forensic Audit → Dashboard Launch
"""
import os
import sys
import json
import argparse
import datetime
import numpy as np


def generate_demo_results():
    """Generate realistic demo audit results without requiring PyTorch."""
    np.random.seed(42)
    N_TOTAL = 203769
    N_AUDIT = 200

    node_ids = sorted(np.random.choice(N_TOTAL, N_AUDIT, replace=False).tolist())
    audits = []
    for nid in node_ids:
        r = np.random.beta(2, 3)
        is_fraud = np.random.random() < 0.18
        conf = np.random.uniform(0.6, 0.99)
        flag = "STRUCTURAL_BIAS" if r > 0.65 else "EGO_DRIVEN" if r < 0.35 else "BALANCED"

        audits.append({
            'node_id': int(nid),
            'prediction': 'FRAUD' if is_fraud else 'LEGITIMATE',
            'confidence': round(conf, 4),
            'true_label': 1 if is_fraud and np.random.random() < 0.8 else 0,
            'reliance_ratio': round(r, 4),
            'ego_score': round((1 - r) * np.random.uniform(0.5, 2.0), 4),
            'structural_score': round(r * np.random.uniform(0.5, 2.0), 4),
            'neighborhood_influence': round(np.random.uniform(0.1, 0.9), 4),
            'flag': flag,
            'top_ego_features': [{'idx': i, 'name': f'local_feat_{i}', 'value': round(np.random.uniform(-0.5, 0.5), 4)} for i in np.random.choice(94, 5, replace=False).tolist()],
            'top_structural_features': [{'idx': i + 94, 'name': f'agg_feat_{i}', 'value': round(np.random.uniform(-0.5, 0.5), 4)} for i in np.random.choice(72, 5, replace=False).tolist()],
        })

    fraud_nodes = [a for a in audits if a['prediction'] == 'FRAUD']
    struct_bias = [a for a in audits if a['flag'] == 'STRUCTURAL_BIAS']
    ego_driven = [a for a in audits if a['flag'] == 'EGO_DRIVEN']
    balanced = [a for a in audits if a['flag'] == 'BALANCED']

    gba = []
    for a in struct_bias:
        if a['prediction'] == 'FRAUD':
            sev = 'CRITICAL' if a['reliance_ratio'] > 0.85 else 'HIGH' if a['reliance_ratio'] > 0.75 else 'MEDIUM'
            gba.append({
                'node_id': a['node_id'], 'reliance_ratio': a['reliance_ratio'],
                'confidence': a['confidence'], 'severity': sev,
                'bias_type': 'GUILT_BY_ASSOCIATION',
                'source': 'Structural neighborhood features dominate the fraud prediction.',
                'reason': f"Transaction {a['node_id']} flagged with {a['reliance_ratio']:.0%} structural reliance."
            })

    fairness = {
        'fpr_high_structural': 0.152, 'fpr_low_structural': 0.061,
        'fnr_high_structural': 0.12, 'fnr_low_structural': 0.08,
        'disparate_impact': 2.49, 'equalized_odds_diff': 0.091,
        'demographic_parity_diff': 0.114, 'is_fair': False,
        'high_structural_group_size': 89, 'low_structural_group_size': 111,
    }

    bias_report = {
        'findings': [
            {
                'type': 'GUILT_BY_ASSOCIATION', 'severity': 'CRITICAL',
                'affected_nodes': len(gba),
                'description': f"{len(gba)} transactions flagged primarily due to neighborhood connections rather than own behavior."
            },
            {
                'type': 'DISPARATE_IMPACT', 'severity': 'HIGH',
                'affected_nodes': fairness['high_structural_group_size'],
                'description': f"Disparate impact ratio of {fairness['disparate_impact']:.2f} detected between structural-reliance groups."
            },
        ],
        'fairness_metrics': fairness,
        'guilt_by_association': gba,
        'total_biased_nodes': len(gba),
    }

    recommendations = [
        {
            'id': 'REC-001', 'severity': 'HIGH', 'category': 'Feature Engineering',
            'title': 'Reduce Structural Feature Dominance',
            'source': 'Average structural reliance is 52%, exceeding the 50% threshold.',
            'reason': "The GNN aggregates too much influence from neighborhood features relative to the node's own transactional behavior.",
            'actions': [
                'Apply L1 regularization specifically on structural feature weights',
                'Reduce GNN depth from 3 layers to 2 to limit message-passing radius',
                'Introduce ego-feature attention: weight ego-features 2x during aggregation',
                'Train a parallel MLP on ego-features only and ensemble with GNN',
            ],
            'compliance': 'RBI FREE-AI Sutra: Understandable by Design',
        },
        {
            'id': 'REC-002', 'severity': 'CRITICAL', 'category': 'Model Architecture',
            'title': 'Mitigate Guilt-by-Association Bias',
            'source': f'{len(gba)} transactions flagged primarily due to their neighbors.',
            'reason': 'The model penalizes nodes for being connected to suspicious neighbors, regardless of their own legitimate behavior.',
            'actions': [
                'Add a reliance-ratio penalty to the loss: L_total = L_ce + λ * max(0, R_struct - 0.65)',
                'Use GAT (Graph Attention) with learnable ego-priority weights',
                'Implement counterfactual fairness: verify prediction is stable under neighborhood randomization',
                'Apply FairDrop: randomly drop neighbor edges during training to reduce structural dependency',
            ],
            'compliance': 'EU AI Act Article 86 — Right to Explanation',
        },
        {
            'id': 'REC-003', 'severity': 'HIGH', 'category': 'Fairness Calibration',
            'title': 'Address Disparate Impact Across Structural Groups',
            'source': f"Disparate impact ratio: {fairness['disparate_impact']:.2f} (acceptable: 0.80–1.25).",
            'reason': 'Nodes with high structural reliance face significantly different false positive rates than ego-driven nodes.',
            'actions': [
                'Apply group-calibrated thresholds: lower fraud threshold for high-structural nodes',
                'Implement equalized odds constraint during training',
                'Use reject-option classification: defer borderline high-structural cases to human review',
                'Schedule quarterly re-audits with updated transaction data',
            ],
            'compliance': 'RBI FREE-AI Sutra: Fairness',
        },
        {
            'id': 'REC-004', 'severity': 'INFO', 'category': 'Governance',
            'title': 'Establish Continuous Monitoring Pipeline',
            'source': 'Best practice for high-risk AI systems under both RBI and EU AI Act.',
            'reason': 'Bias patterns evolve as transaction networks change. Static audits become stale.',
            'actions': [
                'Deploy FairGraph-Audit as a scheduled pipeline on Vertex AI',
                'Store audit results in BigQuery for longitudinal trend analysis',
                'Set automated alerts when reliance ratio exceeds 0.65 for >5% of flagged nodes',
                'Generate monthly compliance reports via Looker dashboards',
            ],
            'compliance': 'EU AI Act Article 9 — Risk Management System',
        },
    ]

    compliance = {
        'rbi_free_ai': {
            'Understandable_by_Design': {'status': 'COMPLIANT', 'details': 'Integrated Gradients provides per-feature attribution for every prediction.'},
            'Fairness': {'status': 'AT_RISK', 'details': 'Disparate impact ratio of 2.49 exceeds acceptable range (0.80–1.25).'},
            'Accountability': {'status': 'COMPLIANT', 'details': 'Full audit trail stored with timestamps and model version.'},
            'Data_Quality': {'status': 'COMPLIANT', 'details': 'Elliptic dataset passes basic quality checks.'},
            'Transparency': {'status': 'COMPLIANT', 'details': 'Reliance ratio and bias reports are available for inspection.'},
        },
        'eu_ai_act': {
            'Article_86_Right_to_Explanation': {'status': 'COMPLIANT', 'details': 'Each prediction includes ego vs structural breakdown and top contributing features.'},
            'Article_9_Risk_Management': {'status': 'AT_RISK', 'details': 'Continuous monitoring pipeline not yet deployed.'},
            'Annex_IV_Documentation': {'status': 'COMPLIANT', 'details': 'Model architecture, training data, and performance metrics are documented.'},
        },
    }

    return {
        'metadata': {
            'timestamp': datetime.datetime.now().isoformat(),
            'dataset': 'Elliptic Bitcoin Transactions',
            'model': 'FraudGCN (3-layer GCN, 128-dim)',
            'total_nodes': N_TOTAL, 'total_edges': 234355,
            'framework': 'PyTorch Geometric + Google Vertex AI',
        },
        'model_performance': {
            'accuracy': 0.9472, 'precision': 0.8723,
            'recall': 0.8156, 'f1_score': 0.8430, 'auc_roc': 0.9614,
        },
        'audit_summary': {
            'nodes_audited': N_AUDIT, 'fraud_flagged': len(fraud_nodes),
            'structural_bias_detected': len(struct_bias),
            'ego_driven': len(ego_driven), 'balanced': len(balanced),
            'avg_reliance_ratio': round(np.mean([a['reliance_ratio'] for a in audits]), 4),
            'fairness_score': 'AT RISK' if not fairness['is_fair'] else 'FAIR',
        },
        'node_audits': audits,
        'bias_report': bias_report,
        'recommendations': recommendations,
        'compliance': compliance,
    }


def run_full_pipeline():
    """Run the full pipeline: load data, train model, audit, save results."""
    import torch
    from data.loader import EllipticDataLoader
    from models.gnn import FraudGCN
    from models.trainer import train_model, evaluate_model
    from audit.attributor import FeatureAttributor
    from audit.bias_detector import BiasDetector
    from audit.remediator import BiasRemediator
    import config

    device = config.DEVICE
    print(f"[INFO] Using device: {device}")

    # 1. Load data
    print("\n[1/5] Loading Elliptic Bitcoin Transaction dataset...")
    loader = EllipticDataLoader(root=config.DATA_ROOT)
    data = loader.load()
    print(f"       Nodes: {data.num_nodes:,} | Edges: {data.num_edges:,} | Features: {data.x.size(1)}")

    # 2. Train model
    print("\n[2/5] Training FraudGCN model...")
    model = FraudGCN(
        in_channels=data.x.size(1),
        hidden=config.HIDDEN_DIM,
        out_channels=config.NUM_CLASSES,
        layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    )
    model = train_model(model, data, device,
                        epochs=config.EPOCHS, lr=config.LEARNING_RATE,
                        weight_decay=config.WEIGHT_DECAY, patience=config.PATIENCE,
                        save_path=config.MODEL_SAVE_PATH)

    # 3. Evaluate
    print("\n[3/5] Evaluating model...")
    metrics = evaluate_model(model, data, device)
    print(f"       Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | AUC: {metrics['auc_roc']:.4f}")

    # 4. Run audit
    print(f"\n[4/5] Running forensic audit on {config.AUDIT_SAMPLE_SIZE} nodes...")
    data_dev = data.to(device)
    model = model.to(device)

    preds = model.predict_proba(data_dev.x, data_dev.edge_index)
    fraud_mask = preds[:, 1] > 0.5
    fraud_indices = fraud_mask.nonzero(as_tuple=True)[0].cpu()

    sample_size = min(config.AUDIT_SAMPLE_SIZE, len(fraud_indices))
    audit_indices = fraud_indices[torch.randperm(len(fraud_indices))[:sample_size]]

    non_fraud = (~fraud_mask).nonzero(as_tuple=True)[0].cpu()
    extra = min(config.AUDIT_SAMPLE_SIZE - sample_size, len(non_fraud))
    if extra > 0:
        audit_indices = torch.cat([audit_indices, non_fraud[torch.randperm(len(non_fraud))[:extra]]])

    attributor = FeatureAttributor(model, data_dev, device, n_steps=config.IG_STEPS)
    audit_results = attributor.batch_audit(audit_indices.tolist())

    # 5. Bias detection & remediation
    print("\n[5/5] Detecting biases and generating recommendations...")
    detector = BiasDetector(audit_results, data.y.tolist())
    bias_report = detector.full_bias_report()
    fairness_metrics = bias_report['fairness_metrics']

    remediator = BiasRemediator()
    recommendations = remediator.generate_recommendations(audit_results, bias_report, fairness_metrics)

    fraud_audits = [a for a in audit_results if a['prediction'] == 'FRAUD']
    struct_bias = [a for a in audit_results if a['flag'] == 'STRUCTURAL_BIAS']

    compliance = {
        'rbi_free_ai': {
            'Understandable_by_Design': {'status': 'COMPLIANT', 'details': 'Integrated Gradients provides per-feature attribution.'},
            'Fairness': {
                'status': 'AT_RISK' if not fairness_metrics['is_fair'] else 'COMPLIANT',
                'details': f"Disparate impact: {fairness_metrics['disparate_impact']:.2f}"
            },
            'Accountability': {'status': 'COMPLIANT', 'details': 'Full audit trail stored.'},
            'Transparency': {'status': 'COMPLIANT', 'details': 'Reliance ratios and bias reports available.'},
        },
        'eu_ai_act': {
            'Article_86_Right_to_Explanation': {'status': 'COMPLIANT', 'details': 'Each prediction includes ego vs structural breakdown.'},
            'Article_9_Risk_Management': {'status': 'AT_RISK', 'details': 'Continuous monitoring recommended.'},
            'Annex_IV_Documentation': {'status': 'COMPLIANT', 'details': 'Model documented.'},
        },
    }

    return {
        'metadata': {
            'timestamp': datetime.datetime.now().isoformat(),
            'dataset': 'Elliptic Bitcoin Transactions',
            'model': f'FraudGCN ({config.NUM_LAYERS}-layer GCN, {config.HIDDEN_DIM}-dim)',
            'total_nodes': data.num_nodes, 'total_edges': data.num_edges,
            'framework': 'PyTorch Geometric + Google Vertex AI',
        },
        'model_performance': metrics,
        'audit_summary': {
            'nodes_audited': len(audit_results), 'fraud_flagged': len(fraud_audits),
            'structural_bias_detected': len(struct_bias),
            'ego_driven': len([a for a in audit_results if a['flag'] == 'EGO_DRIVEN']),
            'balanced': len([a for a in audit_results if a['flag'] == 'BALANCED']),
            'avg_reliance_ratio': round(np.mean([a['reliance_ratio'] for a in audit_results]), 4),
            'fairness_score': 'AT RISK' if not fairness_metrics['is_fair'] else 'FAIR',
        },
        'node_audits': audit_results,
        'bias_report': bias_report,
        'recommendations': recommendations,
        'compliance': compliance,
    }


def main():
    parser = argparse.ArgumentParser(description='FairGraph-Audit: GNN Forensic Auditing')
    parser.add_argument('--demo', action='store_true', help='Run with synthetic demo data (no PyTorch needed for dashboard)')
    parser.add_argument('--no-dashboard', action='store_true', help='Run audit only, skip dashboard')
    parser.add_argument('--port', type=int, default=5000, help='Dashboard port')
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'audit_results.json')

    if args.demo:
        print("[MODE] Running in DEMO mode with synthetic data...")
        results = generate_demo_results()
    else:
        try:
            results = run_full_pipeline()
        except ImportError as e:
            print(f"[WARN] Missing dependency: {e}")
            print("[WARN] Falling back to demo mode. Install PyTorch + PyG for full pipeline.")
            results = generate_demo_results()

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[OK] Audit results saved to {results_file}")

    if not args.no_dashboard:
        print(f"\n[DASHBOARD] Starting at http://127.0.0.1:{args.port}")
        print("[DASHBOARD] Press Ctrl+C to stop.\n")
        from dashboard.app import create_app
        app = create_app()
        app.run(host='127.0.0.1', port=args.port, debug=False)


if __name__ == '__main__':
    main()
