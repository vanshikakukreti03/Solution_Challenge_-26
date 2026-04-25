"""
Bias Remediation Engine.
Generates actionable recommendations to fix detected biases.
"""
import numpy as np


class BiasRemediator:
    def generate_recommendations(self, audit_results, bias_report, fairness_metrics):
        recs = []
        avg_ratio = np.mean([r['reliance_ratio'] for r in audit_results])

        # 1. Structural over-reliance
        if avg_ratio > 0.5:
            recs.append({
                'id': 'REC-001',
                'severity': 'HIGH',
                'category': 'Feature Engineering',
                'title': 'Reduce Structural Feature Dominance',
                'source': f'Average structural reliance is {avg_ratio:.0%}, exceeding the 50% threshold.',
                'reason': 'The GNN aggregates too much influence from neighborhood features relative to the node\'s own transactional behavior.',
                'actions': [
                    'Apply L1 regularization specifically on structural feature weights',
                    'Reduce GNN depth from 3 layers to 2 to limit message-passing radius',
                    'Introduce ego-feature attention: weight ego-features 2x during aggregation',
                    'Train a parallel MLP on ego-features only and ensemble with GNN',
                ],
                'compliance': 'RBI FREE-AI Sutra: Understandable by Design',
            })

        # 2. Guilt-by-association
        gba_count = bias_report.get('total_biased_nodes', 0)
        if gba_count > 0:
            recs.append({
                'id': 'REC-002',
                'severity': 'CRITICAL',
                'category': 'Model Architecture',
                'title': 'Mitigate Guilt-by-Association Bias',
                'source': f'{gba_count} transactions flagged primarily due to their neighbors.',
                'reason': 'The model penalizes nodes for being connected to suspicious neighbors, regardless of their own legitimate behavior.',
                'actions': [
                    'Add a reliance-ratio penalty to the loss: L_total = L_ce + λ * max(0, R_struct - 0.65)',
                    'Use GAT (Graph Attention) with learnable ego-priority weights',
                    'Implement counterfactual fairness: verify prediction is stable under neighborhood randomization',
                    'Apply FairDrop: randomly drop neighbor edges during training to reduce structural dependency',
                ],
                'compliance': 'EU AI Act Article 86 — Right to Explanation',
            })

        # 3. Disparate impact
        if fairness_metrics and not fairness_metrics.get('is_fair', True):
            di = fairness_metrics.get('disparate_impact', 0)
            recs.append({
                'id': 'REC-003',
                'severity': 'HIGH',
                'category': 'Fairness Calibration',
                'title': 'Address Disparate Impact Across Structural Groups',
                'source': f'Disparate impact ratio: {di:.2f} (acceptable: 0.80–1.25).',
                'reason': 'Nodes with high structural reliance face significantly different false positive rates than ego-driven nodes.',
                'actions': [
                    'Apply group-calibrated thresholds: lower fraud threshold for high-structural nodes',
                    'Implement equalized odds constraint during training',
                    'Use reject-option classification: defer borderline high-structural cases to human review',
                    'Schedule quarterly re-audits with updated transaction data',
                ],
                'compliance': 'RBI FREE-AI Sutra: Fairness',
            })

        # 4. Feature leakage
        leakage = bias_report.get('feature_leakage', {})
        if leakage.get('risk') == 'HIGH':
            recs.append({
                'id': 'REC-004',
                'severity': 'MEDIUM',
                'category': 'Data Pipeline',
                'title': 'Investigate Structural Feature Leakage',
                'source': f'{leakage.get("outlier_count", 0)} nodes show anomalous structural attribution patterns.',
                'reason': 'Aggregated neighborhood features may inadvertently encode sensitive attributes or create proxy discrimination.',
                'actions': [
                    'Audit the 72 aggregated features for correlation with protected attributes',
                    'Remove or decorrelate features with >0.3 correlation to sensitive variables',
                    'Apply adversarial debiasing: train a discriminator to remove protected-attribute signal',
                    'Use SHAP interaction values to detect feature-feature leakage paths',
                ],
                'compliance': 'RBI FREE-AI Sutra: Fairness + EU AI Act Annex IV',
            })

        # 5. Always recommend monitoring
        recs.append({
            'id': 'REC-005',
            'severity': 'INFO',
            'category': 'Governance',
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
        })

        return recs
