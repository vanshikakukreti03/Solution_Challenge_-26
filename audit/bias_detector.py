"""
Bias Detection Engine.
Identifies guilt-by-association, disparate impact, and systematic biases.
"""
import numpy as np


class BiasDetector:
    def __init__(self, audit_results, labels):
        self.results = audit_results
        self.labels = labels

    def detect_guilt_by_association(self, threshold=0.65):
        """Find nodes flagged as fraud primarily due to neighborhood."""
        gba = []
        for r in self.results:
            if r['prediction'] == 'FRAUD' and r['reliance_ratio'] > threshold:
                severity = 'CRITICAL' if r['reliance_ratio'] > 0.85 else 'HIGH' if r['reliance_ratio'] > 0.75 else 'MEDIUM'
                gba.append({
                    'node_id': r['node_id'],
                    'reliance_ratio': r['reliance_ratio'],
                    'confidence': r['confidence'],
                    'severity': severity,
                    'bias_type': 'GUILT_BY_ASSOCIATION',
                    'source': 'Structural neighborhood features dominate the fraud prediction.',
                    'reason': (
                        f"Transaction {r['node_id']} was flagged with {r['reliance_ratio']:.0%} "
                        f"structural reliance. Its own behavioral features (ego-features) "
                        f"contribute only {1 - r['reliance_ratio']:.0%} to the decision. "
                        f"The fraud flag is driven by connections to suspicious neighbors, "
                        f"not the transaction's own characteristics."
                    ),
                })
        return gba

    def compute_fairness_metrics(self):
        """Compute fairness metrics across structural-reliance groups."""
        high_struct, low_struct = [], []
        for r in self.results:
            label = r.get('true_label', -1)
            if not isinstance(label, int) or label < 0:
                continue
            pred = 1 if r['prediction'] == 'FRAUD' else 0
            entry = {'pred': pred, 'label': label}
            if r['reliance_ratio'] > 0.5:
                high_struct.append(entry)
            else:
                low_struct.append(entry)

        fpr_high = self._fpr(high_struct)
        fpr_low = self._fpr(low_struct)
        fnr_high = self._fnr(high_struct)
        fnr_low = self._fnr(low_struct)

        di = fpr_high / max(fpr_low, 1e-8)
        eo_diff = abs(fpr_high - fpr_low)
        dp_diff = abs(self._positive_rate(high_struct) - self._positive_rate(low_struct))

        is_fair = 0.8 <= di <= 1.25 and eo_diff < 0.1

        return {
            'fpr_high_structural': round(fpr_high, 4),
            'fpr_low_structural': round(fpr_low, 4),
            'fnr_high_structural': round(fnr_high, 4),
            'fnr_low_structural': round(fnr_low, 4),
            'disparate_impact': round(di, 4),
            'equalized_odds_diff': round(eo_diff, 4),
            'demographic_parity_diff': round(dp_diff, 4),
            'is_fair': is_fair,
            'high_structural_group_size': len(high_struct),
            'low_structural_group_size': len(low_struct),
        }

    def detect_feature_leakage(self):
        """Detect if structural features encode protected attributes."""
        ratios = [r['reliance_ratio'] for r in self.results]
        mu, sigma = np.mean(ratios), np.std(ratios)
        outliers = [r for r in self.results if abs(r['reliance_ratio'] - mu) > 2 * sigma]
        return {
            'mean_reliance': round(mu, 4),
            'std_reliance': round(sigma, 4),
            'outlier_count': len(outliers),
            'outlier_node_ids': [r['node_id'] for r in outliers],
            'risk': 'HIGH' if len(outliers) > len(self.results) * 0.05 else 'LOW',
        }

    def full_bias_report(self):
        """Generate comprehensive bias report."""
        gba = self.detect_guilt_by_association()
        fairness = self.compute_fairness_metrics()
        leakage = self.detect_feature_leakage()

        findings = []
        if gba:
            findings.append({
                'type': 'GUILT_BY_ASSOCIATION',
                'severity': 'CRITICAL' if any(g['severity'] == 'CRITICAL' for g in gba) else 'HIGH',
                'affected_nodes': len(gba),
                'description': f"{len(gba)} transactions flagged primarily due to neighborhood connections rather than own behavior.",
                'details': gba,
            })
        if not fairness['is_fair']:
            findings.append({
                'type': 'DISPARATE_IMPACT',
                'severity': 'HIGH',
                'affected_nodes': fairness['high_structural_group_size'],
                'description': f"Disparate impact ratio of {fairness['disparate_impact']:.2f} detected between structural-reliance groups.",
                'details': fairness,
            })
        if leakage['risk'] == 'HIGH':
            findings.append({
                'type': 'FEATURE_LEAKAGE_RISK',
                'severity': 'MEDIUM',
                'affected_nodes': leakage['outlier_count'],
                'description': f"{leakage['outlier_count']} nodes show anomalous structural reliance patterns.",
                'details': leakage,
            })
        return {
            'findings': findings,
            'fairness_metrics': fairness,
            'feature_leakage': leakage,
            'guilt_by_association': gba,
            'total_biased_nodes': len(gba),
        }

    @staticmethod
    def _fpr(entries):
        negatives = [e for e in entries if e['label'] == 0]
        if not negatives:
            return 0.0
        fp = sum(1 for e in negatives if e['pred'] == 1)
        return fp / len(negatives)

    @staticmethod
    def _fnr(entries):
        positives = [e for e in entries if e['label'] == 1]
        if not positives:
            return 0.0
        fn = sum(1 for e in positives if e['pred'] == 0)
        return fn / len(positives)

    @staticmethod
    def _positive_rate(entries):
        if not entries:
            return 0.0
        return sum(1 for e in entries if e['pred'] == 1) / len(entries)
