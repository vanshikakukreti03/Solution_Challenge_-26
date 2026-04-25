"""
Feature Attribution Engine using Integrated Gradients.
Separates Ego-feature vs Structural-feature reliance for each node.
"""
import torch
import numpy as np
from tqdm import tqdm

EGO_FEATURE_COUNT = 94


class FeatureAttributor:
    def __init__(self, model, data, device, n_steps=50):
        self.model = model
        self.data = data
        self.device = device
        self.n_steps = n_steps

    def integrated_gradients(self, node_idx, target_class=None):
        """Compute Integrated Gradients for a single node."""
        self.model.eval()
        x = self.data.x.to(self.device)
        edge_index = self.data.edge_index.to(self.device)
        baseline = torch.zeros_like(x)

        # Determine target class
        if target_class is None:
            with torch.no_grad():
                out = self.model(x, edge_index)
                target_class = out[node_idx].argmax().item()

        # Accumulate gradients along interpolation path
        total_grads = torch.zeros(x.size(1), device=self.device)

        for step in range(1, self.n_steps + 1):
            alpha = step / self.n_steps
            interp = baseline + alpha * (x - baseline)
            interp = interp.detach().requires_grad_(True)

            out = self.model(interp, edge_index)
            score = out[node_idx, target_class]

            self.model.zero_grad()
            score.backward()

            total_grads += interp.grad[node_idx]

        # Scale: avg gradient * (input - baseline)
        avg_grads = total_grads / self.n_steps
        attributions = avg_grads * (x[node_idx] - baseline[node_idx])
        return attributions.detach().cpu()

    def compute_reliance_ratio(self, node_idx, target_class=None):
        """Compute the Ego vs Structural reliance ratio for a node."""
        attrs = self.integrated_gradients(node_idx, target_class)

        ego_score = attrs[:EGO_FEATURE_COUNT].abs().sum().item()
        struct_score = attrs[EGO_FEATURE_COUNT:].abs().sum().item()
        total = ego_score + struct_score

        if total == 0:
            return 0.5, ego_score, struct_score, attrs

        ratio = struct_score / total
        return ratio, ego_score, struct_score, attrs

    def neighborhood_influence(self, node_idx):
        """Measure prediction change when neighborhood edges are removed."""
        self.model.eval()
        x = self.data.x.to(self.device)
        edge_index = self.data.edge_index.to(self.device)

        with torch.no_grad():
            full_pred = torch.softmax(self.model(x, edge_index), dim=1)[node_idx]

            # Remove edges connected to this node
            mask = (edge_index[0] != node_idx) & (edge_index[1] != node_idx)
            isolated_edges = edge_index[:, mask]
            iso_pred = torch.softmax(self.model(x, isolated_edges), dim=1)[node_idx]

        influence = (full_pred - iso_pred).abs().sum().item()
        return influence, full_pred.cpu().numpy(), iso_pred.cpu().numpy()

    def batch_audit(self, node_indices, progress=True):
        """Run full attribution audit on a batch of nodes."""
        results = []
        iterator = tqdm(node_indices, desc="Auditing nodes") if progress else node_indices

        for idx in iterator:
            idx = int(idx)
            ratio, ego, struct, attrs = self.compute_reliance_ratio(idx)
            neigh_inf, full_p, iso_p = self.neighborhood_influence(idx)

            # Combined reliance: blend feature-level and perturbation-level
            combined_ratio = 0.6 * ratio + 0.4 * min(neigh_inf, 1.0)

            # Top features
            abs_attrs = attrs.abs()
            top_ego_idx = abs_attrs[:EGO_FEATURE_COUNT].topk(min(5, EGO_FEATURE_COUNT)).indices.tolist()
            top_struct_idx = abs_attrs[EGO_FEATURE_COUNT:].topk(min(5, len(attrs) - EGO_FEATURE_COUNT)).indices.tolist()

            pred_class = int(full_p.argmax())
            confidence = float(full_p.max())

            if combined_ratio > 0.65:
                flag = "STRUCTURAL_BIAS"
            elif combined_ratio < 0.35:
                flag = "EGO_DRIVEN"
            else:
                flag = "BALANCED"

            results.append({
                'node_id': idx,
                'prediction': "FRAUD" if pred_class == 1 else "LEGITIMATE",
                'confidence': round(confidence, 4),
                'true_label': int(self.data.y[idx].item()) if self.data.y[idx] >= 0 else "UNKNOWN",
                'reliance_ratio': round(combined_ratio, 4),
                'ego_score': round(ego, 4),
                'structural_score': round(struct, 4),
                'neighborhood_influence': round(neigh_inf, 4),
                'flag': flag,
                'top_ego_features': [{'idx': i, 'name': f'local_feat_{i}', 'value': round(attrs[i].item(), 4)} for i in top_ego_idx],
                'top_structural_features': [{'idx': i + EGO_FEATURE_COUNT, 'name': f'agg_feat_{i}', 'value': round(attrs[i + EGO_FEATURE_COUNT].item(), 4)} for i in top_struct_idx],
            })
        return results
