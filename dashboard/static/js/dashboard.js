/* ═══════════════════════════════════════
   FairGraph-Audit Dashboard — Frontend
   ═══════════════════════════════════════ */

let DATA = {};

document.addEventListener('DOMContentLoaded', () => {
    const raw = document.getElementById('app-data');
    if (raw) {
        DATA = JSON.parse(raw.textContent);
    }
    renderSummary();
    renderModelMetrics();
    renderRelianceChart();
    renderEgoVsStructChart();
    renderBiasAlerts();
    renderNodeTable();
    renderRecommendations();
    renderCompliance();
    setupNavigation();
});

/* ── Summary Cards ── */
function renderSummary() {
    const s = DATA.audit_summary || {};
    setText('total-nodes', fmtNum(DATA.metadata?.total_nodes || 0));
    setText('nodes-audited', fmtNum(s.nodes_audited || 0));
    setText('fraud-flagged', fmtNum(s.fraud_flagged || 0));
    setText('bias-alerts', fmtNum(s.structural_bias_detected || 0));
    setText('fairness-score', s.fairness_score || 'N/A');
}

/* ── Model Metrics Bar ── */
function renderModelMetrics() {
    const m = DATA.model_performance || {};
    setText('m-accuracy', (m.accuracy * 100).toFixed(1) + '%');
    setText('m-precision', (m.precision * 100).toFixed(1) + '%');
    setText('m-recall', (m.recall * 100).toFixed(1) + '%');
    setText('m-f1', (m.f1_score * 100).toFixed(1) + '%');
    setText('m-auc', (m.auc_roc * 100).toFixed(1) + '%');
}

/* ── Reliance Distribution Chart ── */
function renderRelianceChart() {
    const audits = DATA.node_audits || [];
    const ratios = audits.map(a => a.reliance_ratio);

    Plotly.newPlot('reliance-dist-chart', [{
        x: ratios,
        type: 'histogram',
        nbinsx: 30,
        marker: {
            color: ratios.map(r => r > 0.65 ? '#ef4444' : r < 0.35 ? '#10b981' : '#3b82f6'),
            line: { color: 'rgba(255,255,255,0.1)', width: 1 }
        },
        hovertemplate: 'Ratio: %{x:.2f}<br>Count: %{y}<extra></extra>'
    }], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#94a3b8', family: 'Inter' },
        xaxis: {
            title: 'Structural Reliance Ratio',
            gridcolor: 'rgba(255,255,255,0.04)',
            zerolinecolor: 'rgba(255,255,255,0.06)'
        },
        yaxis: {
            title: 'Node Count',
            gridcolor: 'rgba(255,255,255,0.04)',
            zerolinecolor: 'rgba(255,255,255,0.06)'
        },
        margin: { t: 10, r: 20, b: 50, l: 50 },
        shapes: [{
            type: 'line', x0: 0.65, x1: 0.65, y0: 0, y1: 1,
            yref: 'paper', line: { color: '#ef4444', width: 2, dash: 'dash' }
        }, {
            type: 'line', x0: 0.35, x1: 0.35, y0: 0, y1: 1,
            yref: 'paper', line: { color: '#10b981', width: 2, dash: 'dash' }
        }],
        annotations: [{
            x: 0.65, y: 1, yref: 'paper', text: 'Structural Bias →',
            showarrow: false, font: { color: '#ef4444', size: 10 }, xanchor: 'left'
        }, {
            x: 0.35, y: 1, yref: 'paper', text: '← Ego Driven',
            showarrow: false, font: { color: '#10b981', size: 10 }, xanchor: 'right'
        }]
    }, { responsive: true, displayModeBar: false });
}

/* ── Ego vs Structural Scatter ── */
function renderEgoVsStructChart() {
    const audits = DATA.node_audits || [];

    Plotly.newPlot('ego-struct-chart', [{
        x: audits.map(a => a.ego_score),
        y: audits.map(a => a.structural_score),
        mode: 'markers',
        marker: {
            size: 8, opacity: 0.7,
            color: audits.map(a => a.reliance_ratio),
            colorscale: [[0, '#10b981'], [0.5, '#3b82f6'], [1, '#ef4444']],
            colorbar: { title: 'Reliance', tickformat: '.0%', outlinewidth: 0 },
            line: { color: 'rgba(255,255,255,0.1)', width: 1 }
        },
        text: audits.map(a => `Node ${a.node_id}<br>Flag: ${a.flag}`),
        hovertemplate: '%{text}<br>Ego: %{x:.3f}<br>Struct: %{y:.3f}<extra></extra>'
    }], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#94a3b8', family: 'Inter' },
        xaxis: { title: 'Ego Attribution Score', gridcolor: 'rgba(255,255,255,0.04)' },
        yaxis: { title: 'Structural Attribution Score', gridcolor: 'rgba(255,255,255,0.04)' },
        margin: { t: 10, r: 20, b: 50, l: 60 }
    }, { responsive: true, displayModeBar: false });
}

/* ── Bias Alerts ── */
function renderBiasAlerts() {
    const findings = DATA.bias_report?.findings || [];
    const container = document.getElementById('bias-alerts');
    if (!container) return;

    if (findings.length === 0) {
        container.innerHTML = '<div class="alert-card"><div class="alert-severity low"></div><div class="alert-content"><h4>No Critical Biases Detected</h4><p>The model passes basic fairness thresholds.</p></div></div>';
        return;
    }

    container.innerHTML = findings.map(f => `
        <div class="alert-card">
            <div class="alert-severity ${f.severity.toLowerCase()}"></div>
            <div class="alert-content">
                <h4>${f.type.replace(/_/g, ' ')}</h4>
                <p>${f.description}</p>
                <span class="alert-badge badge-${f.severity.toLowerCase()}">${f.severity} · ${f.affected_nodes} nodes</span>
            </div>
        </div>
    `).join('');
}

/* ── Node Explorer Table ── */
function renderNodeTable() {
    const audits = DATA.node_audits || [];
    const tbody = document.getElementById('node-tbody');
    if (!tbody) return;

    tbody.innerHTML = audits.slice(0, 100).map(a => {
        const flagClass = a.flag === 'STRUCTURAL_BIAS' ? 'flag-structural' : a.flag === 'EGO_DRIVEN' ? 'flag-ego' : 'flag-balanced';
        const gaugeClass = a.reliance_ratio > 0.65 ? 'danger' : a.reliance_ratio < 0.35 ? 'success' : 'neutral';
        return `
        <tr onclick="showNodeDetail(${a.node_id})" style="cursor:pointer">
            <td><strong>${a.node_id}</strong></td>
            <td>${a.prediction}</td>
            <td>${(a.confidence * 100).toFixed(1)}%</td>
            <td>
                <div style="display:flex;align-items:center;gap:8px">
                    <div class="gauge-bar" style="width:80px"><div class="gauge-fill ${gaugeClass}" style="width:${a.reliance_ratio * 100}%"></div></div>
                    <span>${(a.reliance_ratio * 100).toFixed(0)}%</span>
                </div>
            </td>
            <td><span class="flag-tag ${flagClass}">${a.flag.replace('_', ' ')}</span></td>
        </tr>`;
    }).join('');
}

/* ── Node Detail Modal ── */
function showNodeDetail(nodeId) {
    const node = (DATA.node_audits || []).find(a => a.node_id === nodeId);
    if (!node) return;

    const modal = document.getElementById('modal-overlay');
    const body = document.getElementById('modal-body');

    body.innerHTML = `
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px">
            <div><span style="color:var(--text-muted);font-size:0.8rem">PREDICTION</span><br><strong style="font-size:1.2rem">${node.prediction}</strong></div>
            <div><span style="color:var(--text-muted);font-size:0.8rem">CONFIDENCE</span><br><strong style="font-size:1.2rem">${(node.confidence * 100).toFixed(1)}%</strong></div>
            <div><span style="color:var(--text-muted);font-size:0.8rem">RELIANCE RATIO</span><br><strong style="font-size:1.2rem;color:${node.reliance_ratio > 0.65 ? 'var(--danger)' : 'var(--success)'}">${(node.reliance_ratio * 100).toFixed(1)}%</strong></div>
            <div><span style="color:var(--text-muted);font-size:0.8rem">FLAG</span><br><strong style="font-size:1.2rem">${node.flag}</strong></div>
        </div>
        <h4 style="margin-bottom:8px;color:var(--accent-cyan)">Top Ego Features</h4>
        <div style="margin-bottom:16px">${(node.top_ego_features || []).map(f => `<div style="display:flex;justify-content:space-between;padding:4px 0;font-size:0.85rem"><span>${f.name}</span><span style="color:var(--success)">${f.value.toFixed(4)}</span></div>`).join('')}</div>
        <h4 style="margin-bottom:8px;color:var(--accent-purple)">Top Structural Features</h4>
        <div>${(node.top_structural_features || []).map(f => `<div style="display:flex;justify-content:space-between;padding:4px 0;font-size:0.85rem"><span>${f.name}</span><span style="color:var(--danger)">${f.value.toFixed(4)}</span></div>`).join('')}</div>
    `;

    document.getElementById('modal-title').textContent = `Transaction #${nodeId}`;
    modal.classList.add('active');
}

function closeModal() {
    document.getElementById('modal-overlay').classList.remove('active');
}

/* ── Recommendations ── */
function renderRecommendations() {
    const recs = DATA.recommendations || [];
    const container = document.getElementById('rec-container');
    if (!container) return;

    container.innerHTML = recs.map(r => `
        <div class="rec-card">
            <div class="rec-header">
                <h4>${r.id}: ${r.title}</h4>
                <span class="alert-badge badge-${r.severity === 'CRITICAL' ? 'critical' : r.severity === 'HIGH' ? 'high' : r.severity === 'MEDIUM' ? 'medium' : 'info'}">${r.severity}</span>
            </div>
            <div class="rec-source"><strong>Source:</strong> ${r.source}</div>
            <p class="rec-reason"><strong>Why:</strong> ${r.reason}</p>
            <ul class="rec-actions">${r.actions.map(a => `<li>${a}</li>`).join('')}</ul>
            <div class="rec-compliance">📋 ${r.compliance}</div>
        </div>
    `).join('');
}

/* ── Compliance ── */
function renderCompliance() {
    const c = DATA.compliance || {};
    const container = document.getElementById('compliance-content');
    if (!container) return;

    const rbi = c.rbi_free_ai || {};
    const eu = c.eu_ai_act || {};

    container.innerHTML = `
        <div class="compliance-grid">
            <div class="compliance-card">
                <h3>🇮🇳 RBI FREE-AI Seven Sutras</h3>
                ${Object.entries(rbi).map(([k, v]) => `
                    <div class="compliance-item">
                        <span class="label">${k.replace(/_/g, ' ')}</span>
                        <span class="status-badge status-${v.status.toLowerCase().replace(' ', '-')}">${v.status}</span>
                    </div>
                `).join('')}
            </div>
            <div class="compliance-card">
                <h3>🇪🇺 EU AI Act</h3>
                ${Object.entries(eu).map(([k, v]) => `
                    <div class="compliance-item">
                        <span class="label">${k.replace(/_/g, ' ')}</span>
                        <span class="status-badge status-${v.status.toLowerCase().replace(' ', '-')}">${v.status}</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

/* ── Navigation ── */
function setupNavigation() {
    document.querySelectorAll('.nav-links a').forEach(link => {
        link.addEventListener('click', e => {
            document.querySelectorAll('.nav-links a').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });
}

/* ── Utilities ── */
function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function fmtNum(n) {
    return Number(n).toLocaleString();
}
