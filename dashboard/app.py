"""
Flask Dashboard Application for FairGraph-Audit.
Serves the interactive forensic auditing dashboard.
"""
import os
import json
from flask import Flask, render_template, jsonify, request

RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'audit_results.json')


def create_app():
    app = Flask(__name__,
                template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                static_folder=os.path.join(os.path.dirname(__file__), 'static'))

    def load_results():
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, 'r') as f:
                return json.load(f)
        return {}

    @app.route('/')
    def index():
        data = load_results()
        return render_template('index.html', data=json.dumps(data))

    @app.route('/api/summary')
    def api_summary():
        data = load_results()
        return jsonify({
            'metadata': data.get('metadata', {}),
            'model_performance': data.get('model_performance', {}),
            'audit_summary': data.get('audit_summary', {}),
        })

    @app.route('/api/audit-results')
    def api_audit_results():
        data = load_results()
        results = data.get('node_audits', [])
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        flag_filter = request.args.get('flag', None)

        if flag_filter:
            results = [r for r in results if r['flag'] == flag_filter]

        start = (page - 1) * per_page
        return jsonify({
            'results': results[start:start + per_page],
            'total': len(results),
            'page': page,
        })

    @app.route('/api/node/<int:node_id>')
    def api_node(node_id):
        data = load_results()
        for r in data.get('node_audits', []):
            if r['node_id'] == node_id:
                return jsonify(r)
        return jsonify({'error': 'Node not found'}), 404

    @app.route('/api/bias-report')
    def api_bias_report():
        data = load_results()
        return jsonify(data.get('bias_report', {}))

    @app.route('/api/recommendations')
    def api_recommendations():
        data = load_results()
        return jsonify(data.get('recommendations', []))

    @app.route('/api/compliance')
    def api_compliance():
        data = load_results()
        return jsonify(data.get('compliance', {}))

    return app
