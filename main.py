import os
import json
from dashboard.app import create_app
from run import generate_demo_results

app = create_app()

results_file = 'results/audit_results.json'
if not os.path.exists(results_file):
    os.makedirs('results', exist_ok=True)
    results = generate_demo_results()
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
