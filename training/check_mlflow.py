import urllib.request, json
req = urllib.request.Request(
    'http://localhost:5000/ajax-api/2.0/mlflow/runs/search',
    data=json.dumps({"experiment_ids": ["1"], "max_results": 10}).encode(),
    headers={"Content-Type": "application/json"}
)
data = json.loads(urllib.request.urlopen(req).read())
runs = data.get("runs", [])
print(f"{len(runs)} runs in experiment:")
for r in runs:
    info = r["info"]
    metrics = r["data"].get("metrics", {})
    print(f"  [{info['run_id'][:8]}] {info.get('run_name','?'):30s} status={info['status']} metrics={metrics}")
