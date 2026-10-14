"""Send each request in examples/ to the API (in-process) and save the real response.

    python -m scripts.make_examples

Run after training a model, so examples/*.response.json match models/<version>.
request_id is random per call and is replaced by a placeholder.
"""
import json

from fastapi.testclient import TestClient

from api.main import create_app
from src.config import load_config, p

EXAMPLES = ["valid", "missing_vitals", "invalid"]


def main() -> None:
    cfg = load_config()
    client = TestClient(create_app(str(p(cfg["model"]["output_dir"]) / cfg["model"]["version"])))
    for name in EXAMPLES:
        body = json.loads(p(f"examples/{name}.json").read_text())
        r = client.post("/predict", json=body)
        out = r.json()
        out["request_id"] = "<random per request>"
        record = {"status_code": r.status_code, "body": out}
        p(f"examples/{name}.response.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"{name}: {r.status_code}")


if __name__ == "__main__":
    main()
