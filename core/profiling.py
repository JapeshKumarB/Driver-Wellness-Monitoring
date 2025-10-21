import json
import os

class ThresholdManager:
    def __init__(self, path: str):
        self.path = path
        self.data = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}
        else:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                json.dump({}, f)
            self.data = {}

    def get_for_driver(self, identity: str | None):
        if not identity or identity not in self.data:
            return {}
        return self.data.get(identity, {})

    def adapt(self, identity: str | None, metrics: dict):
        if not identity:
            return
        entry = self.data.get(identity, {})
        # Simple adaptive logic: track typical EAR
        ear = metrics.get("ear_avg")
        if ear:
            entry["ear_baseline"] = 0.9 * entry.get("ear_baseline", ear) + 0.1 * ear
        self.data[identity] = entry
        try:
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass