"""Minimal local prediction audit: scores and amounts only, no full feature payloads."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class HistoryService:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, model_version TEXT NOT NULL,
                prediction TEXT NOT NULL, probability REAL NOT NULL, risk_level TEXT NOT NULL,
                amount REAL, threshold REAL NOT NULL, source TEXT NOT NULL)""")
            connection.execute("CREATE INDEX IF NOT EXISTS prediction_version_date ON predictions(model_version, created_at DESC)")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def record(self, results, source):
        with self.connect() as connection:
            connection.executemany("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?)", [
                (r["id"], r["created_at"], r["model_version"], r["prediction"], r["fraud_probability"],
                 r["risk_level"], r["amount"], r["threshold"], source) for r in results])

    def snapshot(self, version):
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            total, fraud = connection.execute("SELECT COUNT(*), COALESCE(SUM(prediction = 'Fraud'),0) FROM predictions WHERE model_version=?", (version,)).fetchone()
            risk = [dict(row) for row in connection.execute("SELECT risk_level AS label, COUNT(*) AS count FROM predictions WHERE model_version=? GROUP BY risk_level", (version,))]
            recent = [dict(row) for row in connection.execute("SELECT id, created_at, amount AS Amount, probability AS fraud_probability, risk_level, prediction, threshold, source FROM predictions WHERE model_version=? ORDER BY created_at DESC, rowid DESC LIMIT 12", (version,))]
            probability = []
            for i in range(10):
                count = connection.execute("SELECT COUNT(*) FROM predictions WHERE model_version=? AND probability>=? AND (probability<? OR (?=9 AND probability=1))", (version, i / 10, (i + 1) / 10, i)).fetchone()[0]
                probability.append({"label": f"{i / 10:.1f}–{(i + 1) / 10:.1f}", "count": count})
            amount = []
            edges = [0, 10, 25, 50, 100, 250, 500, 1000, 1e13]
            for left, right in zip(edges, edges[1:]):
                count = connection.execute("SELECT COUNT(*) FROM predictions WHERE model_version=? AND amount>=? AND amount<?", (version, left, right)).fetchone()[0]
                amount.append({"label": f"{left:g}–{right:g}" if right < 1e13 else "1000+", "count": count})
        return {"scope": "live predictions for current model", "summary": {"total": total, "fraud": fraud, "legitimate": total - fraud,
                    "fraud_percentage": fraud / total * 100 if total else 0}, "risk_distribution": risk,
                "probability_distribution": probability, "amount_distribution": amount, "recent": recent, "timeline": []}
