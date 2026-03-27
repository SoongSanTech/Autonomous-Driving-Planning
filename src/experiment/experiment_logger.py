"""
ExperimentLogger: 실험 결과를 JSON + SQLite로 구조화 저장.

SQLite WAL 모드 적용으로 야간 그리드 서치 중 CLI 조회 동시 접근 안전.
JSON fallback으로 SQLite 실패 시에도 데이터 보존.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# SQL schema
_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id   TEXT PRIMARY KEY,
    experiment_type TEXT NOT NULL,
    purpose         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'running',
    parent_id       TEXT,
    FOREIGN KEY (parent_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS experiment_configs (
    experiment_id   TEXT NOT NULL,
    config_key      TEXT NOT NULL,
    config_value    TEXT NOT NULL,
    PRIMARY KEY (experiment_id, config_key),
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS experiment_metrics (
    experiment_id   TEXT NOT NULL,
    metric_key      TEXT NOT NULL,
    metric_value    REAL NOT NULL,
    recorded_at     TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS experiment_analysis (
    experiment_id   TEXT NOT NULL,
    analysis_text   TEXT NOT NULL,
    recommendations TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS experiment_cli_commands (
    experiment_id   TEXT NOT NULL,
    command         TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS experiment_artifacts (
    experiment_id   TEXT NOT NULL,
    artifact_type   TEXT NOT NULL,
    artifact_path   TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
"""


def _numpy_serializer(obj):
    """JSON serializer that handles numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class ExperimentLogger:
    """실험 결과를 JSON + SQLite로 구조화 저장."""

    def __init__(
        self,
        db_path: str = "experiments/experiment_log.db",
        json_dir: str = "experiments/logs/",
    ):
        self.db_path = Path(db_path)
        self.json_dir = Path(json_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)

        self._init_db()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        experiment_type: str,
        purpose: str,
        config: dict,
        parent_id: Optional[str] = None,
    ) -> str:
        """새 실험 생성, experiment_id(UUID) 반환."""
        experiment_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()

        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO experiments (experiment_id, experiment_type, purpose, created_at, status, parent_id) "
                "VALUES (?, ?, ?, ?, 'running', ?)",
                (experiment_id, experiment_type, purpose, created_at, parent_id),
            )
            for key, value in config.items():
                conn.execute(
                    "INSERT INTO experiment_configs (experiment_id, config_key, config_value) "
                    "VALUES (?, ?, ?)",
                    (experiment_id, key, json.dumps(value, default=_numpy_serializer)),
                )
            conn.commit()
        finally:
            conn.close()

        # JSON fallback
        self._save_json(experiment_id, {
            "experiment_id": experiment_id,
            "experiment_type": experiment_type,
            "purpose": purpose,
            "created_at": created_at,
            "status": "running",
            "parent_id": parent_id,
            "config": config,
            "metrics": {},
            "analysis": None,
            "cli_commands": [],
        })

        logger.info("Created experiment %s (%s)", experiment_id, experiment_type)
        return experiment_id

    def log_metrics(self, experiment_id: str, metrics: dict) -> None:
        """메트릭 기록 (mae_steering, mae_throttle, reward 등)."""
        recorded_at = datetime.now().isoformat()

        conn = self._get_conn()
        try:
            for key, value in metrics.items():
                numeric_value = float(value) if not isinstance(value, (int, float)) else value
                conn.execute(
                    "INSERT INTO experiment_metrics (experiment_id, metric_key, metric_value, recorded_at) "
                    "VALUES (?, ?, ?, ?)",
                    (experiment_id, key, float(numeric_value), recorded_at),
                )
            conn.commit()
        finally:
            conn.close()

        # Update JSON
        self._update_json_metrics(experiment_id, metrics)

    def log_analysis(
        self, experiment_id: str, analysis: str, recommendations: list[str]
    ) -> None:
        """분석 결과 및 보정 권고 기록."""
        created_at = datetime.now().isoformat()

        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO experiment_analysis (experiment_id, analysis_text, recommendations, created_at) "
                "VALUES (?, ?, ?, ?)",
                (experiment_id, analysis, json.dumps(recommendations, default=_numpy_serializer), created_at),
            )
            conn.commit()
        finally:
            conn.close()

        self._update_json_field(experiment_id, "analysis", {
            "text": analysis,
            "recommendations": recommendations,
        })

    def log_cli_command(self, experiment_id: str, command: str) -> None:
        """재현용 CLI 명령어 기록."""
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO experiment_cli_commands (experiment_id, command) VALUES (?, ?)",
                (experiment_id, command),
            )
            conn.commit()
        finally:
            conn.close()

        self._append_json_list(experiment_id, "cli_commands", command)

    def get_experiment(self, experiment_id: str) -> dict:
        """단일 실험 조회."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Experiment not found: {experiment_id}")

            result = dict(row)

            # Config
            config_rows = conn.execute(
                "SELECT config_key, config_value FROM experiment_configs WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchall()
            result["config"] = {r["config_key"]: json.loads(r["config_value"]) for r in config_rows}

            # Metrics
            metric_rows = conn.execute(
                "SELECT metric_key, metric_value, recorded_at FROM experiment_metrics "
                "WHERE experiment_id = ? ORDER BY recorded_at",
                (experiment_id,),
            ).fetchall()
            result["metrics"] = {r["metric_key"]: r["metric_value"] for r in metric_rows}

            # Analysis
            analysis_rows = conn.execute(
                "SELECT analysis_text, recommendations, created_at FROM experiment_analysis "
                "WHERE experiment_id = ? ORDER BY created_at DESC LIMIT 1",
                (experiment_id,),
            ).fetchall()
            if analysis_rows:
                result["analysis"] = {
                    "text": analysis_rows[0]["analysis_text"],
                    "recommendations": json.loads(analysis_rows[0]["recommendations"]),
                }
            else:
                result["analysis"] = None

            # CLI commands
            cmd_rows = conn.execute(
                "SELECT command FROM experiment_cli_commands WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchall()
            result["cli_commands"] = [r["command"] for r in cmd_rows]

            return result
        finally:
            conn.close()

    def list_experiments(self, experiment_type: Optional[str] = None) -> list[dict]:
        """실험 목록 조회 (타입별 필터링)."""
        conn = self._get_conn()
        try:
            if experiment_type:
                rows = conn.execute(
                    "SELECT * FROM experiments WHERE experiment_type = ? ORDER BY created_at",
                    (experiment_type,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM experiments ORDER BY created_at"
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def compare_experiments(self, experiment_ids: list[str]) -> dict:
        """여러 실험의 메트릭을 교차 비교."""
        if len(experiment_ids) < 2:
            raise ValueError("At least 2 experiment IDs required for comparison")

        experiments = {}
        for eid in experiment_ids:
            experiments[eid] = self.get_experiment(eid)

        # Collect all metric keys
        all_keys: set[str] = set()
        for exp in experiments.values():
            all_keys.update(exp.get("metrics", {}).keys())

        comparisons = {}
        ids = list(experiment_ids)
        for key in sorted(all_keys):
            values = {}
            for eid in ids:
                values[eid] = experiments[eid].get("metrics", {}).get(key)

            # Compute delta between first two experiments
            v0 = values.get(ids[0])
            v1 = values.get(ids[1])
            delta = None
            improved = None
            if v0 is not None and v1 is not None:
                delta = v1 - v0
                # For loss/error metrics (lower is better): negative delta = improved
                # For reward metrics (higher is better): positive delta = improved
                # Use heuristic: if key contains 'loss', 'mae', 'error', 'collision', 'drop'
                # then lower is better
                lower_is_better = any(
                    kw in key.lower()
                    for kw in ("loss", "mae", "error", "collision", "drop", "anomal")
                )
                improved = delta < 0 if lower_is_better else delta > 0

            comparisons[key] = {
                "values": values,
                "delta": delta,
                "improved": improved,
            }

        return {
            "experiment_ids": ids,
            "comparisons": comparisons,
        }

    def generate_report(self, experiment_ids: Optional[list[str]] = None) -> str:
        """종합 보고서 생성 (Markdown)."""
        if experiment_ids:
            experiments = [self.get_experiment(eid) for eid in experiment_ids]
        else:
            experiments = []
            for exp_summary in self.list_experiments():
                experiments.append(self.get_experiment(exp_summary["experiment_id"]))

        # Sort by created_at ascending
        experiments.sort(key=lambda e: e.get("created_at", ""))

        lines = ["# 실험 종합 보고서", ""]
        lines.append(f"생성 시각: {datetime.now().isoformat()}")
        lines.append(f"총 실험 수: {len(experiments)}")
        lines.append("")

        for i, exp in enumerate(experiments, 1):
            lines.append(f"## {i}. {exp.get('experiment_type', 'unknown')} — {exp.get('purpose', '')}")
            lines.append(f"- ID: {exp['experiment_id']}")
            lines.append(f"- 생성: {exp.get('created_at', '')}")
            lines.append(f"- 상태: {exp.get('status', '')}")

            config = exp.get("config", {})
            if config:
                lines.append("- 설정:")
                for k, v in config.items():
                    lines.append(f"  - {k}: {v}")

            metrics = exp.get("metrics", {})
            if metrics:
                lines.append("- 메트릭:")
                for k, v in metrics.items():
                    lines.append(f"  - {k}: {v}")

            analysis = exp.get("analysis")
            if analysis:
                lines.append(f"- 분석: {analysis.get('text', '')}")
                recs = analysis.get("recommendations", [])
                if recs:
                    lines.append("- 권고:")
                    for r in recs:
                        lines.append(f"  - {r}")

            cli_cmds = exp.get("cli_commands", [])
            if cli_cmds:
                lines.append("- CLI 재현:")
                for cmd in cli_cmds:
                    lines.append(f"  ```\n  {cmd}\n  ```")

            lines.append("")

        return "\n".join(lines)

    def update_status(self, experiment_id: str, status: str) -> None:
        """실험 상태 업데이트."""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE experiments SET status = ? WHERE experiment_id = ?",
                (status, experiment_id),
            )
            conn.commit()
        finally:
            conn.close()

        self._update_json_field(experiment_id, "status", status)

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    def _json_path(self, experiment_id: str) -> Path:
        return self.json_dir / f"{experiment_id}.json"

    def _save_json(self, experiment_id: str, data: dict) -> None:
        try:
            path = self._json_path(experiment_id)
            path.write_text(
                json.dumps(data, indent=2, default=_numpy_serializer, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("JSON save failed for %s: %s", experiment_id, e)

    def _load_json(self, experiment_id: str) -> Optional[dict]:
        path = self._json_path(experiment_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _update_json_metrics(self, experiment_id: str, metrics: dict) -> None:
        data = self._load_json(experiment_id)
        if data is None:
            return
        existing = data.get("metrics", {})
        existing.update(metrics)
        data["metrics"] = existing
        self._save_json(experiment_id, data)

    def _update_json_field(self, experiment_id: str, field: str, value) -> None:
        data = self._load_json(experiment_id)
        if data is None:
            return
        data[field] = value
        self._save_json(experiment_id, data)

    def _append_json_list(self, experiment_id: str, field: str, value) -> None:
        data = self._load_json(experiment_id)
        if data is None:
            return
        lst = data.get(field, [])
        lst.append(value)
        data[field] = lst
        self._save_json(experiment_id, data)
