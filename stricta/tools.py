import json
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .schemas import ActionProposal

DEFAULT_STRICTA_JSONL_PATH = "./stricta_ledger.jsonl"
DEFAULT_STRICTA_SQLITE_PATH = "./stricta_ledger.sqlite"


def _clean_arguments(arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not arguments:
        return {}
    return {k: v for k, v in arguments.items() if not str(k).startswith("_")}


def make_jsonl_ledger_executor(
    path: str = DEFAULT_STRICTA_JSONL_PATH,
    inject_rollback_on: Optional[str] = None,
) -> Callable[[ActionProposal, Dict[str, Any]], Dict[str, Any]]:
    target_path = Path(path)

    def _executor(proposal: ActionProposal, runtime_state: Dict[str, Any]) -> Dict[str, Any]:
        if inject_rollback_on and proposal.target_node_id == inject_rollback_on:
            raise RuntimeError(
                f"Injected Stricta JSONL ledger rollback on node {proposal.target_node_id}"
            )

        existing_rows = query_jsonl_ledger(path=str(target_path))
        entry_id = len(existing_rows) + 1
        cleaned_args = _clean_arguments(proposal.arguments)
        record = {
            "id": entry_id,
            "node_id": proposal.target_node_id,
            "phase": runtime_state.get("current_phase", ""),
            "tool_name": proposal.tool_name,
            "arguments": cleaned_args,
            "expected_state_change": proposal.expected_state_change,
        }

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        return {
            "ok": True,
            "backend": "jsonl",
            "path": str(target_path),
            "entry_id": entry_id,
        }

    return _executor


def _ensure_sqlite_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stricta_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            arguments_json TEXT NOT NULL,
            expected_state_change TEXT NOT NULL
        )
        """
    )
    conn.commit()


def make_sqlite_ledger_executor(
    db_path: str = DEFAULT_STRICTA_SQLITE_PATH,
    inject_rollback_on: Optional[str] = None,
) -> Callable[[ActionProposal, Dict[str, Any]], Dict[str, Any]]:
    target_db = Path(db_path)

    def _executor(proposal: ActionProposal, runtime_state: Dict[str, Any]) -> Dict[str, Any]:
        if inject_rollback_on and proposal.target_node_id == inject_rollback_on:
            raise RuntimeError(
                f"Injected Stricta SQLite ledger rollback on node {proposal.target_node_id}"
            )

        target_db.parent.mkdir(parents=True, exist_ok=True)
        cleaned_args = _clean_arguments(proposal.arguments)
        phase_val = str(runtime_state.get("current_phase", ""))

        with sqlite3.connect(str(target_db)) as conn:
            _ensure_sqlite_schema(conn)
            cur = conn.execute(
                """
                INSERT INTO stricta_ledger (
                    node_id,
                    phase,
                    tool_name,
                    arguments_json,
                    expected_state_change
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    proposal.target_node_id,
                    phase_val,
                    proposal.tool_name,
                    json.dumps(cleaned_args, ensure_ascii=False, sort_keys=True),
                    proposal.expected_state_change,
                ),
            )
            conn.commit()
            entry_id = int(cur.lastrowid)

        return {
            "ok": True,
            "backend": "sqlite",
            "db_path": str(target_db),
            "entry_id": entry_id,
        }

    return _executor


def query_jsonl_ledger(path: str = DEFAULT_STRICTA_JSONL_PATH) -> List[Dict[str, Any]]:
    target_path = Path(path)
    if not target_path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with target_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            rows.append(
                {
                    "id": int(raw.get("id", idx)),
                    "node_id": str(raw.get("node_id", "")),
                    "phase": str(raw.get("phase", "")),
                    "tool_name": str(raw.get("tool_name", "")),
                    "arguments": _clean_arguments(raw.get("arguments")),
                    "expected_state_change": str(raw.get("expected_state_change", "")),
                }
            )
    return rows


def query_sqlite_ledger(db_path: str = DEFAULT_STRICTA_SQLITE_PATH) -> List[Dict[str, Any]]:
    target_db = Path(db_path)
    if not target_db.exists():
        return []

    with sqlite3.connect(str(target_db)) as conn:
        _ensure_sqlite_schema(conn)
        cursor = conn.execute(
            """
            SELECT id, node_id, phase, tool_name, arguments_json, expected_state_change
            FROM stricta_ledger
            ORDER BY id ASC
            """
        )
        fetched = cursor.fetchall()

    rows: List[Dict[str, Any]] = []
    for row_id, node_id, phase, tool_name, arguments_json, expected_state_change in fetched:
        parsed_args = json.loads(arguments_json) if arguments_json else {}
        rows.append(
            {
                "id": int(row_id),
                "node_id": str(node_id),
                "phase": str(phase),
                "tool_name": str(tool_name),
                "arguments": _clean_arguments(parsed_args),
                "expected_state_change": str(expected_state_change),
            }
        )
    return rows


def assert_ledgers_match(
    jsonl_path: str = DEFAULT_STRICTA_JSONL_PATH,
    db_path: str = DEFAULT_STRICTA_SQLITE_PATH,
    run_if_missing: bool = False,
    compare_arguments: bool = True,
) -> Dict[str, Any]:
    jsonl_file = Path(jsonl_path)
    sqlite_file = Path(db_path)

    if not jsonl_file.exists() or not sqlite_file.exists():
        if not run_if_missing:
            raise FileNotFoundError(
                f"Stricta ledger file(s) missing: jsonl={jsonl_file.exists()} ({jsonl_path}), "
                f"sqlite={sqlite_file.exists()} ({db_path})"
            )
        from .drivers import run_multi_branch_driver

        if not jsonl_file.exists():
            run_multi_branch_driver(backend="jsonl", ledger_path=str(jsonl_file), reset_ledger_file=True)
        if not sqlite_file.exists():
            run_multi_branch_driver(backend="sqlite", ledger_path=str(sqlite_file), reset_ledger_file=True)

    jsonl_rows = query_jsonl_ledger(path=str(jsonl_file))
    sqlite_rows = query_sqlite_ledger(db_path=str(sqlite_file))

    assert len(jsonl_rows) == len(sqlite_rows), (
        f"Stricta ledger length mismatch: jsonl={len(jsonl_rows)} vs sqlite={len(sqlite_rows)}"
    )

    for idx, (j_row, s_row) in enumerate(zip(jsonl_rows, sqlite_rows)):
        for field in ("node_id", "phase", "tool_name", "expected_state_change"):
            assert j_row[field] == s_row[field], (
                f"Stricta ledger mismatch at index {idx} on field '{field}': "
                f"jsonl={j_row[field]!r} vs sqlite={s_row[field]!r}"
            )
        if compare_arguments:
            assert j_row["arguments"] == s_row["arguments"], (
                f"Stricta ledger arguments mismatch at index {idx}: "
                f"jsonl={j_row['arguments']!r} vs sqlite={s_row['arguments']!r}"
            )

    return {
        "matched": True,
        "entries": len(jsonl_rows),
        "jsonl_path": str(jsonl_file),
        "db_path": str(sqlite_file),
        "compare_arguments": compare_arguments,
    }
