#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Dict, List


def _run(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return ""
    return proc.stdout


def _tail_text(path: Path, max_bytes: int = 512_000) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with open(path, "rb") as handle:
        if size > max_bytes:
            handle.seek(-max_bytes, os.SEEK_END)
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def discover_latest_launches(repo_root: Path) -> Dict[str, Dict[str, str]]:
    pattern = repo_root / "local_artifacts" / "experiments" / "baselines" / "tmux_relaunch_*" / "tmux_launch_manifest.csv"
    manifests = sorted(repo_root.glob(str(pattern.relative_to(repo_root))))
    launches: Dict[str, Dict[str, str]] = {}
    for manifest in manifests:
        parent = manifest.parent.parent.name
        with open(manifest, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                job_name = row.get("job_name", "")
                if not job_name:
                    continue
                row["launch_dir"] = parent
                row["manifest_path"] = str(manifest)
                launches[job_name] = row
    return launches


def parse_srun_job_id(log_file: Path) -> str:
    text = _tail_text(log_file)
    matches = re.findall(r"srun: job (\d+)", text)
    if matches:
        return matches[-1]
    return ""


def get_tmux_sessions() -> set[str]:
    output = _run(["tmux", "ls"])
    sessions = set()
    for line in output.splitlines():
        if ":" in line:
            sessions.add(line.split(":", 1)[0].strip())
    return sessions


def get_squeue_map(user: str) -> Dict[str, Dict[str, str]]:
    output = _run(["squeue", "-h", "-u", user, "-o", "%i|%T|%M|%P|%j|%R"])
    data: Dict[str, Dict[str, str]] = {}
    for line in output.splitlines():
        parts = line.split("|", 5)
        if len(parts) != 6:
            continue
        job_id, state, elapsed, partition, name, reason = parts
        data[job_id.strip()] = {
            "state": state.strip(),
            "elapsed": elapsed.strip(),
            "partition": partition.strip(),
            "name": name.strip(),
            "reason": reason.strip(),
        }
    return data


def get_sacct_map(job_ids: List[str]) -> Dict[str, Dict[str, str]]:
    ids = [job_id for job_id in job_ids if job_id]
    if not ids:
        return {}
    output = _run([
        "sacct",
        "-j",
        ",".join(ids),
        "--format=JobIDRaw,State,Elapsed,ExitCode,Partition",
        "-P",
        "-n",
    ])
    data: Dict[str, Dict[str, str]] = {}
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        raw_id, state, elapsed, exit_code, partition = [p.strip() for p in parts[:5]]
        top_id = raw_id.split(".", 1)[0]
        if not top_id.isdigit():
            continue
        if top_id in data:
            continue
        data[top_id] = {
            "state": state,
            "elapsed": elapsed,
            "exit_code": exit_code,
            "partition": partition,
        }
    return data


def write_outputs(rows: List[Dict[str, str]], output_csv: Path, output_md: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "job_name",
        "dataset",
        "splitter",
        "session",
        "tmux_alive",
        "partition",
        "srun_job_id",
        "state",
        "elapsed",
        "reason",
        "log_file",
        "launch_dir",
        "updated_at_utc",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})

    total = len(rows)
    running = sum(1 for row in rows if row.get("state") == "RUNNING")
    pending = sum(1 for row in rows if row.get("state") == "PENDING")
    failed = sum(1 for row in rows if row.get("state") in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"})
    completed = sum(1 for row in rows if row.get("state") == "COMPLETED")

    lines = [
        "# FL tmux status",
        "",
        f"Updated (UTC): {dt.datetime.now(dt.UTC).isoformat(timespec='seconds')}",
        "",
        f"- Total jobs: {total}",
        f"- Running: {running}",
        f"- Pending: {pending}",
        f"- Completed: {completed}",
        f"- Failed/Cancelled: {failed}",
        "",
        "| job_name | dataset | session | tmux | partition | srun_job_id | state | elapsed | reason |",
        "|---|---|---:|:---:|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {job_name} | {dataset} | {session} | {tmux_alive} | {partition} | {srun_job_id} | {state} | {elapsed} | {reason} |".format(
                job_name=row.get("job_name", ""),
                dataset=row.get("dataset", ""),
                session=row.get("session", ""),
                tmux_alive="yes" if row.get("tmux_alive") == "yes" else "no",
                partition=row.get("partition", ""),
                srun_job_id=row.get("srun_job_id", ""),
                state=row.get("state", ""),
                elapsed=row.get("elapsed", ""),
                reason=row.get("reason", "").replace("|", "/"),
            )
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_rows(repo_root: Path, user: str) -> List[Dict[str, str]]:
    launches = discover_latest_launches(repo_root)
    tmux_sessions = get_tmux_sessions()
    rows: List[Dict[str, str]] = []

    for job_name, row in sorted(launches.items()):
        session = row.get("session", "")
        log_file = Path(row.get("log_file", ""))
        srun_job_id = parse_srun_job_id(log_file) if log_file else ""
        rows.append(
            {
                "job_name": job_name,
                "dataset": row.get("dataset", ""),
                "splitter": row.get("splitter", ""),
                "session": session,
                "tmux_alive": "yes" if session in tmux_sessions else "no",
                "partition": row.get("partition", ""),
                "srun_job_id": srun_job_id,
                "state": "UNKNOWN",
                "elapsed": "",
                "reason": "",
                "log_file": str(log_file),
                "launch_dir": row.get("launch_dir", ""),
                "updated_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            }
        )

    squeue_map = get_squeue_map(user)
    sacct_map = get_sacct_map([r.get("srun_job_id", "") for r in rows])

    for row in rows:
        jid = row.get("srun_job_id", "")
        if jid and jid in squeue_map:
            row["state"] = squeue_map[jid]["state"]
            row["elapsed"] = squeue_map[jid]["elapsed"]
            row["reason"] = squeue_map[jid]["reason"]
            row["partition"] = squeue_map[jid]["partition"] or row["partition"]
            continue
        if jid and jid in sacct_map:
            row["state"] = sacct_map[jid]["state"]
            row["elapsed"] = sacct_map[jid]["elapsed"]
            row["reason"] = sacct_map[jid]["exit_code"]
            row["partition"] = sacct_map[jid]["partition"] or row["partition"]

    return rows


def monitor(repo_root: Path, user: str, output_csv: Path, output_md: Path, interval: int, loop: bool) -> None:
    while True:
        rows = build_rows(repo_root, user)
        write_outputs(rows, output_csv, output_md)
        print(f"[{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')}] updated {len(rows)} rows")
        if not loop:
            return
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and refresh FL tmux status table.")
    parser.add_argument("--repo-root", type=str, default="/path/to/data/FDS")
    parser.add_argument("--user", type=str, default=os.environ.get("USER", ""))
    parser.add_argument(
        "--output-csv",
        type=str,
        default="/path/to/data/FDS/local_artifacts/experiments/baselines/fl_tmux_status.csv",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="/path/to/data/FDS/local_artifacts/experiments/baselines/fl_tmux_status.md",
    )
    parser.add_argument("--interval", type=int, default=600, help="Polling interval in seconds.")
    parser.add_argument("--loop", action="store_true", help="Continuously refresh every --interval seconds.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_md = Path(args.output_md).resolve()

    monitor(
        repo_root=repo_root,
        user=args.user,
        output_csv=output_csv,
        output_md=output_md,
        interval=max(10, int(args.interval)),
        loop=bool(args.loop),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
