import json
from pathlib import Path

from cprflow.types import DATASETS, LANGUAGES, PROFILE


def read_manifest(path, expected_split=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    seen = set()
    for row in rows:
        if row["uid"] in seen:
            raise ValueError(f"Duplicate UID {row['uid']}")
        seen.add(row["uid"])
        if expected_split and row["split"] != expected_split:
            raise ValueError("Split mismatch")
        if row["dataset_id"] not in DATASETS or row["split"] not in ("train", "dev", "test"):
            raise ValueError("Unknown corpus or split")
        language = DATASETS[row["dataset_id"]][0]
        if row["sign_language_id"] != LANGUAGES[language] or row["profile"] != PROFILE:
            raise ValueError("Language/profile mismatch")
    return rows


def check_disjoint(manifests):
    sets = [{row["uid"] for row in rows} for rows in manifests]
    for i, a in enumerate(sets):
        for b in sets[i + 1 :]:
            if a & b:
                raise ValueError(f"Cross-split UID leakage: {sorted(a & b)[:3]}")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    )
    tmp.replace(path)
