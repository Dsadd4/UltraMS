"""Link a frozen UltraMSdata bundle to the public training layout.

The original Parquet files and manifests stay in place. Dataset fingerprints
select the exact clean and polarity-balanced inputs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ALIASES = {
    "clean": "ultramsdata_clean",
    "polarity": "ultramsdata_polarity",
}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _fingerprints(config_path: Path) -> dict[str, str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assets = config["expected_assets"]
    result = {}
    for role in ALIASES:
        value = assets[role]["dataset_fingerprint"]
        if not isinstance(value, str) or SHA256.fullmatch(value) is None:
            raise ValueError(f"{role} dataset_fingerprint must be a SHA-256 value")
        result[role] = value
    if result["clean"] == result["polarity"]:
        raise ValueError("clean and polarity dataset fingerprints must differ")
    return result


def link_ultramsdata(data_root: Path, config_path: Path) -> dict[str, str]:
    derived = data_root.resolve() / "derived"
    if not derived.is_dir():
        raise FileNotFoundError(f"UltraMSdata derived directory is missing: {derived}")
    fingerprints = _fingerprints(config_path)
    manifests = sorted(derived.glob("*/shards/manifest.json"))
    selected = {}
    for role, digest in fingerprints.items():
        candidates = []
        for path in manifests:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if (
                manifest.get("status") == "complete"
                and manifest.get("dataset_fingerprint") == digest
            ):
                candidates.append(path.parent.parent.resolve())
        targets = sorted(set(candidates))
        if len(targets) != 1:
            raise ValueError(
                f"expected exactly one {role} UltraMSdata dataset with fingerprint {digest}; "
                f"found {len(targets)}"
            )
        selected[role] = targets[0]

    aliases = {}
    for role, target in selected.items():
        alias = derived / ALIASES[role]
        if alias.is_symlink() or alias.exists():
            if alias.resolve() != target:
                raise ValueError(f"UltraMSdata alias points to another dataset: {alias}")
        else:
            alias.symlink_to(target.name, target_is_directory=True)
        aliases[role] = str(alias / "shards")
    return aliases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(link_ultramsdata(args.data_root, args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
