"""Fine-tune UltraMS on MassSpecGym spectra sharing a molecular InChIKey."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from ultrams import UltraMS


DATASET = "roman-bushuiev/MassSpecGym"
REVISION = "d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a"
FILENAME = "data/MassSpecGym.tsv"
SHA256 = "0c9cc50450def3f0d4fe2dc09dea1105fc15e635db8c6656bc3e3be37a3bcd95"
FOLDS = ("train", "val", "test")


def checked_data_file(path: Path | None) -> Path:
    if path is None:
        path = Path(hf_hub_download(repo_id=DATASET, repo_type="dataset", filename=FILENAME, revision=REVISION))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise ValueError(f"MassSpecGym SHA-256 mismatch for {path}")
    return path


def rows_in(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        needed = {"identifier", "mzs", "intensities", "inchikey", "precursor_mz", "fold"}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"MassSpecGym is missing columns: {sorted(missing)}")
        yield from reader


def parse_spectrum(row: dict[str, str]):
    key = row["inchikey"].strip()
    if not key:
        return None
    try:
        mz = np.fromstring(row["mzs"], sep=",", dtype=np.float32)
        intensity = np.fromstring(row["intensities"], sep=",", dtype=np.float32)
        precursor = float(row["precursor_mz"])
    except ValueError:
        return None
    if mz.size != intensity.size or not np.isfinite(precursor) or precursor <= 0:
        return None
    valid = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0)
    effective = valid & (intensity > 0)
    if int(effective.sum()) < 3:
        return None
    # Keep zero-intensity peaks, as UltraMS inference does.
    mz, intensity = mz[valid], intensity[valid]
    fingerprint = hashlib.blake2b(
        mz.tobytes() + intensity.tobytes() + np.float32(precursor).tobytes(), digest_size=16
    ).hexdigest()
    return {
        "id": row["identifier"],
        "inchikey": key,
        "mz": mz,
        "intensity": intensity,
        "precursor_mz": precursor,
        "fingerprint": fingerprint,
    }


def selected_keys(groups: dict[str, set[str]], fold: str, seed: int, limit: int | None):
    keys = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}:{fold}:{key}".encode()).digest())
    return set(keys if limit is None else keys[:limit])


def prepare(path: Path, *, seed: int, full: bool, limits: dict[str, int]):
    raw = Counter()
    valid = Counter()
    fingerprints = {fold: defaultdict(set) for fold in FOLDS}
    for row in rows_in(path):
        fold = row["fold"]
        if fold not in FOLDS:
            raise ValueError(f"unexpected MassSpecGym fold: {fold!r}")
        raw[fold] += 1
        spectrum = parse_spectrum(row)
        if spectrum is None:
            continue
        valid[fold] += 1
        fingerprints[fold][spectrum["inchikey"]].add(spectrum["fingerprint"])

    keys = {fold: set(fingerprints[fold]) for fold in FOLDS}
    for i, left in enumerate(FOLDS):
        for right in FOLDS[i + 1 :]:
            if keys[left] & keys[right]:
                raise ValueError(f"molecular overlap between {left} and {right}")

    eligible = {
        fold: {key: values for key, values in fingerprints[fold].items() if len(values) >= 2}
        for fold in FOLDS
    }
    chosen = {
        "train": selected_keys(eligible["train"], "train", seed, None if full else limits["train"]),
        "val": selected_keys(fingerprints["val"] if full else eligible["val"], "val", seed, None if full else limits["val"]),
        "test": selected_keys(fingerprints["test"] if full else eligible["test"], "test", seed, None if full else limits["test"]),
    }
    if any(not chosen[fold] for fold in FOLDS):
        raise ValueError("at least one fold has no selected molecules with valid spectra")

    spectra = {fold: defaultdict(list) for fold in FOLDS}
    seen = {fold: defaultdict(set) for fold in FOLDS}
    duplicates = Counter()
    for row in rows_in(path):
        fold, key = row["fold"], row["inchikey"].strip()
        if key not in chosen[fold]:
            continue
        spectrum = parse_spectrum(row)
        if spectrum is None:
            continue
        fingerprint = spectrum.pop("fingerprint")
        if fingerprint in seen[fold][key]:
            duplicates[fold] += 1
            continue
        seen[fold][key].add(fingerprint)
        if not full and len(spectra[fold][key]) >= 2:
            continue
        spectra[fold][key].append(spectrum)

    for fold in FOLDS:
        if fold == "train" or not full:
            if any(len(group) < 2 for group in spectra[fold].values()):
                raise ValueError(f"{fold} contains a selected molecule without two distinct spectra")

    audit = {
        "source": {"dataset": DATASET, "revision": REVISION, "file": FILENAME, "sha256": SHA256},
        "mode": "full" if full else "teaching subset",
        "selection_seed": seed,
        "folds": {
            fold: {
                "source_rows": raw[fold],
                "valid_rows_at_least_three_peaks": valid[fold],
                "source_molecules_after_filter": len(fingerprints[fold]),
                "molecules_with_two_distinct_spectra": len(eligible[fold]),
                "selected_rows": sum(map(len, spectra[fold].values())),
                "selected_molecules": len(spectra[fold]),
                "selected_molecules_with_two_spectra": sum(len(group) >= 2 for group in spectra[fold].values()),
                "retrieval_queries_with_positive": sum(
                    len(group) for group in spectra[fold].values() if len(group) >= 2
                ),
                "exact_duplicate_spectra_removed": duplicates[fold],
            }
            for fold in FOLDS
        },
        "molecule_overlap_between_folds": 0,
    }
    return spectra, audit


def pairs_for_epoch(groups, seed: int, epoch: int):
    rng = random.Random(seed + epoch)
    keys = list(groups)
    rng.shuffle(keys)
    return [rng.sample(groups[key], 2) for key in keys]


def pair_batches(pairs, batch_size: int):
    batches = [pairs[start : start + batch_size] for start in range(0, len(pairs), batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return batches


def top1(model: UltraMS, groups, *, batch_size: int, device: torch.device):
    records = [record for molecule in groups.values() for record in molecule]
    labels = [record["inchikey"] for record in records]
    counts = Counter(labels)
    embeddings = model.encode_batch(records, batch_size=batch_size)
    vectors = F.normalize(torch.from_numpy(embeddings).to(device), dim=-1)
    correct = queries = 0
    for start in range(0, len(records), 256):
        end = min(start + 256, len(records))
        scores = vectors[start:end] @ vectors.T
        scores[torch.arange(end - start, device=device), torch.arange(start, end, device=device)] = -float("inf")
        nearest = scores.argmax(dim=1).cpu().tolist()
        for offset, neighbour in enumerate(nearest):
            index = start + offset
            if counts[labels[index]] >= 2:
                queries += 1
                correct += labels[index] == labels[neighbour]
    return {"top1": correct / queries, "correct": correct, "queries": queries, "candidates": len(records)}


def write_json(path: Path, content) -> None:
    path.write_text(json.dumps(content, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, help="Use an already downloaded copy of the pinned TSV")
    parser.add_argument("--output-dir", type=Path, help="Keep run files here; otherwise use a temporary directory")
    parser.add_argument("--full", action="store_true", help="Use all valid spectra, not the teaching subset")
    parser.add_argument("--train-molecules", type=int, default=64)
    parser.add_argument("--val-molecules", type=int, default=24)
    parser.add_argument("--test-molecules", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8, help="Number of positive pairs per training batch")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prepare-only", action="store_true", help="Audit and select data without loading the model")
    args = parser.parse_args()
    if any(value < 1 for value in (args.train_molecules, args.val_molecules, args.test_molecules, args.epochs)) or args.batch_size < 2:
        parser.error("molecule limits and epochs must be positive; batch size needs at least two positive pairs")
    if args.lr <= 0 or args.temperature <= 0:
        parser.error("learning rate and temperature must be positive")

    output = args.output_dir or Path(tempfile.mkdtemp(prefix="ultrams-massspecgym-"))
    output.mkdir(parents=True, exist_ok=True)
    path = checked_data_file(args.data_file)
    groups, audit = prepare(
        path,
        seed=args.seed,
        full=args.full,
        limits={"train": args.train_molecules, "val": args.val_molecules, "test": args.test_molecules},
    )
    if len(groups["train"]) < 2:
        raise ValueError("contrastive training needs at least two distinct training molecules")
    write_json(output / "data_audit.json", audit)
    config = {
        "pretrained_model": "unsupervised",
        "objective": "in-batch contrastive learning; positives are two distinct spectra with the same MassSpecGym inchikey",
        "evaluation": "same-fold spectrum retrieval Top1, self-match excluded; singleton queries excluded",
        "dataset": audit["source"],
        "mode": audit["mode"],
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size_in_positive_pairs": args.batch_size,
        "learning_rate": args.lr,
        "temperature": args.temperature,
        "device": args.device,
        "output_dir": str(output.resolve()),
    }
    write_json(output / "config.json", config)
    print(json.dumps(audit["folds"], indent=2), flush=True)
    print(f"Run directory: {output.resolve()}", flush=True)
    if args.prepare_only:
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = UltraMS.from_pretrained("unsupervised", device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    best_top1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pairs = pairs_for_epoch(groups["train"], args.seed, epoch)
        sum_loss = 0.0
        for mini in pair_batches(pairs, args.batch_size):
            batch = model.collate([spectrum for pair in mini for spectrum in pair])
            embeddings = model(
                batch["peaks"].to(device), batch["attention_mask"].to(device), batch["precursor_mz"].to(device)
            )
            logits = embeddings @ embeddings.T / args.temperature
            logits = logits.masked_fill(torch.eye(len(embeddings), device=device, dtype=torch.bool), -float("inf"))
            target = torch.arange(len(embeddings), device=device) ^ 1
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += float(loss.detach()) * len(mini)
        validation = top1(model, groups["val"], batch_size=args.batch_size * 2, device=device)
        result = {"epoch": epoch, "train_loss": sum_loss / len(pairs), "validation": validation}
        history.append(result)
        write_json(output / "history.json", history)
        print(json.dumps(result), flush=True)
        if validation["top1"] > best_top1:
            best_top1 = validation["top1"]
            torch.save({"model_state_dict": model.model.state_dict(), "config": model.config}, output / "best.pt")

    best = UltraMS.from_checkpoint(output / "best.pt", device=device)
    test = top1(best, groups["test"], batch_size=args.batch_size * 2, device=device)
    write_json(output / "test.json", test)
    print("Test on the best validation checkpoint:", json.dumps(test), flush=True)


if __name__ == "__main__":
    main()
