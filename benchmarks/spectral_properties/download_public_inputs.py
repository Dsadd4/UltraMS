"""Download the public source files used by spectral-property experiments.

Each asset is requested explicitly. Hugging Face Hub keeps partial downloads and
resumes them on a later invocation. Files are checked against source checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import Request, urlopen

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download
from requests.exceptions import RequestException


@dataclass(frozen=True)
class Asset:
    repo: str
    repo_type: str
    revision: str
    filename: str
    size: int
    sha256: str


ASSETS = {
    "massspecgym": Asset(
        "roman-bushuiev/MassSpecGym", "dataset",
        "d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a",
        "data/MassSpecGym.tsv", 262334768,
        "0c9cc50450def3f0d4fe2dc09dea1105fc15e635db8c6656bc3e3be37a3bcd95",
    ),
    "gems": Asset(
        "roman-bushuiev/MassSpecGym", "dataset",
        "d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a",
        "data/spectra/GeMS_A10.hdf5", 14625229880,
        "572065f6e63fc3f6019eda312006d8f3d4be9890157467553501148b322f0488",
    ),
    "mona_exclusion": Asset(
        "roman-bushuiev/GeMS", "dataset",
        "0603a815cef2cc78591a6dc88dfd7f87c41037cb",
        "data/auxiliary/MoNA_A_Murcko_split_neighbours_[M+H]+_0.05Da.pkl", 81307355,
        "62860286bee766c647cdd3b5860da7c7ea961197f7d27e39a65643cfa80b333a",
    ),
    "dreams_ssl": Asset(
        "roman-bushuiev/DreaMS", "model",
        "c81a62766b10dd1d39fcda3edec5ef88623e5f6b",
        "ssl_model.ckpt", 1392710212,
        "4b73da583a4b4e4abef4bb3ab496dc12f716ed484ea0e4066ad45d6952856fef",
    ),
    "dreams_embedding": Asset(
        "roman-bushuiev/DreaMS", "model",
        "c81a62766b10dd1d39fcda3edec5ef88623e5f6b",
        "embedding_model.ckpt", 1241290418,
        "630ba2e5fd0d2ac288fe32772ed73f9bc7d0f4c45759490cc856a96087dd12f4",
    ),
    "ultrams_unsupervised": Asset(
        "dsadd4/UltraMS-Unsupervised", "model",
        "0dda5bc548e5de76b5e149a76c74fe86d3fef961",
        "model.pt", 834511727,
        "6a4c6660999848c409303119f6caa54fbae9444d0b75c7fcd8bafd303cde9830",
    ),
    "ultrams_mona": Asset(
        "dsadd4/UltraMS-MoNA-Contrastive", "model",
        "846ec9ac8c4917549626c7c6a582996e51de713c",
        "model.pt", 1245933868,
        "3b8ae5bd85ff8f6991f8a78b6d70531aab74f32914878730e7296aaddb79afb0",
    ),
    "masked_peak_reconstruction": Asset(
        "dsadd4/UltraMS-Benchmark-Assets", "dataset",
        "535afb61c9a4c6fb05dd5426cd5ff3299b24bdb8",
        "spectral_properties/masked_peak_reconstruction.pt", 830229650,
        "fa7d194d707eb82a7d1dfbffa591da42a4aa14d363d8f707fea48306a4805f3d",
    ),
    "massspecgym_labels": Asset(
        "dsadd4/UltraMS-Benchmark-Assets", "dataset",
        "535afb61c9a4c6fb05dd5426cd5ff3299b24bdb8",
        "spectral_properties/massspecgym_labels.parquet", 70952194,
        "e5b036b507f69bed00bb37a0742211723f8c66dbcd99e5525a8a9b65f499e535",
    ),
}


@dataclass(frozen=True)
class ZenodoAsset:
    filename: str
    size: int
    md5: str


ZENODO_ASSETS = {
    "msnlib_spectra": ZenodoAsset(
        "SpecBridge_MSnLib_dataset.mgf", 1165459632,
        "e7c648b89841d10759f6b796aa7e3e50",
    ),
    "spectraverse_spectra": ZenodoAsset(
        "SpecBridge_Spectraverse_dataset.mgf", 751173810,
        "6af0588372e00781ed23a018841938aa",
    ),
}
ZENODO_RECORD = "https://zenodo.org/api/records/18357418/files"

PEAK_NEIGHBOR_REVISION = "535afb61c9a4c6fb05dd5426cd5ff3299b24bdb8"
PEAK_NEIGHBOR_FILES = {
    "msnlib_magma_annotations.json": (131619459, "e48c9498ca54c1b93976009fd049a8217b5f8795c9f17ef0aaa0942467fff3cb"),
    "msnlib_magma_subset.csv": (22120656, "a2979dc3fe1ff49d433e5c2be0539034836d15e0c8e17e6922a55b0819976d91"),
    "msnlib_magma_subset.summary.json": (494, "1cdae97de5ab01f71b5005c226806669abaa8e0028e04f7d1aa435f626477c0e"),
    "peak_embedding_metadata.csv": (43885916, "ae9b08c995a847763d3e8c2d956f7180f8f75b17a8339ec791c857cc2b6c7cd1"),
    "peak_embedding_summary.json": (492, "2d8415b688b9d18fd5e9332efa5202a25927d17b8e83cdd4d879814608c5bfc8"),
    "peak_embeddings_dreams.npy": (721461376, "0bdd270c0d35bf9c348b16f612145915736eea68cbc2a538f60abc8b18451570"),
    "peak_embeddings_ultra.npy": (721461376, "9126cbf9565a5b3c2a597baed8bb06649b9283ea85329f3d1ea68a3daf533527"),
}

NEGATIVE_ION_STRUCTURE_REVISION = "535afb61c9a4c6fb05dd5426cd5ff3299b24bdb8"
NEGATIVE_ION_STRUCTURE_FILES = {
    "benchmark_manifest.json": (1171, "c3d4e287f3ed44ef1a2f2ae79bd574484ec3b532a0772a23124d67a2c5502dfe"),
    "test_metadata.csv": (280875, "ba2ed918b745a8a2402c4aef913b72efe30bd44f1536df53872da523c9935ae7"),
    "test_morgan_r2_2048.npy": (3260544, "8cde9a6d37713926cb320c305dd1eaa1253b212286d697d525355ab8371d7209"),
    "test_top150_peaks.npz": (599101, "f262073c8c462667c2368d9ed659aa596cd4645af1eea912b4dcc84b89747223"),
    "val_metadata.csv": (270098, "a192448d5b4119c52327c1c760c9b382dbeeaaa43b418202d39bb42df162df9b"),
    "val_morgan_r2_2048.npy": (3199104, "87d50eb1e87e5275181b894a404c7e87a99f360b517a6440fe1feef7924ab025"),
    "val_top150_peaks.npz": (587920, "155fd6b1ab4f98009a9158138b7f86dfae965684bf57428f33f48b5c757321aa"),
}

ATOM_QUERY_HEADS_REVISION = "73cdf075b9f0493cd8ce561828272aa566c28f1b"
ATOM_QUERY_HEADS_FILES = {
    "ultrams_probe.pt": (42067381, "2e61867fbe9505e4d78bff9150632e9fe36b5af7ab8677e4f2aa683781c0bf65"),
    "dreams_probe.pt": (42067414, "44ed616bae6a283bd05a9955036a823dc9d5af36705e670701e4b7f3f0ae0ccc"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_hf(asset: Asset, directory: Path) -> Path:
    destination = directory / asset.filename
    if destination.exists() and destination.stat().st_size == asset.size and sha256(destination) == asset.sha256:
        return destination
    for attempt in range(20):
        try:
            path = Path(hf_hub_download(
                repo_id=asset.repo,
                repo_type=asset.repo_type,
                revision=asset.revision,
                filename=asset.filename,
                local_dir=directory,
            ))
            break
        except (RequestException, OSError):
            if attempt == 19:
                raise
            print(f"Resuming interrupted download: {asset.filename} (attempt {attempt + 2}/20)", flush=True)
            time.sleep(min(2 ** attempt, 30))
    actual_size = path.stat().st_size
    actual_sha = sha256(path)
    if actual_size != asset.size or actual_sha != asset.sha256:
        raise ValueError(f"Source mismatch for {asset.filename}: size={actual_size}, sha256={actual_sha}")
    return path


def download_zenodo(asset: ZenodoAsset, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / asset.filename
    if destination.exists() and destination.stat().st_size == asset.size and md5(destination) == asset.md5:
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists() and partial.stat().st_size > asset.size:
        partial.unlink()
    if partial.exists() and partial.stat().st_size == asset.size:
        if md5(partial) == asset.md5:
            partial.replace(destination)
            return destination
        partial.unlink()
    offset = partial.stat().st_size if partial.exists() else 0
    if offset < asset.size:
        url = f"{ZENODO_RECORD}/{asset.filename}/content"
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        response = urlopen(Request(url, headers=headers), timeout=60)
        if offset and response.status != 206:
            response.close()
            partial.unlink()
            offset = 0
            response = urlopen(Request(url), timeout=60)
        with response, partial.open("ab" if offset else "wb") as output:
            for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                output.write(chunk)
    actual_size = partial.stat().st_size
    actual_md5 = md5(partial)
    if actual_size != asset.size or actual_md5 != asset.md5:
        raise ValueError(f"Source mismatch for {asset.filename}: size={actual_size}, md5={actual_md5}")
    partial.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show available files and sizes")
    parser.add_argument("--asset", nargs="+", choices=sorted(set(ASSETS) | set(ZENODO_ASSETS) | {"peak_neighbors", "negative_ion_structure", "atom_query_heads"}))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    if args.list:
        for name, asset in ASSETS.items():
            print(f"{name:21s} {asset.size / 1e9:6.2f} GB  {asset.repo}/{asset.filename}")
        for name, asset in ZENODO_ASSETS.items():
            print(f"{name:21s} {asset.size / 1e9:6.2f} GB  Zenodo {asset.filename}")
        print(f"{'peak_neighbors':21s} {sum(size for size, _ in PEAK_NEIGHBOR_FILES.values()) / 1e9:6.2f} GB  seven frozen benchmark inputs")
        print(f"{'negative_ion_structure':21s} {sum(size for size, _ in NEGATIVE_ION_STRUCTURE_FILES.values()) / 1e9:6.2f} GB  seven frozen benchmark inputs")
        print(f"{'atom_query_heads':21s} {sum(size for size, _ in ATOM_QUERY_HEADS_FILES.values()) / 1e9:6.2f} GB  two selected heteroatom-count probes")
    if not args.asset:
        if not args.list:
            parser.error("select one or more files with --asset, or use --list")
        return
    if args.output_root is None:
        parser.error("--output-root is required when downloading")

    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in args.asset:
        if name == "atom_query_heads":
            files = []
            for filename, (size, digest) in ATOM_QUERY_HEADS_FILES.items():
                asset = Asset(
                    "dsadd4/UltraMS-Benchmark-Assets", "dataset", ATOM_QUERY_HEADS_REVISION,
                    f"spectral_properties/heteroatom_count/{filename}", size, digest,
                )
                path = download_hf(asset, root / "benchmark_assets")
                files.append({**asdict(asset), "path": str(path)})
                print(f"{name}: {path}", flush=True)
            manifest[name] = files
            continue
        if name == "peak_neighbors":
            files = []
            for filename, (size, digest) in PEAK_NEIGHBOR_FILES.items():
                asset = Asset(
                    "dsadd4/UltraMS-Benchmark-Assets", "dataset", PEAK_NEIGHBOR_REVISION,
                    f"spectral_properties/peak_neighbors/{filename}", size, digest,
                )
                path = download_hf(asset, root / "benchmark_assets")
                files.append({**asdict(asset), "path": str(path)})
                print(f"{name}: {path}", flush=True)
            manifest[name] = files
            continue
        if name == "negative_ion_structure":
            files = []
            for filename, (size, digest) in NEGATIVE_ION_STRUCTURE_FILES.items():
                asset = Asset(
                    "dsadd4/UltraMS-Benchmark-Assets", "dataset", NEGATIVE_ION_STRUCTURE_REVISION,
                    f"spectral_properties/negative_ion_structure/{filename}", size, digest,
                )
                path = download_hf(asset, root / "benchmark_assets")
                files.append({**asdict(asset), "path": str(path)})
                print(f"{name}: {path}", flush=True)
            manifest[name] = files
            continue
        if name in ZENODO_ASSETS:
            asset = ZENODO_ASSETS[name]
            path = download_zenodo(asset, root / name)
            manifest[name] = {**asdict(asset), "path": str(path), "record": ZENODO_RECORD}
            print(f"{name}: {path}", flush=True)
            continue
        asset = ASSETS[name]
        directory = root / "benchmark_assets" if name in {"masked_peak_reconstruction", "massspecgym_labels"} else root / name
        path = download_hf(asset, directory)
        manifest[name] = {**asdict(asset), "path": str(path)}
        print(f"{name}: {path}", flush=True)
    manifest_path = root / "public_inputs.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    previous.update(manifest)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(previous, indent=2) + "\n")
    temporary.replace(manifest_path)
    print(f"Paths and sources: {manifest_path}")


if __name__ == "__main__":
    main()
