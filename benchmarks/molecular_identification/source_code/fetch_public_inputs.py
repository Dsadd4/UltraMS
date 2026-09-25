#!/usr/bin/env python3
"""Download public benchmark inputs and model weights with resumable transfers."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import os
import time
import urllib.request
from pathlib import Path


MASS_HF_REVISION = "d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a"
BENCHMARK_REVISION = "a974c40d65af90b217589c8a6980cdd114868fed"
SINGLE_SPECTRUM_PROJECTION_REVISION = "e9fe24af878f29ff1dda8cef919aab5d68d8ea33"
SINGLE_SPECTRUM_READOUT_REVISION = "b15af172a7b875fedcc3aa85554d097f8d250829"
SINGLE_SPECTRUM_DREAMS_PROJECTION_REVISION = "171426d7754368f02f5d7ab9bfd5e65b69c17c88"
SOURCES = {
    "msnlib": [
        ("MSnLib/SpecBridge_MSnLib_dataset.mgf",
         "https://zenodo.org/api/records/18357418/files/SpecBridge_MSnLib_dataset.mgf/content",
         1_165_459_632, "md5", "e7c648b89841d10759f6b796aa7e3e50"),
        ("MSnLib/SpecBridge_MSnLib_candidates.pkl",
         "https://zenodo.org/api/records/18357418/files/SpecBridge_MSnLib_candidates.pkl/content",
         1_338_443_999, "md5", "743a9dcf98f8004388f6707c1be868f2"),
    ],
    "massspecgym": [
        ("MassSpecGym/MassSpecGym.tsv",
         f"https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/{MASS_HF_REVISION}/data/MassSpecGym.tsv",
         262_334_768, "sha256", "0c9cc50450def3f0d4fe2dc09dea1105fc15e635db8c6656bc3e3be37a3bcd95"),
        ("MassSpecGym/MassSpecGym_candidates_mass.json",
         f"https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/{MASS_HF_REVISION}/data/molecules/MassSpecGym_retrieval_candidates_mass.json",
         454_710_480, "sha256", "6256d8414fe02ef28c4179135ce454d1a7b77b64454241785935ad29114cfca5"),
        ("MassSpecGym/MassSpecGym_candidates_formula.json",
         f"https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/{MASS_HF_REVISION}/data/molecules/MassSpecGym_retrieval_candidates_formula.json",
         370_650_823, "sha256", "209f59f532752e10c4457a61a208d8b3fc1dc03cacfe3a05f5eb20a23b7b6d05"),
    ],
    "nplib1": [
        ("NPLIB1/canopus_train.zip",
         "https://zenodo.org/api/records/8151490/files/canopus_train.zip/content",
         934_001_407, "md5", "ac2277361b7d0bf48288e212d2f8dea3"),
        ("NPLIB1/splits_ms_pred/split_1.tsv",
         f"https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets/resolve/{BENCHMARK_REVISION}/molecular_identification/nplib1/split_1.tsv",
         246_418, "sha256", "24c66ff1c16c0dc55a06645e20dbf41d8f77572731b19bf06bdb6a7c5ec4ff66"),
        ("NPLIB1/splits_ms_pred/split_2.tsv",
         f"https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets/resolve/{BENCHMARK_REVISION}/molecular_identification/nplib1/split_2.tsv",
         246_505, "sha256", "567b1e789fa22cc7b0c0431000b5227b9ce0b551c4a1a016e37c6145367eea68"),
        ("NPLIB1/splits_ms_pred/split_3.tsv",
         f"https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets/resolve/{BENCHMARK_REVISION}/molecular_identification/nplib1/split_3.tsv",
         246_400, "sha256", "d576fe464ba0ba5405d79acd9ea978e73cf232dfb0b7ae13dbf7296d7c78dfac"),
        ("NPLIB1/retrieval_candidates/cands_df.tsv",
         f"https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets/resolve/{BENCHMARK_REVISION}/molecular_identification/nplib1/cands_df.tsv",
         49_873_428, "sha256", "671cfba8c8f5b5f9bb619ccca09bd6f7485be564bd87a3ab31a555e272c2e641"),
    ],
    "mona_contrastive": [
        ("DreaMS_contrastive/MoNA_A_Murcko_split_neighbours_[M+H]+_0.05Da.pkl",
         "https://huggingface.co/datasets/roman-bushuiev/GeMS/resolve/d67f74258999ab2dba9f23df6df6b65a7c5ca7c1/data/auxiliary/MoNA_A_Murcko_split_neighbours_%5BM%2BH%5D%2B_0.05Da.pkl",
         81_307_355, "sha256", "62860286bee766c647cdd3b5860da7c7ea961197f7d27e39a65643cfa80b333a"),
    ],
}
WEIGHTS = [
    ("train/output/phase2_rt_only/stage_d_epoch_11.pt",
     "https://huggingface.co/dsadd4/UltraMS-Unsupervised/resolve/0dda5bc548e5de76b5e149a76c74fe86d3fef961/model.pt",
     834_511_727, "sha256", "6a4c6660999848c409303119f6caa54fbae9444d0b75c7fcd8bafd303cde9830"),
    ("train/comparison/resources/DreaMS_Check/embedding_model.ckpt",
     "https://huggingface.co/roman-bushuiev/DreaMS/resolve/c81a62766b10dd1d39fcda3edec5ef88623e5f6b/embedding_model.ckpt",
     1_241_290_418, "sha256", "630ba2e5fd0d2ac288fe32772ed73f9bc7d0f4c45759490cc856a96087dd12f4"),
    ("train/comparison/resources/DreaMS_Check/ssl_model.ckpt",
     "https://huggingface.co/roman-bushuiev/DreaMS/resolve/c81a62766b10dd1d39fcda3edec5ef88623e5f6b/ssl_model.ckpt",
     1_392_710_212, "sha256", "4b73da583a4b4e4abef4bb3ab496dc12f716ed484ea0e4066ad45d6952856fef"),
]
BENCHMARK_WEIGHTS = [
    ("train/output/comparison/dreams_contrastive_ultrams_finetune/u5_fp005_t01_elr8e6_e3_top100_seed3407_20260702e/best.pt",
     f"https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets/resolve/{BENCHMARK_REVISION}/molecular_identification/contrastive_candidate_ranking.pt",
     847_035_604, "sha256", "02c6883c934a8548603024ee50184b981335fbb988055dd56ee988711015c6f9"),
    ("train/output/comparison/dreams_contrastive_ultrams_finetune/raw_hard4_elr8e6_m010_e3_top100_seed3407_20260703h/best.pt",
     f"https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets/resolve/{BENCHMARK_REVISION}/molecular_identification/fragment_candidate_ranking.pt",
     847_034_290, "sha256", "5c2eed7b454d962ec1c000cd504251fad49919120097e5d0a14d48e247171724"),
]
COLLISION_ENERGY_WEIGHTS = [
    ("train/output/showcase/cross_ce_ultra_v2_ul2_best.pt",
     f"https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets/resolve/{BENCHMARK_REVISION}/molecular_identification/collision_energy_ultrams.pt",
     853_348_154, "sha256", "7aebb8dc86b424ed994ba70049106b9ceb9d6396d99b5fa4d36b219a68d1752c"),
    ("train/output/showcase/cross_ce_dreams_v2_ul2_best.pt",
     f"https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets/resolve/{BENCHMARK_REVISION}/molecular_identification/collision_energy_dreams.pt",
     387_480_386, "sha256", "d1d4b87c0d96f6389ee4d4d404150fea2162f4a761ef848c98abcc2a15161257"),
]
SINGLE_SPECTRUM_PREDICTIONS = [
    ("train/output/comparison/10_per_spectrum_msnlib_mass_rt_only_d11_test.jsonl", "ultrams", 221_202_344, "ff512e02f2842e34438823c3e12c79c71d0c46690d02d4049c7ba861e9763cf3"),
    ("train/output/comparison/10_per_spectrum_msnlib_mass_dreams_test.jsonl", "dreams", 221_393_163, "e34fc45ee6c729defc4c188d8a92f25df196693f669d0492825deb13e8fc22df"),
    ("train/output/comparison/fig3_simple_baselines_20260821/chemberta_readout_v1/formal_runs/linear/seed_0/evaluation/per_spectrum_msnlib_mass_linear_chemberta_test.jsonl", "linear", 234_650_054, "bd669d59d89a6f9eb348c06ac1594870d5cff4e0ff853c8d2a54447eb848d316"),
    ("train/output/comparison/fig3_simple_baselines_20260821/chemberta_readout_v1/formal_runs/deepsets/seed_0/evaluation/per_spectrum_msnlib_mass_deepsets_chemberta_test.jsonl", "deepsets", 231_068_376, "115d091897a64efe36efbeb147900bf2cb62c4f29908eb05d188df29c3b6d187"),
    ("train/output/comparison/fig3_simple_baselines_20260821/chemberta_readout_v1/formal_runs/fourier_projection/seed_0/evaluation/per_spectrum_msnlib_mass_fourier_projection_chemberta_test.jsonl", "fourier", 235_892_562, "d0b463ce373b16c1b4002f3371215e9e46b66398e20a332e0f60c1d7b760e2c0"),
    ("train/output/comparison/fig3_simple_baselines_20260821/chemberta_readout_v1/formal_runs/ultrams_codebook/seed_0/evaluation/per_spectrum_msnlib_mass_ultrams_codebook_chemberta_test.jsonl", "codebook", 236_426_166, "14f2603429c02b231a226a81da1383344992df2f4cadd79c9e58c5fe62d69b1d"),
]
SINGLE_SPECTRUM_PROJECTION = (
    "train/output/comparison/proj_msnlib_mass_rt_only_d11_seed42.pt",
    f"https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets/resolve/{SINGLE_SPECTRUM_PROJECTION_REVISION}/molecular_identification/single_spectrum_projection.pt",
    10_505_021, "sha256", "619cc886b6dc6b11c22f97d203a2788eb4ff2eb32cfb475e110500c39dbfd786",
)
SINGLE_SPECTRUM_DREAMS_PROJECTION = (
    "train/output/comparison/proj_msnlib_mass_dreams_seed42.pt",
    f"https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets/resolve/{SINGLE_SPECTRUM_DREAMS_PROJECTION_REVISION}/molecular_identification/single_spectrum_dreams_projection.pt",
    10_504_513, "sha256", "c66591093862884084fe598a2ed6ed883474556aee31e9e8d3d2427cdeeda" "e31",
)
SINGLE_SPECTRUM_READOUTS = [
    ("linear", 3_093_062, "71c50ec7bc0de6c1edfb0ea28599dce3133343d2b2b5f168c8dbe429115ee989"),
    ("deepsets", 800_362, "14fd0dbd0450da72bad9e3d27f25a13727f0f578dcd5fe84ac1ec1eb49ad2e6d"),
    ("fourier", 399_487, "6ca3e95013c35d2721b06e84425902e1c83f5b4b13842d96ca666b6a57cefe59"),
    ("codebook", 3_151_558, "1902177fa52806cd4469b61da873a62373652498524f4e50b0b68ea801d474dc"),
]
CHEMBERTA_REVISION = "f5c45f44d3061f0346888f5c09db17ec1146d29d"
CHEMBERTA = [
    ("config.json", 636, "eac5ce0c6cd8369fcc27788ba4637f2e3e5359302b45f9cc75aee8e7753654b5"),
    ("merges.txt", 101_307, "711b94fe4512ea656c3e849b9cbad47033bcc9521e0042bad5bf807d7d990020"),
    ("model.safetensors", 368_569_864, "cb04edf30124fbd44817ba7715b174aa80b1981b423f721bc8a6007a0c798a99"),
    ("special_tokens_map.json", 957, "8293ae960b0a0852d4d3813118030a1149a3ed9fe37bc2e1e7b3c2e62eb2d4b7"),
    ("tokenizer.json", 384_023, "5c2d36c785a2ebe814fb9dca9b12e7ae7661ab1e2113eb613d549cf57f3d49ba"),
    ("tokenizer_config.json", 1_269, "2a73d80d3a3699dca9a567a3499de0425f717659076eedc1b04ebc87d3420012"),
    ("vocab.json", 148_693, "9f82021617fc361ccdcb7e7a695d3940b38d342ef39af666f5b6817eb56c5df3"),
]


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path, size: int, algorithm: str, expected: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == size and digest(target, algorithm) == expected:
        print(f"ready {target}", flush=True)
        return
    partial = target.with_name(target.name + ".part")
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            partial.unlink()
            offset = 0
        if offset == size:
            break
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = response.status
                content_range = response.headers.get("Content-Range", "")
                if offset and status == 206:
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise RuntimeError(f"unexpected Content-Range: {content_range}")
                    mode = "ab"
                elif status == 200:
                    mode = "wb"
                else:
                    raise RuntimeError(f"unexpected HTTP status {status} for {url}")
                with partial.open(mode) as stream:
                    while chunk := response.read(8 << 20):
                        stream.write(chunk)
            if partial.stat().st_size == size:
                break
        except (OSError, http.client.IncompleteRead) as error:
            if attempt == 4:
                raise
            print(f"retrying {target.name} after {error}", flush=True)
        if attempt < 4:
            time.sleep(min(2 ** attempt, 16))
    if partial.stat().st_size != size:
        raise RuntimeError(f"incomplete download {partial}: {partial.stat().st_size} of {size} bytes; rerun to resume")
    actual = digest(partial, algorithm)
    if actual != expected:
        raise RuntimeError(f"checksum mismatch for {partial}: {actual} != {expected}")
    os.replace(partial, target)
    print(f"ready {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Benchmark workspace; creates ROOT/datasets")
    parser.add_argument("--dataset", choices=[*SOURCES, "msnlib_spectra", "ultrams_weight", "dreams_embedding_weight", "dreams_ssl_weight", "base_weights", "benchmark_weights", "collision_energy_weights", "single_spectrum_predictions", "single_spectrum_projection", "single_spectrum_dreams_projection", "single_spectrum_readouts", "chemberta", "all"], required=True)
    args = parser.parse_args()
    if args.dataset == "all":
        datasets = SOURCES
    elif args.dataset == "msnlib_spectra":
        datasets = {"msnlib_spectra": SOURCES["msnlib"][:1]}
    elif args.dataset in SOURCES:
        datasets = {args.dataset: SOURCES[args.dataset]}
    else:
        datasets = {}
    for sources in datasets.values():
        for relative, url, size, algorithm, expected in sources:
            download(url, args.root / "datasets" / relative, size, algorithm, expected)
    if args.dataset in ("ultrams_weight", "dreams_embedding_weight", "dreams_ssl_weight", "base_weights", "all"):
        selected = WEIGHTS if args.dataset in ("base_weights", "all") else [WEIGHTS[{
            "ultrams_weight": 0,
            "dreams_embedding_weight": 1,
            "dreams_ssl_weight": 2,
        }[args.dataset]]]
        for relative, url, size, algorithm, expected in selected:
            download(url, args.root / relative, size, algorithm, expected)
    if args.dataset in ("benchmark_weights", "all"):
        for relative, url, size, algorithm, expected in BENCHMARK_WEIGHTS:
            download(url, args.root / relative, size, algorithm, expected)
    if args.dataset in ("collision_energy_weights", "all"):
        for relative, url, size, algorithm, expected in COLLISION_ENERGY_WEIGHTS:
            download(url, args.root / relative, size, algorithm, expected)
    if args.dataset in ("single_spectrum_predictions", "all"):
        for relative, name, size, expected in SINGLE_SPECTRUM_PREDICTIONS:
            url = ("https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets/resolve/"
                   f"{BENCHMARK_REVISION}/molecular_identification/single_spectrum_test_predictions/{name}.jsonl")
            download(url, args.root / relative, size, "sha256", expected)
    if args.dataset in ("single_spectrum_projection", "all"):
        relative, url, size, algorithm, expected = SINGLE_SPECTRUM_PROJECTION
        download(url, args.root / relative, size, algorithm, expected)
    if args.dataset in ("single_spectrum_dreams_projection", "all"):
        relative, url, size, algorithm, expected = SINGLE_SPECTRUM_DREAMS_PROJECTION
        download(url, args.root / relative, size, algorithm, expected)
    if args.dataset in ("single_spectrum_readouts", "all"):
        for name, size, expected in SINGLE_SPECTRUM_READOUTS:
            url = ("https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets/resolve/"
                   f"{SINGLE_SPECTRUM_READOUT_REVISION}/molecular_identification/"
                   f"single_spectrum_readouts/reported/{name}.pt")
            download(url, args.root / "benchmark_assets/molecular_identification/single_spectrum_readouts" / f"{name}.pt",
                     size, "sha256", expected)
    if args.dataset in ("chemberta", "all"):
        for filename, size, expected in CHEMBERTA:
            url = ("https://huggingface.co/DeepChem/ChemBERTa-100M-MLM/resolve/"
                   f"{CHEMBERTA_REVISION}/{filename}")
            download(url, args.root / "model/feature/ChemBERTa-100M-MLM" / filename,
                     size, "sha256", expected)


if __name__ == "__main__":
    main()
