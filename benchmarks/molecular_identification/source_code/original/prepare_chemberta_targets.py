#!/usr/bin/env python3
"""Extract the compact MSnLib target subset from the frozen ChemBERTa cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baseline_common import atomic_write_json, sha256_file


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--manifest-dir",required=True)
    parser.add_argument("--chemberta-cache",required=True); parser.add_argument("--output-dir",required=True)
    args=parser.parse_args(); manifest=Path(args.manifest_dir); output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    smiles_path = manifest / "target_smiles.json"
    if not smiles_path.exists():
        smiles_path = manifest / "unique_target_smiles.json"
    smiles=json.loads(smiles_path.read_text())
    print(f"loading ChemBERTa cache {args.chemberta_cache}",flush=True)
    cache=torch.load(args.chemberta_cache,map_location="cpu",weights_only=False)
    first=next(iter(cache.values())); dim=int(np.asarray(first).shape[-1])
    targets=np.lib.format.open_memmap(output/"target_chemberta_float32.npy","w+",dtype=np.float32,
                                     shape=(len(smiles),dim))
    valid=np.zeros(len(smiles),dtype=bool)
    for index,smi in enumerate(smiles):
        if smi in cache:
            value=cache[smi]
            targets[index]=value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            valid[index]=True
        else: targets[index]=0
    targets.flush(); np.save(output/"target_chemberta_valid.npy",valid)
    summary={"status":"complete","n_target_smiles":len(smiles),"embedding_dim":dim,
             "n_valid":int(valid.sum()),"n_missing":int((~valid).sum()),
             "chemberta_cache_sha256":sha256_file(args.chemberta_cache),
             "target_smiles_sha256":sha256_file(smiles_path),
             "targets_sha256":sha256_file(output/"target_chemberta_float32.npy"),
             "valid_sha256":sha256_file(output/"target_chemberta_valid.npy")}
    atomic_write_json(output/"projection_target_summary.json",summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__": main()
