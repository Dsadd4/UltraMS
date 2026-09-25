"""
为 MassSpecGym 生成包含 4 个任务标签的综合 Parquet。

Task 1 — formula_count  : C/H/N/O/S/Cl/F/Br 原子数 (float32, 回归)
Task 2 — adduct         : 0=[M+H]+  1=[M+Na]+  (int8, 二分类)
Task 3 — neutral_loss   : H2O/NH3/CO2/HF/HCl/CO/CH3 中性丢失存在性 (int8 多标签)
Task 4 — mol_props      : MolWt/TPSA/LogP/nHDon/nHAcc/RingCount (float32, 回归)

Use `python benchmarks/spectral_properties/run_experiment.py ... prepare_massspecgym
-- --src MASSGYM.csv --out massgym_showcase.parquet` from the repository root.
"""
from __future__ import annotations
import argparse, os, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC_CSV     = os.environ.get('ULTRAMS_MASSGYM_CSV', 'datasets/MassSpecGym/MassSpecGym.csv')
DEFAULT_OUT = os.environ.get('ULTRAMS_MASSGYM_PARQUET', 'datasets/MassSpecGym/massgym_showcase.parquet')
MAX_PEAKS   = 150
NL_TOL_DA   = 0.02   # 中性丢失检测容差

# 元素计数目标
COUNT_ELEMENTS = ['C', 'H', 'N', 'O', 'S', 'Cl', 'F', 'Br']

# Adduct 映射（只保留这两类）
ADDUCT_MAP = {'[M+H]+': 0, '[M+Na]+': 1}

# 中性丢失：名称 → 精确质量
NEUTRAL_LOSSES = {
    'H2O':  18.0106,
    'NH3':  17.0265,
    'CO2':  43.9898,
    'HF':   20.0063,
    'HCl':  35.9767,
    'CO':   27.9949,
    'CH3':  15.0235,
}

_ELEM_RE = re.compile(r'([A-Z][a-z]?)(\d*)')


def formula_counts(formula: str) -> dict[str, int]:
    atoms: dict[str, int] = {}
    for sym, cnt in _ELEM_RE.findall(str(formula)):
        atoms[sym] = atoms.get(sym, 0) + (int(cnt) if cnt else 1)
    return atoms


def prep_spectrum(mzs_str: str, ints_str: str) -> np.ndarray | None:
    mz    = np.fromstring(mzs_str,  dtype=np.float32, sep=',')
    inten = np.fromstring(ints_str, dtype=np.float32, sep=',')
    valid = mz > 0
    mz, inten = mz[valid], inten[valid]
    if len(mz) < 3:
        return None
    mx = inten.max()
    if mx > 0:
        inten = inten / mx
    inten = np.clip(inten, 0, 1)
    if len(mz) > MAX_PEAKS:
        idx = np.argsort(inten)[-MAX_PEAKS:]
        idx = np.sort(idx)
        mz, inten = mz[idx], inten[idx]
    return np.stack([mz, inten], axis=-1).astype(np.float32)


def detect_neutral_losses(spec: np.ndarray, prec_mz: float) -> dict[str, int]:
    mzs = spec[:, 0]
    result = {}
    for name, delta in NEUTRAL_LOSSES.items():
        target = prec_mz - delta
        result[name] = int(np.any(np.abs(mzs - target) <= NL_TOL_DA))
    return result


def mol_props_from_smiles(smiles: str) -> dict[str, float] | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            'MolWt':      Descriptors.MolWt(mol),
            'TPSA':       Descriptors.TPSA(mol),
            'LogP':       Descriptors.MolLogP(mol),
            'nHDonors':   rdMolDescriptors.CalcNumHBD(mol),
            'nHAcceptors':rdMolDescriptors.CalcNumHBA(mol),
            'RingCount':  rdMolDescriptors.CalcNumRings(mol),
        }
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC_CSV)
    ap.add_argument('--out', default=DEFAULT_OUT)
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f'Already exists: {args.out}')
        _print_stats(args.out)
        return

    print(f'Loading {args.src} ...')
    t0 = time.time()
    df = pd.read_csv(args.src)
    print(f'  {len(df):,} rows  ({time.time()-t0:.1f}s)')

    rows, skip = [], 0
    for i, row in enumerate(df.itertuples(index=False)):
        if i % 20_000 == 0:
            print(f'  {i:,}/{len(df):,}', flush=True)

        spec = prep_spectrum(row.mzs, row.intensities)
        if spec is None:
            skip += 1
            continue

        # adduct filter
        adduct_cls = ADDUCT_MAP.get(str(row.adduct).strip(), -1)
        if adduct_cls == -1:
            skip += 1
            continue

        # element counts
        counts = formula_counts(row.formula)

        # molecular properties
        props = mol_props_from_smiles(row.smiles)
        if props is None:
            skip += 1
            continue

        # neutral losses
        nl = detect_neutral_losses(spec, float(row.precursor_mz))

        # pack spectrum
        buf = np.zeros((MAX_PEAKS, 2), dtype=np.float32)
        k = min(len(spec), MAX_PEAKS)
        buf[:k] = spec[:k]

        rows.append({
            'fold':        row.fold,
            'precursor_mz': float(row.precursor_mz),
            'spec_bytes':  buf.tobytes(),
            'attn_len':    np.int16(k),
            # task1: element counts
            **{f'cnt_{e}': np.float32(counts.get(e, 0)) for e in COUNT_ELEMENTS},
            # task2: adduct
            'adduct_class': np.int8(adduct_cls),
            # task3: neutral losses
            **{f'nl_{n}': np.int8(nl[n]) for n in NEUTRAL_LOSSES},
            # task4: mol props
            **{f'prop_{k2}': np.float32(v) for k2, v in props.items()},
        })

    print(f'Skipped {skip:,}  Valid {len(rows):,}  ({time.time()-t0:.1f}s)')
    out_df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f'Saved -> {args.out}')
    _print_stats(args.out)


def _print_stats(path: str):
    df = pd.read_parquet(path)
    print(f'\nTotal: {len(df):,} rows')
    for fold in ('train', 'val', 'test'):
        sub = df[df['fold'] == fold]
        if len(sub) == 0:
            continue
        print(f'\n[{fold}] N={len(sub):,}')
        # task1
        for e in COUNT_ELEMENTS:
            col = f'cnt_{e}'
            if col in sub:
                print(f'  cnt_{e}: mean={sub[col].mean():.2f}  max={sub[col].max():.0f}')
        # task2
        if 'adduct_class' in sub:
            vc = sub['adduct_class'].value_counts()
            for cls, name in [(0, '[M+H]+'), (1, '[M+Na]+')]:
                n = vc.get(cls, 0)
                print(f'  adduct {name}: {n:,} ({100*n/len(sub):.1f}%)')
        # task3
        for nl in NEUTRAL_LOSSES:
            col = f'nl_{nl}'
            if col in sub:
                pos = int(sub[col].sum())
                print(f'  nl_{nl}: {pos:,} pos ({100*pos/len(sub):.1f}%)')
        # task4
        for prop in ['MolWt', 'TPSA', 'LogP', 'nHDonors', 'nHAcceptors', 'RingCount']:
            col = f'prop_{prop}'
            if col in sub:
                print(f'  {prop}: mean={sub[col].mean():.1f}  std={sub[col].std():.1f}')


if __name__ == '__main__':
    main()
