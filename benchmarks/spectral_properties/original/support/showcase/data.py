"""
从 parquet 加载各任务数据。

task_name → (task_type, label_cols, metric, parquet_path)
  'formula_count' : regression,   cnt_{C,H,N,O,S,Cl,F,Br}
  'adduct'        : binary,        adduct_class  ([M+H]+ vs [M+Na]+)
  'neutral_loss'  : multilabel,    nl_{H2O,NH3,CO2,HF,HCl,CO,CH3}
  'mol_props'     : regression,    prop_{MolWt,TPSA,LogP,nHDonors,nHAcceptors,RingCount}
  'adduct_multi'  : multiclass(4), adduct_class  ([M+H]+ / [M+Na]+ / [M+NH4]+ / [M-H]-)
  'dreams_props'  : regression,    dp_{BertzComplexity,FractionCSP3,NumAromaticRings,...}
"""
from __future__ import annotations
import os, sys
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAX_PEAKS = 150
PARQUET_PATH         = os.environ.get('ULTRAMS_MASSGYM_PARQUET', 'datasets/MassSpecGym/massgym_showcase.parquet')
ADDUCT_MULTI_PATH    = os.environ.get('ULTRAMS_ADDUCT_MULTI_PARQUET', 'datasets/adduct_multi_showcase.parquet')
DREAMS_PROPS_PATH    = os.environ.get('ULTRAMS_DREAMS_PROPS_PARQUET', 'datasets/MassSpecGym/massgym_dreams_props.parquet')

TASKS = {
    'formula_count': {
        'type': 'regression',
        'cols': ['cnt_C', 'cnt_H', 'cnt_N', 'cnt_O', 'cnt_S', 'cnt_Cl', 'cnt_F', 'cnt_Br'],
        'names': ['C', 'H', 'N', 'O', 'S', 'Cl', 'F', 'Br'],
        'parquet': PARQUET_PATH,
    },
    'adduct': {
        'type': 'binary',
        'cols': ['adduct_class'],
        'names': ['[M+Na]+'],   # 1=Na, 0=H
        'parquet': PARQUET_PATH,
    },
    'neutral_loss': {
        'type': 'multilabel',
        'cols': ['nl_H2O', 'nl_NH3', 'nl_CO2', 'nl_HF', 'nl_HCl', 'nl_CO', 'nl_CH3'],
        'names': ['H2O', 'NH3', 'CO2', 'HF', 'HCl', 'CO', 'CH3'],
        'parquet': PARQUET_PATH,
    },
    'mol_props': {
        'type': 'regression',
        'cols': ['prop_MolWt', 'prop_TPSA', 'prop_LogP',
                 'prop_nHDonors', 'prop_nHAcceptors', 'prop_RingCount'],
        'names': ['MolWt', 'TPSA', 'LogP', 'nHDonors', 'nHAcceptors', 'RingCount'],
        'parquet': PARQUET_PATH,
    },
    # ── 新任务 ─────────────────────────────────────────────────────────────
    'adduct_multi': {
        'type': 'multiclass',
        'cols': ['adduct_class'],
        'names': ['[M+H]+', '[M+Na]+', '[M+NH4]+', '[M-H]-'],
        'n_classes': 4,
        'parquet': ADDUCT_MULTI_PATH,
    },
    'dreams_props': {
        'type': 'regression',
        'cols': [
            'dp_BertzComplexity', 'dp_FractionCSP3',
            'dp_NumAromaticRings', 'dp_NumAliphaticRings',
            'dp_NumRotatableBonds', 'dp_QED',
            'dp_AtomicLogP', 'dp_PolarSurfaceArea',
            'dp_NumHAcceptors', 'dp_NumHDonors',
        ],
        'names': [
            'BertzComplexity', 'FractionCSP3',
            'NumAromaticRings', 'NumAliphaticRings',
            'NumRotatableBonds', 'QED',
            'AtomicLogP', 'PolarSurfaceArea',
            'NumHAcceptors', 'NumHDonors',
        ],
        'parquet': DREAMS_PROPS_PATH,
    },
}


def load_fold(
    parquet_path: str,
    fold: str,
    task_name: str,
    max_rows: Optional[int] = None,
) -> Tuple[List[dict], np.ndarray, dict]:
    """
    返回 (samples, labels, task_meta)
      samples   : list of {'spectrum': (M,2), 'precursor_mz': float}
      labels    : ndarray (N, K)  float32/int64
      task_meta : TASKS[task_name] dict

    注: parquet_path 参数保留向后兼容，但优先使用 task 定义中的 parquet 路径。
    """
    task = TASKS[task_name]
    # 优先使用任务自带的 parquet 路径
    pq_path = task.get('parquet', parquet_path)
    need_cols = ['fold', 'precursor_mz', 'spec_bytes', 'attn_len'] + task['cols']
    df = pd.read_parquet(pq_path, columns=need_cols)
    df = df[df['fold'] == fold].reset_index(drop=True)
    if max_rows:
        df = df.iloc[:max_rows]

    samples: List[dict] = []
    spec_bytes_arr = df['spec_bytes'].values
    pmz_arr        = df['precursor_mz'].values
    attn_arr       = df['attn_len'].values
    label_arr      = df[task['cols']].values

    if task['type'] == 'regression':
        label_arr = label_arr.astype(np.float32)
    elif task['type'] == 'multiclass':
        # 多分类：(N, 1) int64，训练时会 squeeze 成 (N,)
        label_arr = label_arr.astype(np.int64)
    else:
        label_arr = label_arr.astype(np.int64)

    for i in range(len(df)):
        padded = np.frombuffer(spec_bytes_arr[i], dtype=np.float32).reshape(MAX_PEAKS, 2).copy()
        k = int(attn_arr[i])
        samples.append({'spectrum': padded[:k], 'precursor_mz': float(pmz_arr[i])})

    return samples, label_arr, task


def print_stats(labels: np.ndarray, task: dict, split: str):
    N = len(labels)
    print(f'[{split}] N={N:,}')
    if task['type'] == 'regression':
        for i, name in enumerate(task['names']):
            col = labels[:, i]
            print(f'  {name}: mean={col.mean():.2f}  std={col.std():.2f}  max={col.max():.0f}')
    elif task['type'] == 'multiclass':
        flat = labels.squeeze(-1) if labels.ndim == 2 else labels
        for i, name in enumerate(task['names']):
            cnt = int((flat == i).sum())
            print(f'  {name}: {cnt:,} ({100*cnt/N:.1f}%)')
    else:
        for i, name in enumerate(task['names']):
            col = labels[:, i] if labels.ndim > 1 else labels
            pos = int((col == 1).sum())
            print(f'  {name}: {pos:,} pos ({100*pos/N:.1f}%)')
