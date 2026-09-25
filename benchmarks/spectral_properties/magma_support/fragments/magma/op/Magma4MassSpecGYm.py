#!/usr/bin/env python
"""
使用 v8 优化版本的 MAGMa 算法

v8 优化：分数延迟计算
- 碎片生成时不计算分数
- 只对匹配到的碎片计算分数
- 大幅减少 score_fragment 调用次数

支持 MassSpecGym 数据集格式：
- smiles: SMILES 字符串
- prec_type: 加离子类型 (如 [M+H]+, [M+Na]+)
- peaks: 质谱峰数据，格式为 Python 元组列表的字符串表示，例如：
  "[('83.0491', '0.162'), ('123.0417', '0.047'), ...]"
  其中第一个元素是 m/z 值，第二个元素是相对强度
"""

import argparse
import time
import ast
import json
import re
import os
import tempfile
from functools import wraps, partial
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from fragments.magma.op import fragmentation_op_v8 as v8
from fragments.magma.op.fragmentation_op_v8 import FragProfiler
from fragments.magma import common


DEFAULT_PPM = 10.0
DEFAULT_MAX_TREE_DEPTH = 3
DEFAULT_MAX_BROKEN_BONDS = 6


def _warmup_numba():
    """预热 numba JIT 函数，避免多进程 fork 后同时编译导致缓存竞争/死锁"""
    dummy_bonds = np.array([3], dtype=np.int64)
    dummy_scores = np.array([1], dtype=np.int64)
    v8._score_fragment_numba(3, dummy_bonds, dummy_scores)
    dummy_bonded = np.zeros((2, v8.MAX_ATOM_BONDS), dtype=np.int64)
    dummy_bonded[0, 0] = 1
    dummy_bonded[1, 0] = 0
    dummy_num = np.array([1, 1], dtype=np.int64)
    v8._extend_numba(0, dummy_bonded, dummy_num, 3, 2)


FUNCTION_TIMING_ENABLED = False
FUNCTION_TIMING_VERBOSE = False
FUNCTION_TIMINGS: Dict[str, Dict[str, float]] = defaultdict(lambda: {'total': 0.0, 'count': 0})
_PEAK_RE = re.compile(r"\('([^']+)',\s*'([^']+)'\)")


def timed(name: Optional[str] = None):
    """函数计时修饰器：记录总耗时与调用次数，并可选实时打印。"""
    def decorator(func):
        timer_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not FUNCTION_TIMING_ENABLED:
                return func(*args, **kwargs)
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - t0
                FUNCTION_TIMINGS[timer_name]['total'] += elapsed
                FUNCTION_TIMINGS[timer_name]['count'] += 1
                if FUNCTION_TIMING_VERBOSE:
                    print(f"[FUNC_TIME] {timer_name}: {elapsed*1000:.3f} ms")

        return wrapper

    return decorator


def print_function_timing_report():
    """打印函数耗时汇总（按总耗时降序）。"""
    if not FUNCTION_TIMING_ENABLED:
        return
    print("\n" + "=" * 70)
    print("Function Timing Report")
    print("=" * 70)
    total = sum(item['total'] for item in FUNCTION_TIMINGS.values())
    for func_name, item in sorted(FUNCTION_TIMINGS.items(), key=lambda kv: -kv[1]['total']):
        t = item['total']
        c = int(item['count'])
        avg = (t / c * 1000) if c > 0 else 0.0
        pct = (t / total * 100) if total > 0 else 0.0
        print(f"{func_name:30s}: {t*1000:10.2f}ms ({pct:5.1f}%) | {c:6d}x | avg {avg:8.3f}ms")
    print("-" * 70)
    print(f"{'Total':30s}: {total*1000:10.2f}ms")


class Profiler:
    """性能分析器：记录各环节耗时"""

    def __init__(self):
        self.timings: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)
        self._start_times: Dict[str, float] = {}

    def start(self, name: str):
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str):
        if name in self._start_times:
            elapsed = time.perf_counter() - self._start_times[name]
            self.timings[name] += elapsed
            self.counts[name] += 1
            del self._start_times[name]
            return elapsed
        return 0.0

    def report(self) -> str:
        """生成性能报告"""
        lines = ["=" * 60, "Profile Report", "=" * 60]
        total = sum(self.timings.values())

        # 按耗时排序
        sorted_items = sorted(self.timings.items(), key=lambda x: -x[1])
        for name, t in sorted_items:
            pct = (t / total * 100) if total > 0 else 0
            cnt = self.counts[name]
            avg = (t / cnt * 1000) if cnt > 0 else 0
            lines.append(f"{name:30s}: {t*1000:8.2f}ms ({pct:5.1f}%) | {cnt:5d}x | avg: {avg:.3f}ms")

        lines.append("-" * 60)
        lines.append(f"{'Total':30s}: {total*1000:8.2f}ms")
        return "\n".join(lines)


@timed()
def generate_fragments(smiles: str, max_tree_depth: int = 3, max_broken_bonds: int = 6):
    """生成碎片"""
    fe = v8.FragmentEngine(
        mol_str=smiles,
        max_tree_depth=max_tree_depth,
        max_broken_bonds=max_broken_bonds
    )
    fe.generate_fragments()
    return fe


@timed()
def match_peaks(
    fe: v8.FragmentEngine,
    observed_mz: np.ndarray,
    observed_intensity: Optional[np.ndarray] = None,
    ppm_threshold: float = 10.0,
    adduct: str = "[M+H]+",
    profiler: Profiler = None,
) -> pd.DataFrame:
    """
    谱峰匹配（排序+二分搜索优化版，分数延迟计算）

    优化策略：
    1. 排序碎片质量 + 二分搜索候选窗口，避免 O(P*F) 全量矩阵
    2. 只对 PPM 最小的候选计算分数
    3. 选择分数最低的作为最佳匹配
    """
    if profiler:
        profiler.start("match.get_frag_masses")
    frag_ids, frag_inds, shift_inds, masses = fe.get_frag_masses()
    if profiler:
        profiler.stop("match.get_frag_masses")

    n_peaks = len(observed_mz)

    if len(masses) == 0:
        return pd.DataFrame([
            {
                'peak_id': i,
                'observed_mz': observed_mz[i],
                'intensity': None if observed_intensity is None or i >= len(observed_intensity) else float(observed_intensity[i]),
                'matched_mass': None, 'ppm_diff': float('inf'),
                'score': None, 'h_shift': None, 'formula': None,
                'frag_id': None, 'n_matches': 0,
            }
            for i in range(n_peaks)
        ])

    # 校正 m/z
    adduct_mass = common.ion2mass.get(adduct, common.ion2mass["[M+H]+"])
    adjusted_mz = observed_mz - adduct_mass

    # 一次排序，后续每个峰用二分搜索 O(log F)
    if profiler:
        profiler.start("match.sort_masses")
    order = np.argsort(masses)
    masses_sorted = masses[order]
    ppm_frac = ppm_threshold * 1e-6
    if profiler:
        profiler.stop("match.sort_masses")

    # 向量化二分搜索：一次为所有峰计算候选窗口
    if profiler:
        profiler.start("match.search_candidates")
    safe_target = np.maximum(adjusted_mz, 1e-30)
    mass_lo_all = safe_target / (1.0 + ppm_frac)
    mass_hi_all = safe_target / (1.0 - ppm_frac)
    lo_all = np.searchsorted(masses_sorted, mass_lo_all, side='left')
    hi_all = np.searchsorted(masses_sorted, mass_hi_all, side='right')
    if profiler:
        profiler.stop("match.search_candidates")

    # 预分配结果数组（比 list-of-dicts 快得多）
    matched_mass_arr = np.full(n_peaks, np.nan)
    ppm_diff_arr = np.full(n_peaks, np.inf)
    score_arr = np.full(n_peaks, np.nan)
    h_shift_arr = np.full(n_peaks, np.nan)
    formula_arr = [None] * n_peaks
    frag_id_arr = [None] * n_peaks
    n_matches_arr = np.zeros(n_peaks, dtype=int)

    # 仅遍历有候选且 target > 0 的峰
    has_candidates = (lo_all < hi_all) & (adjusted_mz > 0)
    candidate_peaks = np.nonzero(has_candidates)[0]

    if profiler:
        profiler.start("match.score_calc")
    for i in candidate_peaks:
        lo, hi = int(lo_all[i]), int(hi_all[i])
        target = adjusted_mz[i]

        cand_idx = order[lo:hi]
        cand_m = masses[cand_idx]
        cand_ppm = np.abs(cand_m - target) / cand_m * 1e6
        valid = cand_ppm <= ppm_threshold
        cand_idx = cand_idx[valid]
        cand_ppm = cand_ppm[valid]

        if len(cand_idx) == 0:
            continue

        min_ppm = cand_ppm.min()
        best_mask = cand_ppm == min_ppm
        best_cand = cand_idx[best_mask]

        if len(best_cand) == 1:
            best_idx = best_cand[0]
            score = fe.get_score(frag_ids[best_idx])
            n_best = 1
        else:
            scores = np.array([fe.get_score(frag_ids[j]) for j in best_cand], dtype=float)
            min_score = scores.min()
            smask = scores == min_score
            best_idx = best_cand[np.argmax(smask)]
            score = min_score
            n_best = int(smask.sum())

        matched_mass_arr[i] = masses[best_idx]
        ppm_diff_arr[i] = min_ppm
        score_arr[i] = score
        h_shift_arr[i] = int(shift_inds[best_idx]) - fe.max_broken_bonds
        formula_arr[i] = fe.get_form(frag_ids[best_idx])
        frag_id_arr[i] = frag_ids[best_idx]
        n_matches_arr[i] = n_best
    if profiler:
        profiler.stop("match.score_calc")

    # 从数组构建 DataFrame（避免 list-of-dicts 开销）
    intensity_col = observed_intensity.astype(float) if observed_intensity is not None else np.full(n_peaks, np.nan)
    result_df = pd.DataFrame({
        'peak_id': np.arange(n_peaks),
        'observed_mz': observed_mz,
        'intensity': intensity_col,
        'matched_mass': matched_mass_arr,
        'ppm_diff': ppm_diff_arr,
        'score': score_arr,
        'h_shift': h_shift_arr,
        'formula': formula_arr,
        'frag_id': frag_id_arr,
        'n_matches': n_matches_arr,
    })
    # NaN matched_mass 自然被 .notna() 正确判读为未匹配
    unmatched = np.isnan(matched_mass_arr)
    if unmatched.any():
        result_df.loc[unmatched, 'matched_mass'] = None
    return result_df


@timed()
def parse_massspecgym_peaks(peaks_str: str) -> Tuple[np.ndarray, np.ndarray]:
    """解析 MassSpecGym 的 peaks 字段（正则加速，避免 ast.literal_eval）。"""
    matches = _PEAK_RE.findall(peaks_str)
    if not matches:
        peaks_list = ast.literal_eval(peaks_str)
        matches = [(str(p[0]), str(p[1])) for p in peaks_list]
    mzs = np.array([float(m) for m, _ in matches], dtype=float)
    intensities = np.array([float(v) for _, v in matches], dtype=float)
    return mzs, intensities


@timed()
def get_fragment_atom_indices(fe: v8.FragmentEngine, frag_bits: int) -> List[int]:
    """将位图形式的 fragment 转换为原分子原子索引列表（优化版）。"""
    atom_indices = []
    bits = frag_bits
    while bits:
        lsb = bits & -bits
        atom_idx = lsb.bit_length() - 1
        if atom_idx < fe.natoms:
            atom_indices.append(atom_idx)
        bits ^= lsb
    return atom_indices


@timed()
def get_fragment_cleaved_bonds(fe: v8.FragmentEngine, frag_bits: int) -> Tuple[List[int], List[List[int]]]:
    """计算形成该碎片时相对原分子被切断的键（优化版，返回元组）。"""
    cut_bond_indices = []
    cut_bond_atoms = []

    for bond in fe.mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        in_frag_a1 = bool(frag_bits & (1 << a1))
        in_frag_a2 = bool(frag_bits & (1 << a2))
        if in_frag_a1 != in_frag_a2:
            cut_bond_indices.append(bond.GetIdx())
            cut_bond_atoms.append([a1, a2])

    return cut_bond_indices, cut_bond_atoms


@timed()
def build_fragment_lineage(fe: v8.FragmentEngine, frag_key: int, lineage_cache: Dict[int, List[int]]) -> List[int]:
    """沿第一条父链回溯到根碎片，返回 [root, ..., frag_key]（带缓存优化）。"""
    if frag_key in lineage_cache:
        return lineage_cache[frag_key]

    lineage = [frag_key]
    visited = {frag_key}
    current = frag_key

    while True:
        parent_hashes = fe.frag_to_entry[current].get('parent_hashes', [])
        if not parent_hashes:
            break
        parent = parent_hashes[0]
        if parent in visited:
            break
        lineage.append(parent)
        visited.add(parent)
        current = parent

    lineage.reverse()
    lineage_cache[frag_key] = lineage
    return lineage


@timed()
def build_annotated_data(
    fe: v8.FragmentEngine,
    results: pd.DataFrame,
    smiles: str,
    adduct: str,
    ppm_threshold: float,
    spec_id: str = "",
) -> Dict[str, Any]:
    """构建 MAGMA 标注中间数据，包含碎片映射、断键和层级关系（优化版）。"""
    root_frag = fe.get_root_frag()

    # 预计算键端点，避免每个碎片重复读取 RDKit bond 属性
    bond_endpoints = [
        (bond.GetIdx(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in fe.mol.GetBonds()
    ]

    # 预计算所有碎片的元数据（避免重复计算）
    frag_meta_cache: Dict[int, Dict[str, Any]] = {}
    for frag_key, entry in fe.frag_to_entry.items():
        frag_bits = entry['frag']
        atom_indices = get_fragment_atom_indices(fe, frag_bits)

        bond_indices = []
        bond_atom_pairs = []
        for bond_idx, a1, a2 in bond_endpoints:
            in_a1 = bool(frag_bits & (1 << a1))
            in_a2 = bool(frag_bits & (1 << a2))
            if in_a1 != in_a2:
                bond_indices.append(bond_idx)
                bond_atom_pairs.append([a1, a2])
        parent_ids = sorted({int(p) for p in entry.get('parent_hashes', [])})

        frag_meta_cache[frag_key] = {
            'frag_id': int(frag_key),
            'atom_indices': atom_indices,
            'bond_indices': bond_indices,
            'bond_atom_pairs': bond_atom_pairs,
            'tree_depth': entry.get('tree_depth', 0),
            'parent_ids': parent_ids,
            'formula': fe.get_form(frag_key),
            'is_root': int(frag_key) == root_frag,
        }

    # 层级关系缓存
    lineage_cache: Dict[int, List[int]] = {}

    annotations = []
    peak_annotations = []
    autoregressive_targets = []

    # itertuples 比 iterrows 快数十倍
    matched_rows = results[results['matched_mass'].notna()]

    for row in matched_rows.itertuples(index=False):
        _inten = row.intensity
        peak_record = {
            'peak_id': int(row.peak_id),
            'mz': float(row.observed_mz),
            'intensity': None if _inten is None or pd.isna(_inten) else float(_inten),
        }

        frag_id = int(row.frag_id)
        meta = frag_meta_cache[frag_id]
        lineage = build_fragment_lineage(fe, frag_id, lineage_cache)

        # 构建步进对
        step_pairs = [[lineage[i], lineage[i + 1]] for i in range(len(lineage) - 1)]

        _matched_mass = float(row.matched_mass)
        _ppm_diff = float(row.ppm_diff)
        _score = float(row.score)
        _h_shift = int(row.h_shift)

        # Legacy 格式
        legacy_item = {
            'peak': peak_record,
            'matched': True,
            'fragment_mapping': {
                'frag_id': meta['frag_id'],
                'matched_mass': _matched_mass,
                'ppm_diff': _ppm_diff,
                'score': _score,
                'formula': meta['formula'],
                'h_shift': _h_shift,
                'atom_indices': meta['atom_indices'],
            },
            'bond_cleavage_records': {
                'bond_indices': meta['bond_indices'],
                'bond_atom_pairs': meta['bond_atom_pairs'],
                'n_bonds_cut': len(meta['bond_indices']),
            },
            'hierarchical_relationship': {
                'tree_depth': meta['tree_depth'],
                'parent_frag_ids': meta['parent_ids'],
                'lineage_to_root': lineage,
                'is_root': meta['is_root'],
            },
        }
        annotations.append(legacy_item)

        # 新格式
        peak_annotations.append({
            'peak_id': peak_record['peak_id'],
            'mz': peak_record['mz'],
            'intensity': peak_record['intensity'],
            'matched': True,
            'matched_fragment': {
                'fragment_id': meta['frag_id'],
                'atom_indices': meta['atom_indices'],
                'bond_indices_cut': meta['bond_indices'],
                'bond_atom_pairs_cut': meta['bond_atom_pairs'],
                'formula': meta['formula'],
                'matched_mass': _matched_mass,
                'ppm_diff': _ppm_diff,
                'score': _score,
                'h_shift': _h_shift,
                'tree_depth': meta['tree_depth'],
                'parent_fragment_ids': meta['parent_ids'],
                'lineage_root_to_fragment': lineage,
            },
        })

        # 自回归目标
        autoregressive_targets.append({
            'peak_id': peak_record['peak_id'],
            'target_fragment_id': meta['frag_id'],
            'path_root_to_fragment': lineage,
            'next_step_pairs': step_pairs,
        })

    # 处理未匹配的峰
    unmatched_rows = results[results['matched_mass'].isna()]
    for row in unmatched_rows.itertuples(index=False):
        _inten = row.intensity
        peak_record = {
            'peak_id': int(row.peak_id),
            'mz': float(row.observed_mz),
            'intensity': None if _inten is None or pd.isna(_inten) else float(_inten),
        }

        annotations.append({
            'peak': peak_record,
            'matched': False,
            'fragment_mapping': None,
            'bond_cleavage_records': None,
            'hierarchical_relationship': None,
        })

        peak_annotations.append({
            'peak_id': peak_record['peak_id'],
            'mz': peak_record['mz'],
            'intensity': peak_record['intensity'],
            'matched': False,
            'matched_fragment': None,
        })

    # 构建碎片图（使用缓存）
    fragment_nodes = list(frag_meta_cache.values())
    edge_set = set()
    for frag_id, meta in frag_meta_cache.items():
        for p in meta['parent_ids']:
            edge_set.add((p, frag_id))

    fragment_edges = [
        {'parent_fragment_id': parent, 'child_fragment_id': child}
        for parent, child in sorted(edge_set)
    ]

    return {
        'schema_version': 'magma_annotation_v1',
        'stage': 'magma_annotation',
        'spec_id': spec_id,
        'smiles': smiles,
        'adduct': adduct,
        'ppm_threshold': float(ppm_threshold),
        'n_fragments_generated': int(len(fe.frag_to_entry)),
        'n_peaks': int(len(results)),
        'n_matched': int(len(matched_rows)),
        'peak_annotations': peak_annotations,
        'fragment_graph': {
            'root_fragment_id': int(root_frag),
            'nodes': fragment_nodes,
            'edges': fragment_edges,
        },
        'autoregressive_targets': autoregressive_targets,
        'annotations': annotations,
    }


@timed()
def write_annotated_json(path: str, payload: Any):
    """写出 JSON 文件（无缩进，大幅减少写入耗时和文件体积）。"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))


@timed()
def run_magma(
    smiles: str,
    peaks: np.ndarray,
    peak_intensities: Optional[np.ndarray] = None,
    ppm_threshold: float = 10.0,
    adduct: str = "[M+H]+",
    max_tree_depth: int = 3,
    max_broken_bonds: int = 6,
    verbose: bool = True,
    profile: bool = False,
    skip_canonicalization: bool = False,
) -> Tuple[v8.FragmentEngine, pd.DataFrame]:
    """运行完整的MAGMa算法"""
    profiler = Profiler() if profile else None
    frag_profiler = FragProfiler() if profile else None

    if verbose:
        print(f"分子: {smiles}")
        print(f"谱峰数: {len(peaks)}")
        print(f"参数: PPM={ppm_threshold}, adduct={adduct}")
        print("-" * 50)

    # Step 1: 生成碎片（不计算分数）
    if profiler:
        profiler.start("frag.init")
    fe = v8.FragmentEngine(
        mol_str=smiles,
        max_tree_depth=max_tree_depth,
        max_broken_bonds=max_broken_bonds,
        skip_canonicalization=skip_canonicalization,
        profiler=frag_profiler
    )
    if profiler:
        profiler.stop("frag.init")
        profiler.start("frag.generate")
    fe.generate_fragments()
    if profiler:
        profiler.stop("frag.generate")

    if verbose:
        t_frag = (profiler.timings.get("frag.init", 0) + profiler.timings.get("frag.generate", 0)) if profiler else 0
        print(f"碎片生成: {len(fe.frag_to_entry)} 碎片, {t_frag*1000:.1f}ms")

    # Step 2: 谱峰匹配（分数延迟计算，内部排序+二分搜索）
    results = match_peaks(fe, peaks, peak_intensities, ppm_threshold, adduct, profiler)

    matched = results['matched_mass'].notna().sum()

    if verbose:
        score_computed = len(fe._score_cache)
        t_match = sum(v for k, v in profiler.timings.items() if k.startswith("match.")) if profiler else 0
        print(f"谱峰匹配: {matched}/{len(peaks)} 匹配, {t_match*1000:.1f}ms")
        print(f"分数计算: {score_computed}/{len(fe.frag_to_entry)} 碎片 ({score_computed/len(fe.frag_to_entry)*100:.1f}%)")

    if profile and profiler:
        print("\n" + profiler.report())
        if frag_profiler:
            print("\n" + frag_profiler.report())

    return fe, results


@timed()
def process_csv_row(
    row: Any,
    ppm_threshold: float = 10.0,
    verbose: bool = False,
    profile: bool = False,
    build_annotation: bool = True,
    skip_canonicalization: bool = False,
    output_file: Optional[str] = None,
):
    """处理CSV中的一行数据。支持直接写入文件避免内存累积。"""
    def row_get(key: str, default=None):
        if isinstance(row, dict):
            return row.get(key, default)
        if hasattr(row, 'get'):
            return row.get(key, default)
        return getattr(row, key, default)

    smiles = row_get('smiles')
    peaks_str = row_get('peaks')
    adduct = row_get('prec_type', '[M+H]+')
    n_peaks = 0

    try:
        if not peaks_str or (isinstance(peaks_str, float) and np.isnan(peaks_str)):
            raise ValueError("peaks is empty or NaN")
        mzs, intensities = parse_massspecgym_peaks(str(peaks_str))
        n_peaks = len(mzs)
        fe, results = run_magma(
            smiles=smiles,
            peaks=mzs,
            peak_intensities=intensities,
            ppm_threshold=ppm_threshold,
            adduct=adduct,
            verbose=verbose,
            profile=profile,
            skip_canonicalization=skip_canonicalization,
        )
        matched = results['matched_mass'].notna().sum()

        annotation_json = None
        if build_annotation and output_file:
            annotation = build_annotated_data(
                fe=fe,
                results=results,
                smiles=smiles,
                adduct=adduct,
                ppm_threshold=ppm_threshold,
                spec_id=str(row_get('spec_id', '')),
            )
            annotation_json = json.dumps(annotation, ensure_ascii=False, separators=(',', ':'))

        return {
            'spec_id': row_get('spec_id', ''),
            'smiles': smiles,
            'n_peaks': n_peaks,
            'n_matched': matched,
            'match_rate': matched / n_peaks * 100 if n_peaks > 0 else 0,
            'n_frags': len(fe.frag_to_entry),
            'n_scores': len(fe._score_cache),
            'success': True,
            'annotation_json': annotation_json,
        }
    except Exception as e:
        return {
            'spec_id': row_get('spec_id', ''),
            'smiles': smiles,
            'n_peaks': n_peaks,
            'n_matched': 0,
            'match_rate': 0,
            'n_frags': 0,
            'n_scores': 0,
            'success': False,
            'error': str(e),
            'annotation_json': None,
        }


@timed()
def run_csv(
    csv_path: str,
    ppm_threshold: float = 10.0,
    n_samples: int = None,
    profile: bool = False,
    annotated_output: Optional[str] = None,
    n_workers: int = 1,
    skip_canonicalization: bool = False,
):
    """处理CSV文件中的所有样本（支持多进程加速，优化内存使用）"""
    from tqdm import tqdm

    df = pd.read_csv(csv_path)
    if n_samples:
        df = df.head(n_samples)

    n_total = len(df)
    need_annotation = bool(annotated_output)
    print(f"处理 {n_total} 个样本 (workers={n_workers}, annotation={'ON' if need_annotation else 'OFF'}, canon={'OFF' if skip_canonicalization else 'ON'})...")
    print("=" * 70)

    wall_t0 = time.perf_counter()
    results = []

    if need_annotation:
        os.makedirs(os.path.dirname(annotated_output), exist_ok=True)
        annotation_file = open(annotated_output, 'w', encoding='utf-8')
        annotation_file.write('{"stage":"magma_annotation","samples":[\n')
    else:
        annotation_file = None

    n_annotated = 0

    if n_workers > 1:
        _warmup_numba()

        worker = partial(
            process_csv_row,
            ppm_threshold=ppm_threshold,
            build_annotation=need_annotation,
            skip_canonicalization=skip_canonicalization,
            output_file=annotated_output,
        )

        chunksize = max(1, n_total // (n_workers * 8))

        with ProcessPoolExecutor(
            max_workers=n_workers,
            max_tasks_per_child=50
        ) as executor:
            iterator = executor.map(
                worker,
                (row for _, row in df.iterrows()),
                chunksize=chunksize
            )

            for r in tqdm(iterator, total=n_total, desc="MAGMa v8", mininterval=1.0):
                if annotation_file is not None and r.get('annotation_json'):
                    if n_annotated > 0:
                        annotation_file.write(',\n')
                    annotation_file.write(r['annotation_json'])
                    n_annotated += 1

                summary = {k: v for k, v in r.items() if k != 'annotation_json'}
                results.append(summary)
    else:
        for idx, row in tqdm(df.iterrows(), total=n_total, desc="MAGMa v8", mininterval=1.0):
            t0 = time.perf_counter()
            r = process_csv_row(
                row,
                ppm_threshold,
                verbose=False,
                profile=False,
                build_annotation=need_annotation,
                skip_canonicalization=skip_canonicalization,
                output_file=annotated_output,
            )

            if annotation_file is not None and r.get('annotation_json'):
                if n_annotated > 0:
                    annotation_file.write(',\n')
                annotation_file.write(r['annotation_json'])
                n_annotated += 1

            summary = {k: v for k, v in r.items() if k != 'annotation_json'}
            summary['time'] = time.perf_counter() - t0
            results.append(summary)

    if annotation_file is not None:
        annotation_file.write(f'\n],"n_annotated":{n_annotated}}}')
        annotation_file.close()
        print(f"Annotated data saved to: {annotated_output}")

    total_time = time.perf_counter() - wall_t0
    results_df = pd.DataFrame(results)

    success = results_df[results_df['success'] == True]
    total_frags = success['n_frags'].sum()
    total_scores = success['n_scores'].sum()

    print("\n" + "=" * 70)
    print("Statistics")
    print("=" * 70)
    print(f"Success: {len(success)}/{len(results_df)}")
    print(f"Avg match rate: {success['match_rate'].mean():.1f}%")
    print(f"Avg frags: {success['n_frags'].mean():.0f}")
    if total_frags > 0:
        print(f"Score calc saved: {total_frags - total_scores:,}/{total_frags:,} ({(1 - total_scores/total_frags)*100:.1f}%)")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg time: {total_time/n_total*1000:.1f}ms/sample")
    print(f"Throughput: {n_total/total_time:.1f} samples/s")

    if profile and n_total > 0:
        print("\n" + "=" * 70)
        print("Detailed Profile (first 5 samples)")
        print("=" * 70)
        for idx, row in df.head(min(5, len(df))).iterrows():
            smiles = row['smiles']
            peaks_str = row['peaks']
            try:
                mzs, _ = parse_massspecgym_peaks(peaks_str)
            except Exception as e:
                print(f"Warning: Failed to parse peaks for sample {idx}: {e}")
                continue
            adduct = row.get('prec_type', '[M+H]+')
            try:
                run_magma(
                    smiles=smiles,
                    peaks=mzs,
                    peak_intensities=None,
                    ppm_threshold=ppm_threshold,
                    adduct=adduct,
                    verbose=False,
                    profile=True,
                )
            except Exception as e:
                print(f"Warning: Profile failed for sample {idx}: {e}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description='MAGMa v8 - Score deferred, binary search, multiprocess')
    parser.add_argument('--smiles', type=str, help='Molecule SMILES')
    parser.add_argument('--peaks', type=str, help='Comma-separated m/z values')
    parser.add_argument('--peaks-file', type=str, help='Peaks file')
    parser.add_argument('--csv', type=str, help='CSV file')
    parser.add_argument('-n', '--num', type=int, help='Process first n samples')
    parser.add_argument('--ppm', type=float, default=DEFAULT_PPM, help='PPM threshold')
    parser.add_argument('--adduct', type=str, default='[M+H]+', help='Adduct type')
    parser.add_argument('--output', '-o', type=str, help='Output file')
    parser.add_argument('--annotated-output', type=str, help='Output JSON for MAGMA annotation stage')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    parser.add_argument('--profile', '-p', action='store_true', help='Enable profiling')
    parser.add_argument('--func-timing', action='store_true', help='Enable decorator-based function timing report')
    parser.add_argument('--func-timing-verbose', action='store_true', help='Print timing for every function call')
    parser.add_argument('--workers', '-w', type=int, default=1, help='Number of parallel workers for CSV mode (default: 1)')
    parser.add_argument('--no-canonicalize', action='store_true',
                        help='Skip tautomer canonicalization (much faster, safe for canonical SMILES input)')
    args = parser.parse_args()

    global FUNCTION_TIMING_ENABLED, FUNCTION_TIMING_VERBOSE
    FUNCTION_TIMING_ENABLED = bool(args.func_timing or args.func_timing_verbose)
    FUNCTION_TIMING_VERBOSE = bool(args.func_timing_verbose)

    if args.csv:
        results_df = run_csv(
            args.csv,
            args.ppm,
            args.num,
            profile=args.profile,
            annotated_output=args.annotated_output,
            n_workers=args.workers,
            skip_canonicalization=args.no_canonicalize,
        )
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            results_df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
        print_function_timing_report()
        return results_df

    if not args.smiles:
        parser.error("Need --smiles or --csv argument")

    if args.peaks:
        peaks = np.array([float(x.strip()) for x in args.peaks.split(',')])
        peak_intensities = None
    elif args.peaks_file:
        with open(args.peaks_file) as f:
            peaks = np.array([float(line.strip()) for line in f if line.strip()])
        peak_intensities = None
    else:
        peaks = np.array([43.0, 77.0, 93.0, 121.0, 138.0, 163.0, 180.0])
        peak_intensities = None
        if not args.quiet:
            print("Using default example peaks")

    fe, results = run_magma(
        smiles=args.smiles,
        peaks=peaks,
        peak_intensities=peak_intensities,
        ppm_threshold=args.ppm,
        adduct=args.adduct,
        verbose=not args.quiet,
        profile=args.profile,
        skip_canonicalization=args.no_canonicalize,
    )

    if not args.quiet:
        print("\n" + "=" * 70)
        print("Match results:")
        print("=" * 70)
        print(results.to_string(index=False))

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        results.to_csv(args.output, index=False)
        if not args.quiet:
            print(f"\nResults saved to: {args.output}")

    if args.annotated_output:
        os.makedirs(os.path.dirname(args.annotated_output), exist_ok=True)
        annotated = build_annotated_data(
            fe=fe,
            results=results,
            smiles=args.smiles,
            adduct=args.adduct,
            ppm_threshold=args.ppm,
            spec_id='',
        )
        write_annotated_json(args.annotated_output, annotated)
        if not args.quiet:
            print(f"Annotated data saved to: {args.annotated_output}")

    print_function_timing_report()
    return results


if __name__ == '__main__':
    main()
