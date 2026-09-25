"""fragmentation_op_v8.py - 分数延迟计算优化

基于 v7 的进一步优化：
1. score 延迟计算（类似 form），只在匹配时才计算
2. 减少 score_fragment 的调用次数（从所有碎片 -> 只有匹配候选）
3. 保持 v7 的增量质量计算优化
"""

import time
import numpy as np
from numba import njit
from collections import defaultdict
from typing import Tuple, List, Dict, Optional
from rdkit import Chem

from .. import common


class FragProfiler:
    """碎片生成内部性能分析器（低开销版本）"""

    def __init__(self):
        self.timings: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)
        self._stack: List[Tuple[str, float]] = []

    def start(self, name: str):
        self._stack.append((name, time.perf_counter()))

    def stop(self, name: str) -> float:
        if self._stack and self._stack[-1][0] == name:
            _, t0 = self._stack.pop()
            elapsed = time.perf_counter() - t0
            self.timings[name] += elapsed
            self.counts[name] += 1
            return elapsed
        return 0.0

    def add(self, name: str, elapsed: float, count: int = 1):
        """直接添加计时结果（用于批量计时）"""
        self.timings[name] += elapsed
        self.counts[name] += count

    def report(self) -> str:
        lines = ["=" * 65, "FragmentEngine Internal Profile", "=" * 65]
        total = sum(self.timings.values())
        sorted_items = sorted(self.timings.items(), key=lambda x: -x[1])
        for name, t in sorted_items:
            pct = (t / total * 100) if total > 0 else 0
            cnt = self.counts[name]
            avg = (t / cnt * 1000) if cnt > 0 else 0
            lines.append(f"{name:35s}: {t*1000:8.2f}ms ({pct:5.1f}%) | {cnt:6d}x | avg: {avg:.4f}ms")
        lines.append("-" * 65)
        lines.append(f"{'Total':35s}: {total*1000:8.2f}ms")
        return "\n".join(lines)

TYPEW = {
    Chem.rdchem.BondType.names["AROMATIC"]: 2,
    Chem.rdchem.BondType.names["DOUBLE"]: 2,
    Chem.rdchem.BondType.names["TRIPLE"]: 3,
    Chem.rdchem.BondType.names["SINGLE"]: 1,
}
MAX_ATOM_BONDS = 6
HETEROW = {False: 2, True: 1}


@njit(cache=True)
def _score_fragment_numba(fragment: int, bonds: np.ndarray, bondscores: np.ndarray) -> Tuple[int, int]:
    score = 0
    breaks = 0
    for i in range(len(bonds)):
        bond = bonds[i]
        if (fragment & bond) > 0 and (fragment & bond) < bond:
            score += bondscores[i]
            breaks += 1
    return breaks, score


@njit(cache=True)
def _extend_numba(atom: int, bonded_atoms: np.ndarray, num_bonds: np.ndarray,
                  template_fragment: int, natoms: int) -> int:
    stack = np.zeros(natoms, dtype=np.int64)
    stack_size = 1
    stack[0] = atom
    new_fragment = 0

    while stack_size > 0:
        stack_size -= 1
        current = stack[stack_size]

        for i in range(num_bonds[current]):
            a = bonded_atoms[current, i]
            atombit = 1 << a

            if (not (atombit & template_fragment)) or (atombit & new_fragment):
                continue

            new_fragment = new_fragment | atombit
            stack[stack_size] = a
            stack_size += 1

    return new_fragment


class FragmentEngine:
    """v8: 分数延迟计算优化"""

    def __init__(
        self,
        mol_str: str,
        max_tree_depth: int = 3,
        max_broken_bonds: int = 6,
        mol_str_type: str = "smiles",
        mol_str_canonicalized: bool = False,
        skip_canonicalization: bool = False,
        profiler: Optional[FragProfiler] = None,
    ):
        self._profiler = profiler

        if self._profiler:
            self._profiler.start("init.mol_parse")

        if mol_str_type == "smiles":
            self.smiles = mol_str
            self.mol = Chem.MolFromSmiles(self.smiles)
            if self.mol is None:
                return
            if skip_canonicalization:
                self.inchi = None
            else:
                self.inchi = Chem.MolToInchi(self.mol)
                if not mol_str_canonicalized:
                    self.mol = common.canonical_mol_from_inchi(self.inchi)
                    self.smiles = Chem.MolToSmiles(self.mol)
                    self.mol = Chem.MolFromSmiles(self.smiles)
        elif mol_str_type == "inchi":
            self.inchi = mol_str
            self.mol = common.canonical_mol_from_inchi(self.inchi)
            if self.mol is None:
                return
            self.smiles = Chem.MolToSmiles(self.mol)
            self.mol = Chem.MolFromSmiles(self.smiles)
        else:
            raise NotImplementedError()

        if self.mol is None:
            raise RuntimeError(f"Invalid molecule: {self.smiles}")

        if self._profiler:
            self._profiler.stop("init.mol_parse")
            self._profiler.start("init.kekulize")

        self.natoms = self.mol.GetNumAtoms()
        Chem.Kekulize(self.mol, clearAromaticFlags=True)

        if self._profiler:
            self._profiler.stop("init.kekulize")
            self._profiler.start("init.atom_props")

        self.atom_symbols = [i.GetSymbol() for i in self.mol.GetAtoms()]
        self.atom_symbols_ar = np.array(self.atom_symbols)
        self.atom_hs = np.array(
            [i.GetNumImplicitHs() + i.GetNumExplicitHs() for i in self.mol.GetAtoms()],
            dtype=np.int64
        )
        self.total_hs = int(self.atom_hs.sum())
        self.atom_weights = np.array(
            [common.ELEMENT_TO_MASS[sym] if i.GetIsotope() == 0 else
             common.P_TBL.GetMassForIsotope(i.GetSymbol(), i.GetIsotope())
             for sym, i in zip(self.atom_symbols, self.mol.GetAtoms())]
        )
        self.atom_weights_h = (
            self.atom_hs * common.ELEMENT_TO_MASS["H"] + self.atom_weights
        )
        self.full_weight = np.sum(self.atom_weights_h)

        if self._profiler:
            self._profiler.stop("init.atom_props")
            self._profiler.start("init.bond_info")

        self.bonded_atoms = [[] for _ in self.atom_symbols]
        self.bonded_types = [[] for _ in self.atom_symbols]
        self.bonded_atoms_np = np.zeros((self.natoms, MAX_ATOM_BONDS), dtype=np.int64)
        self.bonded_types_np = np.zeros((self.natoms, MAX_ATOM_BONDS), dtype=np.int64)
        self.num_bonds_np = np.zeros(self.natoms, dtype=np.int64)

        self.bond_to_type = {}
        self.bonds = set()
        self.bonds_list = []
        self.bondscore = {}

        for bond in self.mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            self.bonded_atoms[a1].append(a2)
            self.bonded_atoms[a2].append(a1)

            self.bonded_atoms_np[a1, self.num_bonds_np[a1]] = a2
            self.bonded_atoms_np[a2, self.num_bonds_np[a2]] = a1

            bondbits = 1 << a1 | 1 << a2
            bondscore = (
                TYPEW[bond.GetBondType()]
                * HETEROW[self.atom_symbols[a1] != "C" or self.atom_symbols[a2] != "C"]
            )
            bondtype = TYPEW[bond.GetBondType()]

            self.bonded_types[a1].append(bondtype)
            self.bonded_types[a2].append(bondtype)

            self.bonded_types_np[a1, self.num_bonds_np[a1]] = bondtype
            self.bonded_types_np[a2, self.num_bonds_np[a2]] = bondtype

            self.num_bonds_np[a1] += 1
            self.num_bonds_np[a2] += 1

            self.bond_to_type[bondbits] = bondtype
            self.bondscore[bondbits] = bondscore
            if bondbits not in self.bonds:
                self.bonds_list.append(bondbits)
            self.bonds.add(bondbits)

        self.bonds_np = np.array(self.bonds_list, dtype=np.int64)
        self.bondscores_np = np.array([self.bondscore[b] for b in self.bonds_list], dtype=np.int64)

        if self._profiler:
            self._profiler.stop("init.bond_info")

        self.max_broken_bonds = max_broken_bonds
        self.max_tree_depth = max_tree_depth

        self.shift_buckets = np.arange(self.max_broken_bonds * 2 + 1) - self.max_broken_bonds
        self.shift_bucket_inds = np.arange(self.max_broken_bonds * 2 + 1)
        self.shift_bucket_masses = self.shift_buckets * common.ELEMENT_TO_MASS["H"]

        self.frag_to_entry = {}
        self._score_cache = {}

    def _compute_score(self, fragment: int) -> int:
        """计算碎片分数（带缓存）"""
        if fragment in self._score_cache:
            return self._score_cache[fragment]
        _, score = _score_fragment_numba(fragment, self.bonds_np, self.bondscores_np)
        self._score_cache[fragment] = score
        return score

    def get_score(self, frag_key: int) -> int:
        """获取碎片分数（延迟计算）"""
        entry = self.frag_to_entry[frag_key]
        if entry["score"] is None:
            entry["score"] = self._compute_score(entry["frag"])
        return entry["score"]

    def _compute_form_string(self, frag: int) -> str:
        """延迟计算化学式字符串"""
        form_vec = np.zeros(len(common.VALID_ELEMENTS))
        h_pos = common.element_to_ind["H"]
        for atom in range(self.natoms):
            if frag & (1 << atom):
                dense_pos = common.element_to_ind[self.atom_symbols[atom]]
                form_vec[dense_pos] += 1
                form_vec[h_pos] += self.atom_hs[atom]
        return common.vec_to_formula(form_vec)

    def get_form(self, frag_key: int) -> str:
        """获取碎片的化学式（延迟计算）"""
        entry = self.frag_to_entry[frag_key]
        if entry["form"] is None:
            entry["form"] = self._compute_form_string(entry["frag"])
        return entry["form"]

    def get_frag_masses(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """获取碎片质量（不包含分数，分数延迟计算）- 预分配数组版本"""
        n_frags = len(self.frag_to_entry)
        n_shifts = len(self.shift_buckets)
        max_entries = n_frags * n_shifts

        out_frag_ids = np.empty(max_entries, dtype=np.int64)
        out_frag_inds = np.empty(max_entries, dtype=np.int64)
        out_shift_inds = np.empty(max_entries, dtype=np.int64)
        out_masses = np.empty(max_entries, dtype=np.float64)

        shift_buckets = self.shift_buckets
        shift_bucket_inds = self.shift_bucket_inds
        shift_bucket_masses = self.shift_bucket_masses
        idx = 0

        for k, v in self.frag_to_entry.items():
            max_remove, max_add = v["max_remove_hs"], v["max_add_hs"]
            frag_int = v["frag"]
            base_mass = v["base_mass"]

            for j in range(n_shifts):
                num_shift = shift_buckets[j]
                if (num_shift >= -max_remove) and (num_shift <= max_add):
                    out_frag_ids[idx] = k
                    out_frag_inds[idx] = frag_int
                    out_shift_inds[idx] = shift_bucket_inds[j]
                    out_masses[idx] = base_mass + shift_bucket_masses[j]
                    idx += 1

        return (
            out_frag_ids[:idx],
            out_frag_inds[:idx],
            out_shift_inds[:idx],
            out_masses[:idx],
        )

    def get_root_frag(self) -> int:
        return (1 << self.natoms) - 1

    def generate_fragments(self):
        """生成碎片（不计算分数，延迟到匹配时）"""
        cur_id = 0
        frag = (1 << self.natoms) - 1

        root_mass = float(np.sum(self.atom_weights_h))
        root_frag_hs = int(np.sum(self.atom_hs))

        root = {
            "frag": frag,
            "id": cur_id,
            "sibling_hashes": [],
            "parents": [],
            "parent_hashes": [],
            "parent_ind_removed": [],
            "max_broken": 0,
            "tree_depth": 0,
            "score": None,
            "base_mass": root_mass,
            "frag_hs": root_frag_hs,
            "max_remove_hs": 0,
            "max_add_hs": 0,
            "form": None,
        }

        frag_key = frag
        self.frag_to_entry[frag_key] = root
        current_fragments = [frag_key]
        new_fragments = []

        for step in range(self.max_tree_depth):
            n_frags_in_layer = len(current_fragments)

            # 只在每层开始/结束计时，避免高频计时开销
            if self._profiler:
                self._profiler.start(f"gen.depth_{step}")

            for frag_key in current_fragments:
                parent_entry = self.frag_to_entry[frag_key]
                cur_parent = parent_entry["id"]
                fragment = parent_entry["frag"]
                parent_broken = parent_entry["max_broken"]
                cur_parent_hash = frag_key

                parent_mass = parent_entry["base_mass"]
                parent_frag_hs = parent_entry["frag_hs"]

                for atom in range(self.natoms):
                    extended_fragments = self.remove_atom(fragment, atom)
                    sibling_keys = set([i["new_key"] for i in extended_fragments])

                    for frag_dict in extended_fragments:
                        removed_atom = frag_dict["removed_atom"]
                        new_frag_key = frag_dict["new_key"]
                        rm_bond_t = frag_dict["rm_bond_t"]
                        new_frag = frag_dict["new_frag"]

                        temp_sibs = list(sibling_keys.difference([new_frag_key]))
                        old_entry = self.frag_to_entry.get(new_frag_key)
                        max_broken = parent_broken + rm_bond_t

                        if old_entry is None:
                            cur_id += 1

                            removed_bits = fragment ^ new_frag
                            mass_removed = 0.0
                            hs_removed = 0
                            for a in range(self.natoms):
                                if removed_bits & (1 << a):
                                    mass_removed += self.atom_weights_h[a]
                                    hs_removed += self.atom_hs[a]

                            new_mass = parent_mass - mass_removed
                            new_frag_hs = parent_frag_hs - hs_removed

                            max_remove = int(min(new_frag_hs, self.max_broken_bonds, max_broken))
                            max_add = int(min(self.total_hs - new_frag_hs, self.max_broken_bonds, max_broken))

                            new_entry = {
                                "frag": new_frag,
                                "id": cur_id,
                                "parents": [cur_parent],
                                "parent_hashes": [cur_parent_hash],
                                "parent_ind_removed": [removed_atom],
                                "sibling_hashes": [temp_sibs],
                                "max_broken": max_broken,
                                "tree_depth": step + 1,
                                "score": None,
                                "base_mass": new_mass,
                                "frag_hs": new_frag_hs,
                                "max_remove_hs": max_remove,
                                "max_add_hs": max_add,
                                "form": None,
                            }
                            self.frag_to_entry[new_frag_key] = new_entry
                            new_fragments.append(new_frag_key)
                        else:
                            old_entry["parent_ind_removed"].append(removed_atom)
                            old_entry["parents"].append(cur_parent)
                            old_entry["parent_hashes"].append(cur_parent_hash)
                            old_entry["sibling_hashes"].append(temp_sibs)
                            if max_broken > old_entry["max_broken"]:
                                old_entry["max_broken"] = max_broken
                                frag_hs = old_entry["frag_hs"]
                                old_entry["max_remove_hs"] = int(min(frag_hs, self.max_broken_bonds, max_broken))
                                old_entry["max_add_hs"] = int(min(self.total_hs - frag_hs, self.max_broken_bonds, max_broken))

            if self._profiler:
                self._profiler.stop(f"gen.depth_{step}")
                self._profiler.counts[f"depth{step}_input"] = n_frags_in_layer
                self._profiler.counts[f"depth{step}_output"] = len(new_fragments)

            current_fragments = new_fragments
            new_fragments = []

    def remove_atom(self, fragment: int, atom: int) -> List[dict]:
        if not ((1 << atom) & fragment):
            return []

        template_fragment = fragment ^ (1 << atom)
        list_ext_atoms = []
        extended_fragments = []
        ext_atom_to_bo = {}

        for a in self.bonded_atoms[atom]:
            if (1 << a) & template_fragment:
                list_ext_atoms.append(a)
                bond_num = (1 << atom) | (1 << a)
                ext_atom_to_bo[a] = self.bond_to_type[bond_num]

        if len(list_ext_atoms) == 1:
            if template_fragment == 0:
                return []

            bo = ext_atom_to_bo[list_ext_atoms[0]]
            extended_fragments.append({
                "new_frag": template_fragment,
                "new_key": template_fragment,
                "removed_atom": atom,
                "rm_bond_t": bo,
            })
        else:
            for a in list_ext_atoms:
                is_ring = False
                rm_bond_t = ext_atom_to_bo[a]
                for frag_dict in extended_fragments:
                    if (1 << a) & frag_dict["new_frag"]:
                        is_ring = True
                        break
                if not is_ring:
                    new_fragment = _extend_numba(
                        a, self.bonded_atoms_np, self.num_bonds_np,
                        template_fragment, self.natoms
                    )
                    if new_fragment == 0:
                        continue

                    extended_fragments.append({
                        "new_frag": new_fragment,
                        "new_key": new_fragment,
                        "removed_atom": atom,
                        "rm_bond_t": rm_bond_t,
                    })
        return extended_fragments

    def export_edges(self, frag_keys: List[int]) -> List[Tuple[int, int]]:
        edges = []
        explored = set(frag_keys)
        for i in frag_keys:
            entry = self.frag_to_entry[i]
            for p in entry["parent_hashes"]:
                if p in explored:
                    edges.append((p, i))
        return edges

    def export_edges_dict(self, frag_keys: List[int]) -> Tuple[dict, dict]:
        incoming_edges = defaultdict(list)
        outgoing_edges = defaultdict(list)
        explored = set(frag_keys)
        for i in frag_keys:
            entry = self.frag_to_entry[i]
            for p in entry["parent_hashes"]:
                if p in explored:
                    incoming_edges[i].append(p)
                    outgoing_edges[p].append(i)
        return incoming_edges, outgoing_edges

    def get_present_atoms(self, frag: int):
        ret_inds, ret_symbs = [], []
        for atom in range(self.natoms):
            if (1 << atom) & frag:
                ret_inds.append(atom)
                ret_symbs.append(self.atom_symbols[atom])
        return ret_inds, ret_symbs
