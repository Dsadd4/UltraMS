"""common.py - MAGMa 依赖的通用模块

整合 chem_utils 和 misc_utils 中 MAGMa 所需的函数
"""

# 从 chem_utils 导入
from .chem_utils import (
    P_TBL,
    ELECTRON_MASS,
    VALID_ELEMENTS,
    VALID_MONO_MASSES,
    ELEMENT_VECTORS,
    ELEMENT_TO_MASS,
    element_to_ind,
    ion2mass,
    is_positive_adduct,
    canonical_mol_from_inchi,
    formula_to_dense,
    vec_to_formula,
    mass_from_smi,
    inchi_from_smiles,
    has_valid_elements,
)

# 从 misc_utils 导入
from .misc_utils import (
    parse_spectra,
    process_spec_file,
    max_inten_spec,
)
# HDF5Dataset 可选导入
try:
    from .misc_utils import HDF5Dataset
except ImportError:
    HDF5Dataset = None

# 从 parallel_utils 导入 (可选)
try:
    from .parallel_utils import (
        chunked_parallel,
        batches_num_chunks,
    )
except ImportError:
    chunked_parallel = None
    batches_num_chunks = None
