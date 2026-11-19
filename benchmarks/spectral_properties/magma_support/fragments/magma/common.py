"""Shared MAGMa dependencies.

Import the functions required by MAGMa from chem_utils and misc_utils.
"""

# Imports from chem_utils
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

# Imports from misc_utils
from .misc_utils import (
    parse_spectra,
    process_spec_file,
    max_inten_spec,
)
# Optional HDF5Dataset import
try:
    from .misc_utils import HDF5Dataset
except ImportError:
    HDF5Dataset = None

# Optional imports from parallel_utils
try:
    from .parallel_utils import (
        chunked_parallel,
        batches_num_chunks,
    )
except ImportError:
    chunked_parallel = None
    batches_num_chunks = None
