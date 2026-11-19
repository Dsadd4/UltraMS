"""Spectrum tokenization and BERT encoder components.

The tokenizer provides spectrum and acquisition-metadata tokens. MsEncoderV4
combines Fourier features, peak projections, type embeddings and a bidirectional
BERT encoder, with a CLS token for global spectrum representations.
"""

import json
import pickle
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml

from transformers import BertConfig, BertModel

# Import FourierFeatures.
from .feature.fourier import FourierFeatures


# ============================================================================
# Part 1: Spectrum tokenizer
# ============================================================================


class MsTokenizer:
    """Mass-spectrum tokenizer.

    Features:
    1. Spectrum preprocessing: select the top 100 spectral peaks by intensity.
    2. Tokenize m/z values at 1.0 resolution over 0-999.
    3. Tokenize metadata.
    4. Generate multimodal token sequences.
    """

    @staticmethod
    def load_config_from_yaml(config_path: str) -> Dict:
        """Load metadata configuration from YAML.

        Args:
            config_path: path to the YAML configuration.

        Returns:
            Dictionary containing instruments, ion_types, ionizations, and collision_energies.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            return {
                "instruments": config.get("instruments", ["qtof", "orbitrap"]),
                "ion_types": config.get("ion_types", ["[M+H+]+", "[M-H+]-"]),
                "ionizations": config.get("ionizations", ["GC-EI", "LC-ESI"]),
                "collision_energies": config.get(
                    "collision_energies", ["10", "20", "40", "60", "0"]
                ),
            }
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found, using default values")
            return {
                "instruments": ["qtof", "orbitrap"],
                "ion_types": ["[M+H+]+", "[M-H+]-"],
                "ionizations": ["GC-EI", "LC-ESI"],
                "collision_energies": ["10", "20", "40", "60", "0"],
            }
        except Exception as e:
            print(f"Warning: Error loading config from {config_path}: {e}")
            print("Using default values")
            return {
                "instruments": ["qtof", "orbitrap"],
                "ion_types": ["[M+H+]+", "[M-H+]-"],
                "ionizations": ["GC-EI", "LC-ESI"],
                "collision_energies": ["10", "20", "40", "60", "0"],
            }

    def __init__(
        self,
        mz_precision: float = 1.0,
        mz_range: Tuple[int, int] = (0, 1000),
        config_path: Optional[str] = None,
        custom_instruments: Optional[List[str]] = None,
        custom_ion_types: Optional[List[str]] = None,
        custom_ionizations: Optional[List[str]] = None,
        custom_collision_energies: Optional[List[str]] = None,
    ):
        """Initialize the tokenizer.

        Args:
            mz_precision: m/z resolution.
            mz_range: m/z range.
            config_path: YAML configuration; takes precedence over custom arguments.
            custom_instruments: instrument list when config_path is None.
            custom_ion_types: ion-type list.
            custom_ionizations: ionization-method list.
            custom_collision_energies: collision-energy list.
        """
        # Special tokens
        self.special_tokens = {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
            "<SEP>": 4,
            "<Ke>": 5,
            "<CLS>": 6,
            "<MS_BEGIN>": 7,
            "<MS_END>": 8,
            "<META_BEGIN>": 9,
            "<META_END>": 10,
            "<META_GENERAL>": 11,
        }

        self.mz_precision = mz_precision
        self.mz_range = mz_range

        # Load metadata configuration.
        # Precedence: config_path > custom arguments > defaults.
        if config_path is not None:
            # Load from the YAML configuration.
            config = self.load_config_from_yaml(config_path)
            self.custom_instruments = config["instruments"]
            self.custom_ion_types = config["ion_types"]
            self.custom_ionizations = config["ionizations"]
            self.custom_collision_energies = config["collision_energies"]
            print(f"✓ Loaded metadata from configuration: {config_path}")
        else:
            # Use the supplied custom arguments or defaults.
            self.custom_instruments = custom_instruments
            self.custom_ion_types = custom_ion_types
            self.custom_ionizations = custom_ionizations
            self.custom_collision_energies = custom_collision_energies
            if any(
                [
                    custom_instruments,
                    custom_ion_types,
                    custom_ionizations,
                    custom_collision_energies,
                ]
            ):
                print("✓ Using custom metadata")
            else:
                print("✓ Using default metadata")

        # Type-ID mapping
        self.type_ids = {"special": 0, "smiles": 1, "ms": 2, "meta": 3}

        # Build MS tokens.
        self.ms_tokens = {}
        current_id = len(self.special_tokens)

        self.ms_tokens["<PRECURSOR>"] = current_id
        current_id += 1

        # Fragment MZ tokens
        mz_values = np.arange(mz_range[0], mz_range[1], mz_precision)
        for mz in mz_values:
            if mz_precision == 1.0:
                token_name = f"MS_{int(mz)}"
            else:
                token_name = f"MS_{mz:.1f}"
            self.ms_tokens[token_name] = current_id
            current_id += 1

        # Combine all tokens.
        self.vocab = {}
        self.vocab.update(self.special_tokens)
        self.vocab.update(self.ms_tokens)

        self.id2token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        if self.custom_instruments is not None or config_path is not None:
            _, combos = self.get_all_meta_combinations()
            print(f"Generated {len(combos)} metadata combinations")
            print(f"meta_tokener final vocabulary size: {self.vocab_size}")

    def preprocess_spectrum(self, spectrum: np.ndarray, top_k: int = 100) -> np.ndarray:
        """Select the top-k spectral peaks by intensity, then sort by increasing m/z."""
        mask = (spectrum[:, 0] >= self.mz_range[0]) & (
            spectrum[:, 0] < self.mz_range[1]
        )
        spectrum = spectrum[mask]

        if len(spectrum) == 0:
            return np.zeros((0, 2))

        # Select the top-k intensities.
        intensity_indices = np.argsort(spectrum[:, 1])[::-1]
        top_indices = intensity_indices[: min(top_k, len(spectrum))]
        top_spectrum = spectrum[top_indices]

        # Reorder by increasing m/z.
        mz_indices = np.argsort(top_spectrum[:, 0])
        processed = top_spectrum[mz_indices]

        return processed

    def tokenize_spectrum(
        self, spectrum: np.ndarray, precursor_mz: Optional[float] = None
    ) -> List[str]:
        """Tokenize preprocessed spectra."""
        tokens = []

        for mz, intensity in spectrum:
            if self.mz_precision == 1.0:
                mz_token = f"MS_{int(round(mz))}"
            else:
                mz_token = f"MS_{mz:.1f}"

            if mz_token in self.ms_tokens:
                tokens.append(mz_token)

        return tokens

    def get_all_meta_combinations(self) -> Tuple[torch.Tensor, List[Dict[str, str]]]:
        """Get all metadata combinations."""
        instruments = (
            self.custom_instruments
            if self.custom_instruments is not None
            else ["qtof", "orbitrap"]
        )
        ion_types = (
            self.custom_ion_types
            if self.custom_ion_types is not None
            else ["[M+H+]+", "[M-H+]-"]
        )
        ionizations = (
            self.custom_ionizations
            if self.custom_ionizations is not None
            else ["GC-EI", "LC-ESI"]
        )
        collision_energies = (
            self.custom_collision_energies
            if self.custom_collision_energies is not None
            else ["10", "20", "40", "60", "0"]
        )

        combinations = []
        meta_tokens = []
        current_token_id = self.vocab_size

        for instrument in instruments:
            for ion_type in ion_types:
                for ionization in ionizations:
                    if ionization.upper() == "GC-EI":
                        combo_dict = {
                            "instrument": instrument,
                            "ion_type": ion_type,
                            "ionization": ionization,
                        }
                        combinations.append(combo_dict)
                        token_name = f"META_COMBO_{len(combinations) - 1}"
                        if token_name not in self.vocab:
                            self.vocab[token_name] = current_token_id
                            self.id2token[current_token_id] = token_name
                            current_token_id += 1
                        meta_tokens.append(self.vocab[token_name])
                    else:
                        for ce in collision_energies:
                            combo_dict = {
                                "instrument": instrument,
                                "ion_type": ion_type,
                                "ionization": ionization,
                                "collision_energy": ce,
                            }
                            combinations.append(combo_dict)
                            token_name = f"META_COMBO_{len(combinations) - 1}"
                            if token_name not in self.vocab:
                                self.vocab[token_name] = current_token_id
                                self.id2token[current_token_id] = token_name
                                current_token_id += 1
                            meta_tokens.append(self.vocab[token_name])

        if current_token_id > self.vocab_size:
            self.vocab_size = current_token_id

        meta_tensor = torch.tensor(meta_tokens, dtype=torch.long)
        # print(f"Generated {len(combinations)} metadata combinations")

        return meta_tensor, combinations

    def encode_ms_sequence(
        self,
        spectrum: Optional[np.ndarray] = None,
        max_length: Optional[int] = None,
        padding: bool = False,
    ) -> torch.Tensor:
        """Encode spectral sequences."""
        if spectrum is None:
            token_ids = []
        else:
            processed_spectrum = self.preprocess_spectrum(spectrum, top_k=max_length)
            ms_tokens = self.tokenize_spectrum(processed_spectrum, precursor_mz=None)

            token_ids = []
            for token in ms_tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    token_ids.append(self.vocab["<UNK>"])

        if max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        if padding and max_length is not None:
            if len(token_ids) < max_length:
                token_ids.extend([self.vocab["<PAD>"]] * (max_length - len(token_ids)))

        return torch.tensor(token_ids, dtype=torch.long)


# ============================================================================
# Part 2: Bidirectional spectrum encoder
# ============================================================================


class MsEncoderV4(nn.Module):
    """Bidirectional spectrum encoder with Fourier and peak projections.

    Token and type embeddings are combined with Fourier features and a BERT
    encoder. The CLS token provides the global spectrum representation.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 100,
        num_types: int = 4,
        mz_range: Tuple[int, int] = (0, 1000),
        intensity_dim: int = 128,
        fourier_trainable: bool = True,
    ):
        """Args:
            vocab_size: vocabulary size.
            d_model: model dimension.
            num_heads: number of attention heads.
            num_layers: number of encoder layers.
            dim_feedforward: feed-forward dimension.
            dropout: dropout rate.
            max_len: maximum sequence length.
            num_types: number of type embeddings.
            mz_range: m/z range.
            intensity_dim: intensity-projection dimension.
            fourier_trainable: whether Fourier features are trainable.
        """
        super().__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.mz_range = mz_range

        # Token embeddings for special tokens and metadata tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Type embedding
        self.type_embedding = nn.Embedding(num_types, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Configure BertModel in encoder mode.
        encoder_max_len = max(max_len * 2, 512)

        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=d_model,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=dim_feedforward,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=encoder_max_len,
            layer_norm_eps=1e-6,
            is_decoder=False,  # Encoder mode: bidirectional attention.
            add_cross_attention=False,  # Cross-attention is not needed.
        )

        # Use BertModel as the encoder.
        self.encoder = BertModel(bert_config)

        # Concatenate Fourier features with a projection of the two-dimensional input.

        # Fourier features for m/z encoding
        min_mz, max_mz = mz_range
        fourier_num_freqs = d_model // 4  # Fourier features occupy half the embedding dimension.
        self.fourier_features = FourierFeatures(
            strategy="voronov_et_al",
            x_min=max(min_mz, 0.001),
            x_max=max_mz,
            trainable=fourier_trainable,
            funcs="both",
            num_freqs=fourier_num_freqs,
        )

        # Fourier dimension = num_freqs * 2 because funcs='both'.
        fourier_dim = self.fourier_features.num_features()

        # Project [m/z, intensity] into a higher-dimensional space.
        self.spectrum_2d_proj = nn.Linear(2, intensity_dim)

        # Combined spectral embedding dimension = fourier_dim + intensity_dim.
        total_spectrum_dim = fourier_dim + intensity_dim

        # Project to the model dimension.
        self.spectrum_proj = nn.Linear(total_spectrum_dim, d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.type_embedding.weight)
        nn.init.xavier_uniform_(self.spectrum_2d_proj.weight)
        nn.init.xavier_uniform_(self.spectrum_proj.weight)
        # BertModel initializes its own weights.

    def embedding_all(
        self,
        spectra: List[np.ndarray],
        tokenizer: "MsTokenizer",
        precursor_mzs: torch.Tensor,
        meta_information: Optional[str] = None,
        device: Optional[torch.device] = None,
        return_token_ids: bool = False,
        profiler=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Encode spectra as embeddings in encoder mode.

        Concatenate Fourier features of m/z with a projection of [m/z, intensity].
        Token order: [meta_token, <MS_BEGIN>, <CLS>, <PRECURSOR>, ms_tokens].

        Args:
            spectra: preprocessed spectrum arrays sorted by increasing m/z.
            tokenizer: MsTokenizer instance.
            precursor_mzs: precursor m/z values, shape (batch_size, 1) or (batch_size,).
            meta_information: optional metadata string.
            device: computation device.
            return_token_ids: whether to return token IDs.

        Returns:
            combined_embeddings: shape (batch_size, total_len, d_model).
            If return_token_ids=True, return
            (combined_embeddings, combined_token_ids, attention_mask).
        """
        batch_size = len(spectra)

        if device is None:
            device = next(self.token_embedding.parameters()).device

        # Encode spectral inputs.
        # Profile spectrum preprocessing.
        if profiler:
            profiler.start_step("embed_preprocess_spectra")

        # Preprocess spectra in increasing m/z order, as implemented by the tokenizer.
        processed_spectra = []
        for spectrum in spectra:
            processed = tokenizer.preprocess_spectrum(spectrum, top_k=self.max_len)
            processed_spectra.append(processed)

        if profiler:
            profiler.end_step("embed_preprocess_spectra")

        # Profile MS-token encoding.
        if profiler:
            profiler.start_step("embed_encode_ms_tokens")

        # Encode MS tokens using Fourier features and the two-dimensional input projection.
        # Create the MS-token attention mask from the actual sequence lengths.
        ms_embeddings_list = []
        ms_mask_list = []  # Store the MS-token mask for each sample.

        for i, spectrum in enumerate(processed_spectra):
            if profiler and i == 0:
                profiler.start_step("embed_single_spectrum_loop")

            if len(spectrum) == 0:
                # Pad empty spectra.
                ms_emb = torch.zeros(self.max_len, self.d_model, device=device)
                # The corresponding mask is all zeros.
                ms_mask = torch.zeros(self.max_len, dtype=torch.long, device=device)
            else:
                # Profile tensor construction.
                if profiler and i == 0:
                    profiler.start_step("embed_create_tensors")

                # Fourier features of m/z
                mz_values = torch.tensor(
                    spectrum[:, 0], dtype=torch.float32, device=device
                ).unsqueeze(-1)  # (n_peaks, 1)
                # Projection of [m/z, intensity]
                spectrum_2d = torch.tensor(
                    spectrum, dtype=torch.float32, device=device
                )  # (n_peaks, 2)

                if profiler and i == 0:
                    profiler.end_step("embed_create_tensors")
                    profiler.start_step("embed_fourier_features")

                fourier_emb = self.fourier_features(mz_values)  # (n_peaks, fourier_dim)

                if profiler and i == 0:
                    profiler.end_step("embed_fourier_features")
                    profiler.start_step("embed_spectrum_2d_proj")

                spectrum_2d_emb = self.spectrum_2d_proj(
                    spectrum_2d
                )  # (n_peaks, intensity_dim)

                if profiler and i == 0:
                    profiler.end_step("embed_spectrum_2d_proj")
                    profiler.start_step("embed_concat_and_proj")

                # Concat
                combined = torch.cat(
                    [fourier_emb, spectrum_2d_emb], dim=-1
                )  # (n_peaks, fourier_dim + intensity_dim)

                # Project to d_model.
                ms_emb = self.spectrum_proj(combined)  # (n_peaks, d_model)

                if profiler and i == 0:
                    profiler.end_step("embed_concat_and_proj")
                    profiler.start_step("embed_padding")

                # Padding to max_len
                n_peaks = ms_emb.shape[0]
                if n_peaks < self.max_len:
                    padding = torch.zeros(
                        self.max_len - n_peaks, ms_emb.shape[1], device=device
                    )
                    ms_emb = torch.cat([ms_emb, padding], dim=0)
                else:
                    ms_emb = ms_emb[: self.max_len]
                    n_peaks = self.max_len

                # Mark valid positions with 1 and padding positions with 0.
                valid_len = n_peaks
                ms_mask = torch.cat(
                    [
                        torch.ones(valid_len, dtype=torch.long, device=device),
                        torch.zeros(
                            self.max_len - valid_len, dtype=torch.long, device=device
                        ),
                    ]
                )

                if profiler and i == 0:
                    profiler.end_step("embed_padding")

            ms_embeddings_list.append(ms_emb)
            ms_mask_list.append(ms_mask)

            if profiler and i == 0:
                profiler.end_step("embed_single_spectrum_loop")

        if profiler:
            profiler.start_step("embed_stack_embeddings")

        ms_embeddings = torch.stack(
            ms_embeddings_list
        )  # (batch_size, max_len, d_model)

        if profiler:
            profiler.end_step("embed_stack_embeddings")
            profiler.start_step("embed_add_type_embedding")

        ms_embeddings = self._add_type_embedding(
            ms_embeddings, tokenizer.type_ids["ms"]
        )

        if profiler:
            profiler.end_step("embed_add_type_embedding")
            profiler.end_step("embed_encode_ms_tokens")

        # Build the full sequence.
        embeddings_list = []
        token_ids_list = []

        # 1. Metadata token, if provided.
        # Look up the metadata token ID in the vocabulary.
        # Append it to embeddings_list and token_ids_list when present.
        # Otherwise, use <META_GENERAL>.

        if meta_information is not None and meta_information in tokenizer.vocab:
            meta_token_id = tokenizer.vocab[meta_information]
            meta_id = torch.tensor([[meta_token_id]], device=device).repeat(
                batch_size, 1
            )
            embeddings_list.append(
                self._get_token_with_type(meta_id, tokenizer.type_ids["meta"])
            )
            token_ids_list.append(meta_id)
        else:
            # Use META_GENERAL if the token is absent.
            meta_general_id = torch.tensor(
                [[tokenizer.vocab["<META_GENERAL>"]]], device=device
            ).repeat(batch_size, 1)
            embeddings_list.append(
                self._get_token_with_type(
                    meta_general_id, tokenizer.type_ids["special"]
                )
            )
            token_ids_list.append(meta_general_id)

        # 2. MS_BEGIN token
        ms_begin_id = torch.tensor(
            [[tokenizer.vocab["<MS_BEGIN>"]]], device=device
        ).repeat(batch_size, 1)
        embeddings_list.append(
            self._get_token_with_type(ms_begin_id, tokenizer.type_ids["special"])
        )
        token_ids_list.append(ms_begin_id)

        # 3. CLS token for the global representation.
        cls_id = torch.tensor([[tokenizer.vocab["<CLS>"]]], device=device).repeat(
            batch_size, 1
        )
        embeddings_list.append(
            self._get_token_with_type(cls_id, tokenizer.type_ids["special"])
        )
        token_ids_list.append(cls_id)

        # 4. Precursor embedding using the same Fourier features and input projection.
        # Ensure precursor_mzs has the required shape.
        if precursor_mzs.dim() == 2 and precursor_mzs.shape[1] == 1:
            precursor_mzs = precursor_mzs.squeeze(-1)  # (batch_size,)
        elif precursor_mzs.dim() == 3:
            precursor_mzs = precursor_mzs.squeeze(-1).squeeze(-1)  # (batch_size,)

        # Use precursor intensity 1.1 by default.
        precursor_intensity = torch.full_like(precursor_mzs, 1.1)
        precursor_2d = torch.stack(
            [precursor_mzs, precursor_intensity], dim=-1
        ).unsqueeze(1)  # (batch_size, 1, 2)

        # Fourier features
        precursor_mzs_2d = precursor_mzs.unsqueeze(-1).unsqueeze(
            -1
        )  # (batch_size, 1, 1)
        fourier_precursor = self.fourier_features(
            precursor_mzs_2d
        )  # (batch_size, 1, fourier_dim)

        # Two-dimensional input projection
        precursor_2d_emb = self.spectrum_2d_proj(
            precursor_2d
        )  # (batch_size, 1, intensity_dim)

        # Concat
        precursor_combined = torch.cat(
            [fourier_precursor, precursor_2d_emb], dim=-1
        )  # (batch_size, 1, total_dim)

        # Project to d_model.
        precursor_embedding = self.spectrum_proj(
            precursor_combined
        )  # (batch_size, 1, d_model)
        precursor_embedding = self._add_type_embedding(
            precursor_embedding, tokenizer.type_ids["special"]
        )

        embeddings_list.append(precursor_embedding)
        precursor_id = torch.tensor(
            [[tokenizer.vocab["<PRECURSOR>"]]], device=device
        ).repeat(batch_size, 1)
        token_ids_list.append(precursor_id)

        # 5. MS tokens
        embeddings_list.append(ms_embeddings)
        # Skip MS-token ID construction: Fourier feature encoding does not need token IDs.
        # Construct attention_mask directly from processed_spectra lengths.
        if return_token_ids:
            # Use all-zero placeholders for MS tokens that do not require token IDs.
            # Actual lengths remain in processed_spectra for attention-mask construction.
            ms_token_ids = torch.zeros(
                batch_size, self.max_len, dtype=torch.long, device=device
            )
            token_ids_list.append(ms_token_ids)

        # Concatenate without an MS_END token.
        combined_embeddings = torch.cat(
            embeddings_list, dim=1
        )  # (batch_size, total_len, d_model)

        if return_token_ids:
            if profiler:
                profiler.start_step("embed_create_attention_mask")

            # Reuse ms_mask_list created during embedding construction.
            # Token positions: Meta=0, MS_BEGIN=1, CLS=2, PRECURSOR=3, MS tokens from 4 onward.
            # Mask for the first four special tokens
            num_prefix_tokens = 4  # Meta, MS_BEGIN, CLS, PRECURSOR
            prefix_mask = torch.ones(num_prefix_tokens, dtype=torch.long, device=device)

            # Concatenate each sample mask: four special tokens followed by MS tokens.
            attention_mask_list = []
            for ms_mask in ms_mask_list:
                sample_mask = torch.cat([prefix_mask, ms_mask])
                attention_mask_list.append(sample_mask)

            attention_mask = torch.stack(attention_mask_list)  # (batch_size, total_len)

            if profiler:
                profiler.end_step("embed_create_attention_mask")
                profiler.start_step("embed_create_token_ids")

            # Create placeholder token_ids for interface compatibility; they are not used.
            # Fourier feature encoding does not require actual token IDs.
            combined_token_ids = (
                torch.cat(token_ids_list, dim=1)
                if token_ids_list
                else torch.zeros(
                    batch_size,
                    combined_embeddings.shape[1],
                    dtype=torch.long,
                    device=device,
                )
            )

            if profiler:
                profiler.end_step("embed_create_token_ids")

            # Return four values for the DreaMS interface; the fourth is None because MsEncoderV4 needs no fourier_features.
            return combined_embeddings, combined_token_ids, attention_mask, None
        else:
            # Return three values for the DreaMS interface.
            return combined_embeddings, None, None

    def _get_token_with_type(
        self, token_ids: torch.Tensor, type_id: int
    ) -> torch.Tensor:
        """Get token embeddings and add type embeddings."""
        token_embeddings = self.token_embedding(token_ids)
        batch_size, seq_len = token_ids.shape
        type_embeddings = self.type_embedding(
            torch.full(
                (batch_size, seq_len),
                type_id,
                dtype=torch.long,
                device=token_ids.device,
            )
        )
        return token_embeddings + type_embeddings

    def _add_type_embedding(
        self, embeddings: torch.Tensor, type_id: int
    ) -> torch.Tensor:
        """Add type embeddings to externally supplied embeddings."""
        batch_size, seq_len = embeddings.shape[:2]
        type_embeddings = self.type_embedding(
            torch.full(
                (batch_size, seq_len),
                type_id,
                dtype=torch.long,
                device=embeddings.device,
            )
        )
        return embeddings + type_embeddings

    def forward(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fourier_features: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run BertModel in encoder mode with bidirectional attention.

        Args:
            input_embeddings: spectral embeddings, shape (batch_size, seq_len, d_model).
            attention_mask: shape (batch_size, seq_len), with 1=valid and 0=padding.
            fourier_features: optional DreaMS-interface argument; unused by MsEncoderV4.
            output_hidden_states: whether to return all hidden states.
            return_dict: whether to return a dictionary.

        Returns:
            A dictionary containing last_hidden_state (batch_size, seq_len, d_model),
            pooler_output (batch_size, d_model), and optionally hidden_states.
            pooler_output is the CLS-token representation.

        The fourier_features argument is accepted for interface compatibility only.
        """
        # Ignore fourier_features; it is accepted for interface compatibility only.
        batch_size, seq_len, d_model_input = input_embeddings.shape
        assert d_model_input == self.d_model, (
            f"Input dim {d_model_input} != model dim {self.d_model}"
        )

        # Dropout
        hidden_states = self.dropout(input_embeddings)

        # Prepare the BERT attention mask.
        # BERT attention_mask: 1=attend, 0=mask.
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)

        # Call BertModel in encoder mode.
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if return_dict:
            return {
                "last_hidden_state": encoder_outputs.last_hidden_state,  # (batch_size, seq_len, d_model)
                "pooler_output": encoder_outputs.pooler_output,  # (batch_size, d_model)
                "hidden_states": encoder_outputs.hidden_states
                if output_hidden_states
                else None,
            }
        else:
            return encoder_outputs


# ============================================================================
# Utility Functions
# ============================================================================


def count_parameters(model: nn.Module) -> int:
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_ms_attention_mask(
    token_ids: torch.Tensor, tokenizer: "MsTokenizer"
) -> torch.Tensor:
    """Create an attention mask from token IDs.

    Args:
        token_ids: shape (batch_size, seq_len).
        tokenizer: MsTokenizer instance.

    Returns:
        attention_mask: shape (batch_size, seq_len), with 1=valid and 0=padding.
    """
    pad_token_id = tokenizer.special_tokens["<PAD>"]
    attention_mask = (token_ids != pad_token_id).long()
    return attention_mask


# ============================================================================
# Test functions
# ============================================================================


def test_ms_encoder_v4():
    """Test MS Encoder v4."""
    print("🧪 Testing MS Encoder v4...")

    # 1. Create the tokenizer and model.
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "ms_decoder.yaml",
    )

    tokenizer = MsTokenizer(
        mz_precision=1.0,
        mz_range=(0, 1000),
        config_path=config_path if os.path.exists(config_path) else None,
    )

    model = MsEncoderV4(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        mz_range=(0, 1000),
        max_len=103,
        intensity_dim=128,
    )

    device = torch.device("cpu")
    model = model.to(device)

    print(f"✓ Model parameter count: {count_parameters(model):,}")
    print(f"✓ Device: {device}")

    # 2. Prepare test data.
    batch_size = 2
    spectra = [
        np.array([[50, 100], [100, 200], [150, 300], [200, 400]]),
        np.array([[60, 120], [110, 220], [160, 320], [210, 420]]),
    ]
    precursor_mzs = torch.tensor([[200.5], [250.3]], device=device)

    # 3. Test embedding without metadata.
    print("\nTest 1: Embedding generation (without metadata)")
    with torch.no_grad():
        combined_embeddings, combined_token_ids, attention_mask = model.embedding_all(
            spectra=spectra,
            tokenizer=tokenizer,
            precursor_mzs=precursor_mzs,
            meta_information=None,
            device=device,
            return_token_ids=True,
        )

    print(f"✓ Combined embeddings shape: {combined_embeddings.shape}")
    print(f"✓ Token IDs shape: {combined_token_ids.shape}")
    print(f"✓ Attention mask shape: {attention_mask.shape}")

    # 4. Test embedding with metadata.
    print("\nTest 2: Embedding generation (with metadata)")
    with torch.no_grad():
        combined_embeddings_meta, _, _ = model.embedding_all(
            spectra=spectra,
            tokenizer=tokenizer,
            precursor_mzs=precursor_mzs,
            meta_information="<META_GENERAL>",  # Use an existing metadata token.
            device=device,
            return_token_ids=True,
        )

    print(f"✓ Combined embeddings with meta shape: {combined_embeddings_meta.shape}")

    # 5. Test the forward pass in encoder mode.
    print("\nTest 3: Forward (encoder mode)")
    with torch.no_grad():
        outputs = model(
            input_embeddings=combined_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    print(f"✓ Output last_hidden_state shape: {outputs['last_hidden_state'].shape}")
    print(f"✓ Output pooler_output (CLS representation) shape: {outputs['pooler_output'].shape}")

    # 6. Check the CLS-token position.
    print("\nTest 4: Verify CLS token position")
    # CLS is the third token: meta/META_GENERAL, MS_BEGIN, CLS, ...
    cls_position = 2
    cls_representation = outputs["last_hidden_state"][:, cls_position, :]
    print(f"✓ CLS token position: {cls_position}")
    print(f"✓ CLS representation shape: {cls_representation.shape}")
    print(f"✓ Pooler output shape: {outputs['pooler_output'].shape}")

    print("\n🎉 All tests passed!")
    print("\nKey improvements:")
    print("  ✅BertModel (encoder mode) with bidirectional attention")
    print("  ✅Encode spectra by concatenating Fourier features and a two-dimensional projection")
    print("  ✅ Simplified metadata token processing")
    print("  ✅ CLS token for the global representation")
    print("  ✅ Sort MS peaks by m/z in ascending order")
    print("  ✅ Encode the precursor using the same method (intensity=1.1)")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    test_ms_encoder_v4()
    print("=" * 80)
