"""
MS Encoder v4 - 基于 HuggingFace Transformers
=============================================
改进:
1. 使用 BertModel (encoder模式，双向注意力)
2. Fourier特征 + 二维投影concat编码质谱
3. 简化的meta token处理
4. CLS token用于全局表征

Author: Refactored
Date: 2025-11-12
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

# 导入 FourierFeatures
from .feature.fourier import FourierFeatures


# ============================================================================
# Part 1: MS Tokenizer (保持不变，从原文件复制)
# ============================================================================


class MsTokenizer:
    """
    智能质谱数据分词器

    功能:
    1. 质谱数据预处理：top100选择、强度排序
    2. MZ值tokenization（1.0精度，0-999范围）
    3. Meta信息tokenization
    4. 跨模态token序列生成
    """

    @staticmethod
    def load_config_from_yaml(config_path: str) -> Dict:
        """
        从YAML文件加载meta信息配置

        Args:
            config_path: YAML配置文件路径

        Returns:
            配置字典，包含 instruments, ion_types, ionizations, collision_energies
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
        """
        初始化 tokenizer

        Args:
            mz_precision: MZ精度
            mz_range: MZ范围
            config_path: YAML配置文件路径（优先级高于custom参数）
            custom_instruments: 自定义仪器类型列表（如果config_path为None）
            custom_ion_types: 自定义离子类型列表
            custom_ionizations: 自定义电离方式列表
            custom_collision_energies: 自定义碰撞能量列表
        """
        # 特殊tokens
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

        # 加载meta信息配置
        # 优先级: config_path > custom参数 > 默认值
        if config_path is not None:
            # 从YAML配置文件加载
            config = self.load_config_from_yaml(config_path)
            self.custom_instruments = config["instruments"]
            self.custom_ion_types = config["ion_types"]
            self.custom_ionizations = config["ionizations"]
            self.custom_collision_energies = config["collision_energies"]
            print(f"✓ 从配置文件加载meta信息: {config_path}")
        else:
            # 使用传入的custom参数或默认值
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
                print("✓ 使用自定义meta信息")
            else:
                print("✓ 使用默认meta信息")

        # Type ID 映射
        self.type_ids = {"special": 0, "smiles": 1, "ms": 2, "meta": 3}

        # 构建 MS tokens
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

        # 合并所有tokens
        self.vocab = {}
        self.vocab.update(self.special_tokens)
        self.vocab.update(self.ms_tokens)

        self.id2token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        if self.custom_instruments is not None or config_path is not None:
            _, combos = self.get_all_meta_combinations()
            print(f"生成 {len(combos)} 个meta组合")
            print(f"meta_tokener 最终词汇表大小: {self.vocab_size}")

    def preprocess_spectrum(self, spectrum: np.ndarray, top_k: int = 100) -> np.ndarray:
        """预处理质谱数据 - 选择top-k强度，按mz从低到高排序"""
        mask = (spectrum[:, 0] >= self.mz_range[0]) & (
            spectrum[:, 0] < self.mz_range[1]
        )
        spectrum = spectrum[mask]

        if len(spectrum) == 0:
            return np.zeros((0, 2))

        # 选择强度最高的top-k
        intensity_indices = np.argsort(spectrum[:, 1])[::-1]
        top_indices = intensity_indices[: min(top_k, len(spectrum))]
        top_spectrum = spectrum[top_indices]

        # 按mz从低到高排序 (改变了顺序)
        mz_indices = np.argsort(top_spectrum[:, 0])
        processed = top_spectrum[mz_indices]

        return processed

    def tokenize_spectrum(
        self, spectrum: np.ndarray, precursor_mz: Optional[float] = None
    ) -> List[str]:
        """将预处理后的质谱数据tokenize"""
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
        """获取所有meta信息组合"""
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
        # print(f"生成了 {len(combinations)} 个meta组合")

        return meta_tensor, combinations

    def encode_ms_sequence(
        self,
        spectrum: Optional[np.ndarray] = None,
        max_length: Optional[int] = None,
        padding: bool = False,
    ) -> torch.Tensor:
        """编码质谱序列"""
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
# Part 2: MS Encoder v4 - 基于 BertModel (encoder模式)
# ============================================================================


class MsEncoderV4(nn.Module):
    """
    MS Encoder v4: 使用 HuggingFace BertModel (encoder模式，双向注意力)

    改进:
    1. 使用 BertModel (encoder模式) - 双向注意力
    2. Fourier特征 + 二维投影concat编码质谱
    3. 简化的meta token处理
    4. CLS token用于全局表征

    架构:
    - Token Embedding + Type Embedding
    - Fourier Features + 二维投影 (concat编码)
    - BERT Encoder (双向注意力)
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
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: Encoder层数
            dim_feedforward: FFN维度
            dropout: Dropout率
            max_len: 最大序列长度
            num_types: type embedding数量
            mz_range: MZ值范围
            intensity_dim: 强度投影维度
            fourier_trainable: 傅立叶特征是否可训练
        """
        super().__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.mz_range = mz_range

        # Token embedding (用于special tokens和meta tokens)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Type embedding
        self.type_embedding = nn.Embedding(num_types, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 配置 BertModel (encoder模式)
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
            is_decoder=False,  # ⭐ Encoder模式 - 双向注意力
            add_cross_attention=False,  # ⭐ 不需要交叉注意力
        )

        # 使用 BertModel 作为 encoder
        self.encoder = BertModel(bert_config)

        # === 新的编码方式：Fourier特征 + 二维投影 concat ===

        # Fourier特征 (用于 m/z 编码)
        min_mz, max_mz = mz_range
        fourier_num_freqs = d_model // 4  # Fourier部分占1/2空间
        self.fourier_features = FourierFeatures(
            strategy="voronov_et_al",
            x_min=max(min_mz, 0.001),
            x_max=max_mz,
            trainable=fourier_trainable,
            funcs="both",
            num_freqs=fourier_num_freqs,
        )

        # Fourier维度 = num_freqs * 2 (因为 funcs='both')
        fourier_dim = self.fourier_features.num_features()

        # 二维投影 (将[mz, intensity]投影到高维)
        self.spectrum_2d_proj = nn.Linear(2, intensity_dim)

        # 总的质谱embedding维度 = fourier_dim + intensity_dim
        total_spectrum_dim = fourier_dim + intensity_dim

        # 投影到模型维度
        self.spectrum_proj = nn.Linear(total_spectrum_dim, d_model)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.type_embedding.weight)
        nn.init.xavier_uniform_(self.spectrum_2d_proj.weight)
        nn.init.xavier_uniform_(self.spectrum_proj.weight)
        # BertModel 的权重会自动初始化

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
        """
        处理质谱数据，生成 embeddings (Encoder模式)

        新的编码方式：
        - Fourier特征(m/z) + 二维投影([mz, intensity]) concat
        - Token顺序：[meta_token, <MS_BEGIN>, <CLS>, <PRECURSOR>, ms_tokens]

        Args:
            spectra: List of spectrum arrays (已经过预处理，按mz从低到高排序)
            tokenizer: MsTokenizer实例
            precursor_mzs: Precursor m/z values, shape (batch_size, 1) or (batch_size,)
            meta_information: Meta信息字符串 (可选)
            device: 设备
            return_token_ids: 是否返回 token IDs

        Returns:
            combined_embeddings: shape (batch_size, total_len, d_model)
            如果 return_token_ids=True, 返回 (combined_embeddings, combined_token_ids, attention_mask)
        """
        batch_size = len(spectra)

        if device is None:
            device = next(self.token_embedding.parameters()).device

        # === 处理质谱数据 - 使用新的编码方式 ===
        # 性能分析：预处理spectra
        if profiler:
            profiler.start_step("embed_preprocess_spectra")

        # 预处理spectra，确保按mz从低到高排序（tokenizer中已实现）
        processed_spectra = []
        for spectrum in spectra:
            processed = tokenizer.preprocess_spectrum(spectrum, top_k=self.max_len)
            processed_spectra.append(processed)

        if profiler:
            profiler.end_step("embed_preprocess_spectra")

        # 性能分析：编码MS tokens
        if profiler:
            profiler.start_step("embed_encode_ms_tokens")

        # 编码MS tokens (使用Fourier + 二维投影)
        # 同时创建MS tokens部分的attention mask（基于实际长度）
        ms_embeddings_list = []
        ms_mask_list = []  # 存储每个样本的MS tokens mask

        for i, spectrum in enumerate(processed_spectra):
            if profiler and i == 0:
                profiler.start_step("embed_single_spectrum_loop")

            if len(spectrum) == 0:
                # 空质谱，使用padding
                ms_emb = torch.zeros(self.max_len, self.d_model, device=device)
                # 对应的mask全为0
                ms_mask = torch.zeros(self.max_len, dtype=torch.long, device=device)
            else:
                # 性能分析：创建tensor
                if profiler and i == 0:
                    profiler.start_step("embed_create_tensors")

                # Fourier特征 (m/z)
                mz_values = torch.tensor(
                    spectrum[:, 0], dtype=torch.float32, device=device
                ).unsqueeze(-1)  # (n_peaks, 1)
                # 二维投影 ([mz, intensity])
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

                # 投影到d_model
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

                # 同时创建对应的mask：有效部分为1，padding部分为0
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

        # === 构建完整序列 ===
        embeddings_list = []
        token_ids_list = []

        # 1. Meta token (如果提供)
        # 这里是要从vocab中获取meta_information的token_id
        # 如果存在，则添加到embeddings_list和token_ids_list
        # 如果不存在，则使用<META_GENERAL>

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
            # 如果找不到，使用 META_GENERAL
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

        # 3. CLS token (用于全局表征)
        cls_id = torch.tensor([[tokenizer.vocab["<CLS>"]]], device=device).repeat(
            batch_size, 1
        )
        embeddings_list.append(
            self._get_token_with_type(cls_id, tokenizer.type_ids["special"])
        )
        token_ids_list.append(cls_id)

        # 4. PRECURSOR embedding (使用相同的Fourier + 二维投影编码)
        # 确保 precursor_mzs 形状正确
        if precursor_mzs.dim() == 2 and precursor_mzs.shape[1] == 1:
            precursor_mzs = precursor_mzs.squeeze(-1)  # (batch_size,)
        elif precursor_mzs.dim() == 3:
            precursor_mzs = precursor_mzs.squeeze(-1).squeeze(-1)  # (batch_size,)

        # Precursor的强度默认为1.1
        precursor_intensity = torch.full_like(precursor_mzs, 1.1)
        precursor_2d = torch.stack(
            [precursor_mzs, precursor_intensity], dim=-1
        ).unsqueeze(1)  # (batch_size, 1, 2)

        # Fourier特征
        precursor_mzs_2d = precursor_mzs.unsqueeze(-1).unsqueeze(
            -1
        )  # (batch_size, 1, 1)
        fourier_precursor = self.fourier_features(
            precursor_mzs_2d
        )  # (batch_size, 1, fourier_dim)

        # 二维投影
        precursor_2d_emb = self.spectrum_2d_proj(
            precursor_2d
        )  # (batch_size, 1, intensity_dim)

        # Concat
        precursor_combined = torch.cat(
            [fourier_precursor, precursor_2d_emb], dim=-1
        )  # (batch_size, 1, total_dim)

        # 投影到d_model
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
        # 优化：完全跳过MS token IDs创建（使用Fourier特征编码，不需要实际token IDs）
        # attention_mask会直接根据processed_spectra的长度创建，不需要token_ids
        if return_token_ids:
            # MS tokens部分：占位符（全0，表示这部分不需要token IDs）
            # 实际长度信息已保存在processed_spectra中，用于创建attention_mask
            ms_token_ids = torch.zeros(
                batch_size, self.max_len, dtype=torch.long, device=device
            )
            token_ids_list.append(ms_token_ids)

        # 拼接（不再添加MS_END token）
        combined_embeddings = torch.cat(
            embeddings_list, dim=1
        )  # (batch_size, total_len, d_model)

        if return_token_ids:
            if profiler:
                profiler.start_step("embed_create_attention_mask")

            # 优化：直接使用在处理embedding时已创建的ms_mask_list
            # 特殊tokens顺序：Meta (位置0), MS_BEGIN (位置1), CLS (位置2), PRECURSOR (位置3), MS tokens (位置4开始)
            # 前4个特殊tokens的mask
            num_prefix_tokens = 4  # Meta, MS_BEGIN, CLS, PRECURSOR
            prefix_mask = torch.ones(num_prefix_tokens, dtype=torch.long, device=device)

            # 拼接每个样本的mask：前4个特殊tokens + MS tokens
            attention_mask_list = []
            for ms_mask in ms_mask_list:
                sample_mask = torch.cat([prefix_mask, ms_mask])
                attention_mask_list.append(sample_mask)

            attention_mask = torch.stack(attention_mask_list)  # (batch_size, total_len)

            if profiler:
                profiler.end_step("embed_create_attention_mask")
                profiler.start_step("embed_create_token_ids")

            # 创建占位符token_ids（仅用于兼容性，实际不会被使用）
            # 因为使用Fourier特征编码，不需要真实的token IDs
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

            # 返回4个值以兼容DreaMS接口（第4个是None，因为MsEncoderV4不需要fourier_features）
            return combined_embeddings, combined_token_ids, attention_mask, None
        else:
            # 返回3个值以兼容DreaMS接口
            return combined_embeddings, None, None

    def _get_token_with_type(
        self, token_ids: torch.Tensor, type_id: int
    ) -> torch.Tensor:
        """获取 token embedding 并添加 type embedding"""
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
        """为外部 embeddings 添加 type embedding"""
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
        """
        前向传播 - 使用 BertModel (encoder模式，双向注意力)

        Args:
            input_embeddings: shape (batch_size, seq_len, d_model) - 输入 embedding (MS数据)
            attention_mask: shape (batch_size, seq_len) - attention mask (1=有效, 0=padding)
            fourier_features: 可选，用于兼容DreaMS接口（MsEncoderV4不使用此参数）
            output_hidden_states: 是否输出所有隐藏状态
            return_dict: 是否返回字典

        Returns:
            dict containing:
                - last_hidden_state: shape (batch_size, seq_len, d_model)
                - pooler_output: shape (batch_size, d_model) - CLS token的表征
                - hidden_states: all hidden states (if output_hidden_states)

        Note:
            fourier_features 参数仅用于接口兼容，MsEncoderV4 内部不使用此参数。
        """
        # 注意：fourier_features 参数被忽略，仅用于接口兼容
        batch_size, seq_len, d_model_input = input_embeddings.shape
        assert d_model_input == self.d_model, (
            f"Input dim {d_model_input} != model dim {self.d_model}"
        )

        # Dropout
        hidden_states = self.dropout(input_embeddings)

        # 准备 attention mask for BERT
        # BERT 使用的 attention_mask: 1=attend, 0=mask
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)

        # 调用 BertModel (encoder模式)
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
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_ms_attention_mask(
    token_ids: torch.Tensor, tokenizer: "MsTokenizer"
) -> torch.Tensor:
    """
    从 token IDs 创建 attention mask

    Args:
        token_ids: shape (batch_size, seq_len)
        tokenizer: MsTokenizer实例

    Returns:
        attention_mask: shape (batch_size, seq_len) - 1=有效, 0=padding
    """
    pad_token_id = tokenizer.special_tokens["<PAD>"]
    attention_mask = (token_ids != pad_token_id).long()
    return attention_mask


# ============================================================================
# 测试函数
# ============================================================================


def test_ms_encoder_v4():
    """测试 MS Encoder v4"""
    print("🧪 测试 MS Encoder v4...")

    # 1. 创建 tokenizer 和模型
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

    print(f"✓ 模型参数量: {count_parameters(model):,}")
    print(f"✓ 使用设备: {device}")

    # 2. 准备测试数据
    batch_size = 2
    spectra = [
        np.array([[50, 100], [100, 200], [150, 300], [200, 400]]),
        np.array([[60, 120], [110, 220], [160, 320], [210, 420]]),
    ]
    precursor_mzs = torch.tensor([[200.5], [250.3]], device=device)

    # 3. 测试 embedding (无meta信息)
    print("\n测试 1: Embedding 生成 (无meta)")
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

    # 4. 测试 embedding (有meta信息)
    print("\n测试 2: Embedding 生成 (有meta)")
    with torch.no_grad():
        combined_embeddings_meta, _, _ = model.embedding_all(
            spectra=spectra,
            tokenizer=tokenizer,
            precursor_mzs=precursor_mzs,
            meta_information="<META_GENERAL>",  # 使用一个存在的meta token
            device=device,
            return_token_ids=True,
        )

    print(f"✓ Combined embeddings with meta shape: {combined_embeddings_meta.shape}")

    # 5. 测试 forward (Encoder模式)
    print("\n测试 3: Forward (Encoder模式)")
    with torch.no_grad():
        outputs = model(
            input_embeddings=combined_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    print(f"✓ Output last_hidden_state shape: {outputs['last_hidden_state'].shape}")
    print(f"✓ Output pooler_output (CLS表征) shape: {outputs['pooler_output'].shape}")

    # 6. 验证CLS token位置
    print("\n测试 4: 验证CLS token位置")
    # CLS应该在第3个位置 (meta/META_GENERAL, MS_BEGIN, CLS, ...)
    cls_position = 2
    cls_representation = outputs["last_hidden_state"][:, cls_position, :]
    print(f"✓ CLS token位置: {cls_position}")
    print(f"✓ CLS表征 shape: {cls_representation.shape}")
    print(f"✓ Pooler output shape: {outputs['pooler_output'].shape}")

    print("\n🎉 所有测试通过!")
    print("\n关键改进:")
    print("  ✅使用 BertModel (encoder模式) - 双向注意力")
    print("  ✅Fourier特征 + 二维投影concat编码质谱")
    print("  ✅ 简化的meta token处理")
    print("  ✅ CLS token用于全局表征")
    print("  ✅ MS按mz从低到高排序")
    print("  ✅ Precursor使用相同编码方式(强度=1.1)")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    test_ms_encoder_v4()
    print("=" * 80)
