#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGE-M3 向量匹配工具 - 云端 vLLM Embedding 版

改动说明（相比 FlagEmbedding 版本）：
- Embedding 引擎：FlagEmbedding/transformers → vLLM LLM(task='embedding')
- 不再依赖 FlagEmbedding、transformers，与 Step2 共用同一 vLLM 环境
- 其余匹配逻辑完全不变：GPU cosine topk、向量缓存、可疑职业验证、输出列结构
- 新增断点续跑：每个 query_chunk_size 批次写一个 chunk 文件并保存 checkpoint，
  崩溃重启后自动跳过已完成批次，最终合并为 data.parquet
- 新增 Reranker 精排：可选启用 BGE-Reranker-v2-M3 对 top-k 候选重排序
- 新增 min_embed_score 阈值：低于该分数的匹配标记为未匹配

使用方式：
    python embed_and_match_gpu.py --config config.yaml --year 2021
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# =============================================================================
# 日志
# =============================================================================

def setup_logging(year: str = "") -> logging.Logger:
    logger = logging.getLogger(f"job2occ.step3_{year}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)

    # 日志文件持久化
    base_dir = os.environ.get("PIPELINE_BASE_DIR", "/root/autodl-tmp")
    log_dir = Path(base_dir) / "data" / "output" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / f"step3_{year}.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class Config:
    # 输入输出（路径由 --year 参数动态填充）
    input_dir: str
    output_dir: str

    # 职业分类大典参考表
    occupation_path: str
    occupation_sheet: str
    occupation_name_field: str
    occupation_task_field: str

    # 向量缓存
    cache_dir: str

    # 模型
    embed_model: str
    rerank_model: str
    device: str
    model_dtype: str                  # 云端用 float16

    # 匹配参数
    embedding_batch_size: int         # 云端用 256
    query_chunk_size: int             # GPU cosine 分块大小，默认 250000
    top_k: int
    use_rerank: bool
    rerank_score_threshold: float
    rerank_min_embed_score: float
    rerank_normalize: bool

    # 最低 embedding 相似度阈值
    min_embed_score: float = 0.55

    # 字段配置
    job_text_field: str = "standardized_job_description"

    # 输出标签（与本地一致）
    unmatched_label: str = "未匹配"
    is_match_yes: str = "是"
    is_match_no: str = "否"

    # 断点目录（默认值，无需修改 config.yaml）
    checkpoint_dir: str = "/root/autodl-tmp/data/output/checkpoints/step3"

    # 可疑职业验证规则
    suspicious_occupations: Dict[str, List[str]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # 环境变量替换路径前缀
        base_dir = os.environ.get("PIPELINE_BASE_DIR", "")
        if base_dir:
            for key in ("input_dir", "output_dir", "occupation_path", "cache_dir",
                        "embed_model", "rerank_model", "checkpoint_dir"):
                if key in data and isinstance(data[key], str):
                    data[key] = data[key].replace("/root/autodl-tmp", base_dir)
        return cls(**data)

    def input_parquet(self, year: str) -> str:
        return str(Path(self.input_dir) / f"year={year}" / "data.parquet")

    def output_chunk_dir(self, year: str) -> str:
        out_dir = Path(self.output_dir) / f"year={year}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir)

    def output_parquet(self, year: str) -> str:
        return str(Path(self.output_chunk_dir(year)) / "data.parquet")

    def checkpoint_path(self, year: str) -> str:
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        return str(Path(self.checkpoint_dir) / f"step3_{year}.json")


# =============================================================================
# 断点管理（与 Step2 结构一致）
# =============================================================================

class CheckpointManager:
    """持久化断点：记录已完成行数和 chunk 数，支持崩溃后续跑。"""

    def __init__(self, path: str):
        self._path = Path(path)
        self._data: dict = {"rows_done": 0, "chunks_written": 0, "complete": False}
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                pass

    @property
    def rows_done(self) -> int:
        return int(self._data.get("rows_done", 0))

    @property
    def chunks_written(self) -> int:
        return int(self._data.get("chunks_written", 0))

    @property
    def is_complete(self) -> bool:
        return bool(self._data.get("complete", False))

    def update(self, rows_done: int, chunks_written: int):
        self._data.update({"rows_done": rows_done, "chunks_written": chunks_written})
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self._path)

    def mark_complete(self):
        self._data["complete"] = True
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self._path)


# =============================================================================
# 工具函数（与本地版本完全一致）
# =============================================================================

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())


def normalize_level_code(code: str) -> str:
    if not isinstance(code, str):
        code = "" if code is None else str(code)
    code = code.strip()
    if not code:
        return ""
    match = re.match(r"^(\d{1,2})(?:-(\d{1,2}))?(?:-(\d{1,2}))?(?:-(\d{1,2}))?$", code)
    if not match:
        return code
    parts = [p for p in match.groups() if p is not None]
    normalized = [parts[0]]
    for p in parts[1:]:
        normalized.append(p.zfill(2))
    return "-".join(normalized)


def build_occupation_text(row: pd.Series, name_field: str, task_field: str) -> str:
    name = normalize_text(row.get(name_field, ""))
    task = normalize_text(row.get(task_field, ""))
    if name and task:
        return f"{name}：{task}"
    return name or task


def is_invalid_input(text: str) -> bool:
    """判断是否为无效输入（无法识别或过短）"""
    t = text.strip()
    return t == "无法识别" or t == "" or len(t) < 10


def validate_suspicious_match(
    occupation: str,
    description: str,
    score: float,
    suspicious_occupations: Dict[str, List[str]],
) -> bool:
    """验证可疑职业匹配是否合理（与本地版本完全一致）"""
    if occupation in suspicious_occupations:
        required_keywords = suspicious_occupations[occupation]
        has_keyword = any(kw in description for kw in required_keywords)
        if not has_keyword and score < 0.75:
            return False
    return True


# =============================================================================
# 向量缓存（与本地版本一致，新增 dtype 字段）
# =============================================================================

def cache_paths(cache_dir: Path, tag: str) -> Tuple[Path, Path]:
    vec_path = cache_dir / f"{tag}.npy"
    meta_path = cache_dir / f"{tag}.json"
    return vec_path, meta_path


def load_cached_vectors(vec_path: Path, meta_path: Path, meta_key: Dict) -> Optional[np.ndarray]:
    if not (vec_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if meta != meta_key:
        return None
    return np.load(vec_path)


def save_cached_vectors(vec_path: Path, meta_path: Path, meta_key: Dict, vectors: np.ndarray):
    np.save(vec_path, vectors)
    meta_path.write_text(json.dumps(meta_key, ensure_ascii=False, indent=2), encoding="utf-8")


# =============================================================================
# BGE-M3 嵌入模型（vLLM 版）
# =============================================================================

class BGEEmbedder:
    """
    BGE-M3 Embedding 模型（vLLM offline embedding 版）

    用 vLLM LLM(task='embedding') 替代 FlagEmbedding/transformers：
    - 无需单独安装 FlagEmbedding，与 Step2 共用同一环境
    - vLLM 内部自动批处理，GPU 利用率更高
    - embed_batch() 接口保持不变，返回 L2 归一化 GPU Tensor
    """

    def __init__(self, model_path: str, device: str = "cuda",
                 dtype: str = "float16", batch_size: int = 256):
        self.device = device
        self.batch_size = batch_size
        self.torch_dtype = torch.float16 if dtype == "float16" else torch.float32

        print(f"正在加载 BGE-M3 (vLLM): {model_path}  (device={device}, dtype={dtype})")
        from vllm import LLM
        self._llm = LLM(
            model=model_path,
            task="embedding",
            dtype=dtype,
            gpu_memory_utilization=0.45,   # 与 Step2 LLM 分开时留余量；Step3 独立跑时可调高
            trust_remote_code=False,
        )
        print(f"✓ BGE-M3 加载完成")

    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        """
        批量生成 L2 归一化的嵌入向量（返回 GPU Tensor）。
        返回: (N, D) float16 Tensor on GPU
        """
        # vLLM embed 返回 EmbeddingRequestOutput 列表
        outputs = self._llm.embed(texts)
        # 每个 output.outputs.embedding 是 List[float]
        embeddings = [out.outputs.embedding for out in outputs]
        emb = torch.tensor(embeddings, dtype=self.torch_dtype, device=self.device)
        emb = F.normalize(emb, p=2, dim=1)   # L2 归一化，cosine = dot product
        return emb                             # (N, D) GPU Tensor


# =============================================================================
# BGE-Reranker-v2-M3 精排模型
# =============================================================================

class BGEReranker:
    """
    BGE-Reranker-v2-M3 交叉编码器精排模型。

    使用 FlagEmbedding.FlagReranker 加载，对 (query, candidate) 文本对打分。
    用于对 embedding top-k 候选进行精排，提升匹配准确率。

    注意：reranker 是交叉编码器，计算量远大于 embedding，
    全量 1.6 亿条数据开启精排会显著增加耗时（每条需对 top_k 个候选逐一打分）。
    建议仅在数据量较小或对准确率要求极高时启用。
    """

    def __init__(self, model_path: str, device: str = "cuda",
                 normalize: bool = True):
        self.device = device
        self.normalize = normalize
        print(f"正在加载 BGE-Reranker: {model_path}  (device={device})")
        from FlagEmbedding import FlagReranker
        self._reranker = FlagReranker(
            model_path,
            use_fp16=(device == "cuda"),
            device=device,
        )
        print(f"✓ BGE-Reranker 加载完成")

    def compute_scores(self, pairs: List[Tuple[str, str]],
                        batch_size: int = 256) -> List[float]:
        """
        对 (query, candidate) 文本对批量打分，内部按 batch_size 分批以避免 OOM。

        Args:
            pairs: [(query_text, candidate_text), ...] 文本对列表
            batch_size: 每批处理的 pair 数量（默认 256，根据 GPU 显存调整）

        Returns:
            List[float]: 每对的相关性分数（normalize=True 时为 0~1 sigmoid 归一化）
        """
        if not pairs:
            return []

        all_scores: List[float] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start: start + batch_size]
            scores = self._reranker.compute_score(
                batch, normalize=self.normalize,
            )
            # FlagReranker 单个 pair 返回 float，多个返回 list
            if isinstance(scores, (int, float)):
                scores = [float(scores)]
            all_scores.extend(float(s) for s in scores)

        return all_scores


# =============================================================================
# GPU 余弦相似度（分块处理，避免 OOM）
# =============================================================================

def gpu_cosine_topk(
    query_vecs: torch.Tensor,    # (N, D) float16，GPU
    key_vecs: torch.Tensor,      # (M, D) float16，GPU
    top_k: int,
    query_chunk: int = 250_000,  # 每次处理的 query 行数（4090 安全值）
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    分块 GPU cosine 相似度 top-k 检索。
    由于 embedding 已经 L2 归一化，cosine 相似度 = dot product。

    返回:
        scores  (N, top_k) CPU float32
        indices (N, top_k) CPU int64
    """
    all_scores: List[torch.Tensor] = []
    all_indices: List[torch.Tensor] = []

    for start in range(0, query_vecs.shape[0], query_chunk):
        chunk = query_vecs[start: start + query_chunk]       # (chunk, D)
        sim = torch.mm(chunk, key_vecs.T)                    # (chunk, M) float16
        scores, indices = sim.topk(top_k, dim=1, largest=True, sorted=True)
        all_scores.append(scores.cpu().float())
        all_indices.append(indices.cpu())

    return torch.cat(all_scores, dim=0), torch.cat(all_indices, dim=0)


# =============================================================================
# 匹配行构建（与本地版本输出列结构完全一致）
# =============================================================================

def build_unmatched_row(cfg: Config, embed_score: float = 0.0,
                        rerank_score: float = 0.0) -> Dict:
    return {
        "match_occupation": cfg.unmatched_label,
        "match_main_task": cfg.unmatched_label,
        "match_Level1_Code": "",
        "match_Level1_Name": "",
        "match_Level2_Code": "",
        "match_Level2_Name": "",
        "match_Level3_Code": "",
        "match_Level3_Name": "",
        "match_Level4_Code": "",
        "match_Level4_Name": "",
        "embed_score": embed_score,
        "rerank_score": rerank_score,
        "match_rank": 1,
        "is_match": cfg.is_match_no,
    }


def build_matched_row(cfg: Config, best: pd.Series, embed_score: float,
                      rerank_score: float = 0.0, match_rank: int = 1) -> Dict:
    return {
        "match_occupation": best.get(cfg.occupation_name_field, ""),
        "match_main_task": best.get(cfg.occupation_task_field, ""),
        "match_Level1_Code": best.get("Level1_Code", ""),
        "match_Level1_Name": best.get("Level1_Name", ""),
        "match_Level2_Code": best.get("Level2_Code", ""),
        "match_Level2_Name": best.get("Level2_Name", ""),
        "match_Level3_Code": best.get("Level3_Code", ""),
        "match_Level3_Name": best.get("Level3_Name", ""),
        "match_Level4_Code": best.get("Level4_Code", ""),
        "match_Level4_Name": best.get("Level4_Name", ""),
        "embed_score": embed_score,
        "rerank_score": rerank_score,
        "match_rank": match_rank,
        "is_match": cfg.is_match_yes,
    }


# =============================================================================
# 主处理流程
# =============================================================================

def load_occupation_table(cfg: Config, embedder: BGEEmbedder) -> Tuple[pd.DataFrame, torch.Tensor]:
    """
    加载职业分类大典，返回 occ_df 和 GPU 上的 L2 归一化向量。
    向量缓存命中时直接加载（新增 dtype 字段，避免精度错误）。
    """
    occ_df = pd.read_excel(cfg.occupation_path, sheet_name=cfg.occupation_sheet, dtype=str)

    for col in ["Level1_Code", "Level2_Code", "Level3_Code", "Level4_Code"]:
        if col in occ_df.columns:
            occ_df[col] = occ_df[col].map(normalize_level_code)

    occ_df["occupation_text"] = occ_df.apply(
        lambda r: build_occupation_text(r, cfg.occupation_name_field, cfg.occupation_task_field),
        axis=1,
    )

    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    occ_meta = {
        "occupation_path": str(cfg.occupation_path),
        "occupation_sheet": cfg.occupation_sheet,
        "embed_model": cfg.embed_model,
        "row_count": int(len(occ_df)),
        "dtype": cfg.model_dtype,    # 新增：精度不同则重建缓存
    }
    vec_path, meta_path = cache_paths(cache_dir, "occupation_vectors")
    occ_vectors_np = load_cached_vectors(vec_path, meta_path, occ_meta)

    if occ_vectors_np is None:
        print("职业向量缓存未命中，开始向量化...")
        occ_texts = occ_df["occupation_text"].fillna("").tolist()
        occ_vecs_gpu = embedder.embed_batch(occ_texts)   # (M, D) GPU float16
        occ_vectors_np = occ_vecs_gpu.cpu().numpy()
        save_cached_vectors(vec_path, meta_path, occ_meta, occ_vectors_np)
        print(f"✓ 职业向量已缓存: {vec_path}")
    else:
        print(f"✓ 职业向量缓存命中: {occ_vectors_np.shape}")
        occ_vecs_gpu = torch.from_numpy(occ_vectors_np).to(
            device=cfg.device,
            dtype=torch.float16 if cfg.model_dtype == "float16" else torch.float32,
        )

    return occ_df, occ_vecs_gpu


def process_chunk(
    chunk_df: pd.DataFrame,
    job_texts: List[str],
    embedder: BGEEmbedder,
    occ_vecs_gpu: torch.Tensor,
    occ_df: pd.DataFrame,
    cfg: Config,
    reranker: Optional[BGEReranker] = None,
) -> pd.DataFrame:
    """处理单个数据块：embed → topk → (可选 rerank) → build_match_rows → 合并"""

    n = len(job_texts)

    # 标记无效输入
    invalid_mask = [is_invalid_input(t) for t in job_texts]
    valid_indices = [i for i, inv in enumerate(invalid_mask) if not inv]

    # 仅对有效输入做嵌入，跳过空字符串和"无法识别"等无效输入
    if valid_indices:
        valid_texts = [job_texts[i] for i in valid_indices]
        valid_vecs_gpu = embedder.embed_batch(valid_texts)   # (V, D) GPU float16

        # GPU top-k 相似度（仅有效输入）
        valid_top_scores, valid_top_indices = gpu_cosine_topk(
            valid_vecs_gpu, occ_vecs_gpu, cfg.top_k, cfg.query_chunk_size
        )                                                    # (V, top_k) CPU

        # 映射回全量索引
        top_scores = torch.zeros(n, cfg.top_k)
        top_indices = torch.zeros(n, cfg.top_k, dtype=torch.long)
        for vi, gi in enumerate(valid_indices):
            top_scores[gi] = valid_top_scores[vi]
            top_indices[gi] = valid_top_indices[vi]
    else:
        top_scores = torch.zeros(n, cfg.top_k)
        top_indices = torch.zeros(n, cfg.top_k, dtype=torch.long)

    # ---- Reranker 精排（可选）----
    # 批量收集需要精排的 (query, candidate) 对，一次性打分
    rerank_results: Optional[Dict[int, List[float]]] = None
    if reranker is not None and cfg.use_rerank:
        all_pairs: List[Tuple[str, str]] = []
        pair_map: List[Tuple[int, int]] = []  # (row_idx, candidate_rank)

        for i in range(n):
            if invalid_mask[i]:
                continue
            best_embed_score = float(top_scores[i, 0])
            # 仅对 embed_score 超过阈值的行精排
            # 取 min_embed_score 和 rerank_min_embed_score 的较大值，
            # 避免对最终会被 min_embed_score 过滤掉的行做无用精排
            rerank_threshold = max(cfg.min_embed_score, cfg.rerank_min_embed_score)
            if best_embed_score < rerank_threshold:
                continue
            for k in range(cfg.top_k):
                cand_idx = int(top_indices[i, k])
                cand_text = occ_df.iloc[cand_idx]["occupation_text"]
                all_pairs.append((job_texts[i], cand_text))
                pair_map.append((i, k))

        if all_pairs:
            all_rerank_scores = reranker.compute_scores(all_pairs)
            rerank_results = {}
            for (row_i, cand_k), rs in zip(pair_map, all_rerank_scores):
                if row_i not in rerank_results:
                    rerank_results[row_i] = [0.0] * cfg.top_k
                rerank_results[row_i][cand_k] = rs

    # ---- 构建匹配行 ----
    match_rows = []
    for i in range(n):
        if invalid_mask[i]:
            match_rows.append(build_unmatched_row(cfg))
            continue

        best_embed_score = float(top_scores[i, 0])

        # min_embed_score 阈值：低于此分数直接标记为未匹配
        if best_embed_score < cfg.min_embed_score:
            match_rows.append(build_unmatched_row(cfg, best_embed_score))
            continue

        # 有 rerank 结果时，按 rerank_score 重排 top-k
        if rerank_results is not None and i in rerank_results:
            rr_scores = rerank_results[i]
            best_rr_rank = int(np.argmax(rr_scores))
            best_rr_score = rr_scores[best_rr_rank]

            # rerank_score 低于阈值 → 未匹配
            if best_rr_score < cfg.rerank_score_threshold:
                match_rows.append(build_unmatched_row(
                    cfg, best_embed_score, best_rr_score))
                continue

            best_idx = int(top_indices[i, best_rr_rank])
            best = occ_df.iloc[best_idx]
            embed_score_for_best = float(top_scores[i, best_rr_rank])

            occupation = best.get(cfg.occupation_name_field, "")
            is_valid = validate_suspicious_match(
                occupation, job_texts[i], embed_score_for_best,
                cfg.suspicious_occupations,
            )
            if not is_valid:
                match_rows.append(build_unmatched_row(
                    cfg, embed_score_for_best, best_rr_score))
            else:
                match_rows.append(build_matched_row(
                    cfg, best, embed_score_for_best,
                    rerank_score=best_rr_score,
                    match_rank=best_rr_rank + 1,
                ))
        else:
            # 无 rerank：直接取 embedding top-1
            best_idx = int(top_indices[i, 0])
            best = occ_df.iloc[best_idx]

            occupation = best.get(cfg.occupation_name_field, "")
            is_valid = validate_suspicious_match(
                occupation, job_texts[i], best_embed_score,
                cfg.suspicious_occupations,
            )
            if not is_valid:
                match_rows.append(build_unmatched_row(cfg, best_embed_score))
            else:
                match_rows.append(build_matched_row(cfg, best, best_embed_score))

    match_df = pd.DataFrame(match_rows)
    result_df = pd.concat(
        [chunk_df.reset_index(drop=True), match_df.reset_index(drop=True)],
        axis=1,
    )
    return result_df


def _normalize_schema(table: pa.Table) -> pa.Table:
    """将 dictionary 编码列转为基础类型，确保跨 chunk schema 一致。"""
    new_arrays, new_fields = [], []
    for i, f in enumerate(table.schema):
        col = table.column(i)
        if pa.types.is_dictionary(f.type):
            col = col.cast(f.type.value_type)
            f = pa.field(f.name, f.type.value_type)
        new_arrays.append(col)
        new_fields.append(f)
    return pa.table(new_arrays, schema=pa.schema(new_fields))


def _merge_chunks(chunk_dir: str, output_parquet: str, chunks_written: int,
                  logger_print=print):
    """将所有 chunk_*.parquet 合并为 data.parquet，自动处理 schema 不一致。"""
    chunk_files = sorted(Path(chunk_dir).glob("chunk_*.parquet"))
    if not chunk_files:
        logger_print(f"  未找到 chunk 文件: {chunk_dir}")
        return

    logger_print(f"  合并 {len(chunk_files)} 个 chunk 文件...")

    first_table = _normalize_schema(pq.ParquetFile(str(chunk_files[0])).read())
    writer = pq.ParquetWriter(output_parquet, first_table.schema, compression="snappy")
    writer.write_table(first_table)

    for cf in chunk_files[1:]:
        table = _normalize_schema(pq.ParquetFile(str(cf)).read())
        table = table.cast(first_table.schema)
        writer.write_table(table)

    writer.close()

    for cf in chunk_files:
        cf.unlink()

    logger_print(f"  ✓ 合并完成: {output_parquet}")


def process_year(year: str, cfg: Config, embedder: BGEEmbedder, occ_vecs_gpu: torch.Tensor,
                 occ_df: pd.DataFrame, reranker: Optional[BGEReranker] = None):
    """
    流式处理单个年份：断点续跑 + chunk 写盘 + 最终合并。

    断点恢复策略：
    - 基于 rows_done 行级精确跳过，不依赖 scanner batch 边界对齐
    - 用 iter_batches 流式读取 parquet，逐 batch 跳过已处理行，无需全量加载到内存
    - 即使 query_chunk_size 在两次运行之间改变，也能正确恢复
    """

    input_parquet  = cfg.input_parquet(year)
    chunk_dir      = cfg.output_chunk_dir(year)
    output_parquet = cfg.output_parquet(year)
    ckpt           = CheckpointManager(cfg.checkpoint_path(year))

    print(f"\n--- Step 3 | 年份: {year} ---")
    print(f"  输入: {input_parquet}")
    print(f"  输出: {output_parquet}")

    if ckpt.is_complete:
        print(f"  年份 {year} Step3 已完成，跳过")
        return

    if not Path(input_parquet).exists():
        raise FileNotFoundError(f"Step2 输出不存在: {input_parquet}")

    # 流式统计总行数（不加载全量数据到内存）
    total_rows = pq.ParquetFile(input_parquet).metadata.num_rows
    print(f"  总行数: {total_rows:,}")

    rows_done      = ckpt.rows_done
    chunks_written = ckpt.chunks_written

    if rows_done > 0:
        print(f"  从断点恢复: 已完成 {rows_done:,} 行 ({chunks_written} 个 chunk)")

    pbar = tqdm(total=total_rows, initial=rows_done,
                desc=f"Step3 {year}", unit="条", ncols=100) if tqdm else None

    try:
        # 流式读取 + 行级精确跳过：
        # 用 pyarrow RecordBatchReader 流式读取，按 query_chunk_size 分批
        # 跳过前 rows_done 行，无需全量加载到内存
        table_reader = pq.ParquetFile(input_parquet)
        row_cursor = 0  # 当前已扫描的行数（含跳过的）

        for batch in table_reader.iter_batches(batch_size=cfg.query_chunk_size):
            batch_len = batch.num_rows
            batch_end = row_cursor + batch_len

            if batch_end <= rows_done:
                # 整个 batch 都在已处理范围内，跳过
                row_cursor = batch_end
                continue

            chunk_df = batch.to_pandas()

            if row_cursor < rows_done:
                # 部分跳过：截掉 batch 前面已处理的行
                skip_n = rows_done - row_cursor
                chunk_df = chunk_df.iloc[skip_n:].copy()

            chunk_df = chunk_df.reset_index(drop=True)
            row_cursor = batch_end

            job_texts = (
                chunk_df[cfg.job_text_field].fillna("").astype(str).tolist()
                if cfg.job_text_field in chunk_df.columns
                else [""] * len(chunk_df)
            )

            result_df = process_chunk(
                chunk_df, job_texts, embedder, occ_vecs_gpu, occ_df, cfg,
                reranker=reranker,
            )
            table = _normalize_schema(pa.Table.from_pandas(result_df, preserve_index=False))

            chunk_file = Path(chunk_dir) / f"chunk_{chunks_written:06d}.parquet"
            pw = pq.ParquetWriter(str(chunk_file), table.schema, compression="snappy")
            pw.write_table(table)
            pw.close()

            chunks_written += 1
            rows_done      += len(chunk_df)
            ckpt.update(rows_done, chunks_written)

            if pbar:
                pbar.update(len(chunk_df))

    finally:
        if pbar:
            pbar.close()

    # 合并所有 chunk → data.parquet
    _merge_chunks(chunk_dir, output_parquet, chunks_written)
    ckpt.mark_complete()

    # 统计匹配率
    result_ds = ds.dataset(output_parquet, format="parquet")
    is_match_col = result_ds.to_table(columns=["is_match"]).to_pandas()
    matched = (is_match_col["is_match"] == cfg.is_match_yes).sum()
    print(f"\n✓ 年份 {year} 匹配完成")
    print(f"  总记录: {rows_done:,} | 匹配: {matched:,} | 未匹配: {rows_done - matched:,}")


# =============================================================================
# 命令行入口
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BGE-M3 向量匹配 - 云端 GPU 版")
    parser.add_argument("-c", "--config", required=True, help="配置文件路径")
    parser.add_argument("--year", required=True, help="处理年份，如 2021")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = Config.from_yaml(args.config)
    year = args.year
    logger = setup_logging(year)

    logger.info("=" * 60)
    logger.info(f"云端 Step 3 | 年份: {year}")
    logger.info(f"设备: {cfg.device} | 精度: {cfg.model_dtype} | Batch: {cfg.embedding_batch_size}")
    logger.info(f"min_embed_score: {cfg.min_embed_score} | use_rerank: {cfg.use_rerank}")
    logger.info("=" * 60)

    # 初始化 Embedder
    embedder = BGEEmbedder(
        model_path=cfg.embed_model,
        device=cfg.device,
        dtype=cfg.model_dtype,
        batch_size=cfg.embedding_batch_size,
    )

    # 可选：初始化 Reranker
    reranker = None
    if cfg.use_rerank:
        reranker = BGEReranker(
            model_path=cfg.rerank_model,
            device=cfg.device,
            normalize=cfg.rerank_normalize,
        )

    # 加载职业表 + 向量（含缓存）
    occ_df, occ_vecs_gpu = load_occupation_table(cfg, embedder)
    logger.info(f"职业分类大典: {len(occ_df)} 条细类职业")

    # 处理年份
    process_year(year, cfg, embedder, occ_vecs_gpu, occ_df, reranker=reranker)


if __name__ == "__main__":
    main()
