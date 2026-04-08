#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O*NET 映射工具 - 云端 DuckDB 版

改动说明（相比本地 add_onet_mapping_transformers.py）：
- 数据处理：pandas merge（全量加载）→ DuckDB SQL LEFT JOIN（流式，不占内存）
- 输出格式：Excel → Parquet（Hive 分区：year + level1）
- JOIN 成功率验证：join 前检查 match_Level4_Code 与 dadian_code 格式一致性
- 新增断点机制：完成后标记 checkpoint，避免重复执行

复用自本地版本（逻辑完全一致）：
- 两步 LEFT JOIN 结构（mapping 表 → O*NET 详情表）
- JOIN 键：left_on=match_Level4_Code, right_on=dadian_code
- 第二步 JOIN 键：onet_code → O*NET-SOC Code
- include_onet_details 控制是否附加 O*NET 详情

使用方式：
    python add_onet_mapping.py --config config.yaml --year 2021
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import duckdb
import yaml


# =============================================================================
# 配置
# =============================================================================

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # 环境变量替换路径前缀
    base_dir = os.environ.get("PIPELINE_BASE_DIR", "")
    if base_dir:
        for key in ("input_dir", "output_dir", "mapping_table_path", "onet_detail_path"):
            if key in cfg and isinstance(cfg[key], str):
                cfg[key] = cfg[key].replace("/root/autodl-tmp", base_dir)
    return cfg


def get_paths(cfg: dict, year: str):
    input_dir = Path(cfg["input_dir"])
    input_parquet = str(input_dir / f"year={year}" / "data.parquet")
    output_dir = str(Path(cfg["output_dir"]))   # DuckDB COPY 会自动创建子目录
    mapping_csv = cfg["mapping_table_path"]
    onet_detail_csv = cfg.get("onet_detail_path", "")
    dadian_code_field = cfg.get("dadian_code_field", "match_Level4_Code")
    include_onet_details = cfg.get("include_onet_details", True)
    return input_parquet, output_dir, mapping_csv, onet_detail_csv, dadian_code_field, include_onet_details


# =============================================================================
# 断点管理
# =============================================================================

class CheckpointManager:
    """简单断点：记录是否已完成，避免重复执行。"""

    def __init__(self, path: str):
        self._path = Path(path)
        self._data: dict = {"complete": False}
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                pass

    @property
    def is_complete(self) -> bool:
        return bool(self._data.get("complete", False))

    def mark_complete(self):
        self._data["complete"] = True
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self._path)


# =============================================================================
# 日志
# =============================================================================

def setup_logger(year: str) -> logging.Logger:
    logger = logging.getLogger(f"job2occ.step4_{year}")
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
    fh = logging.FileHandler(log_dir / f"step4_{year}.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


# =============================================================================
# DuckDB 处理主逻辑
# =============================================================================

def run_step4(
    year: str,
    input_parquet: str,
    output_dir: str,
    mapping_csv: str,
    onet_detail_csv: str,
    dadian_code_field: str,
    include_onet_details: bool,
    logger: logging.Logger,
):
    if not Path(input_parquet).exists():
        raise FileNotFoundError(f"Step3 输出不存在: {input_parquet}")

    logger.info(f"输入: {input_parquet}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"Mapping: {mapping_csv}")

    con = duckdb.connect(database=":memory:")
    con.execute("SET memory_limit='32GB'")
    con.execute("SET threads TO 16")

    # --- 注册视图（懒加载，不占内存）---
    con.execute(f"CREATE VIEW jobs AS SELECT * FROM read_parquet('{input_parquet}')")
    con.execute(f"CREATE VIEW mapping AS SELECT * FROM read_csv_auto('{mapping_csv}')")
    if include_onet_details and onet_detail_csv:
        con.execute(f"CREATE VIEW onet_detail AS SELECT * FROM read_csv_auto('{onet_detail_csv}')")

    # --- 验证 JOIN 成功率 ---
    _validate_join(con, dadian_code_field, logger)

    # --- 构建 SELECT 查询 ---
    if include_onet_details and onet_detail_csv:
        select_sql = f"""
            SELECT
                j.*,
                COALESCE(j.match_Level1_Name, '未匹配')  AS level1,
                COALESCE(j.match_Level1_Code, '00')      AS level1_code,
                m.onet_code,
                m.onet_title,
                m.similarity_score    AS onet_similarity_score,
                m.confidence          AS onet_confidence,
                od.Description        AS onet_description,
                od.All_Tasks          AS onet_tasks
            FROM jobs AS j
            LEFT JOIN mapping AS m
                ON j.{dadian_code_field} = m.dadian_code
            LEFT JOIN onet_detail AS od
                ON m.onet_code = od."O*NET-SOC Code"
        """
    else:
        select_sql = f"""
            SELECT
                j.*,
                COALESCE(j.match_Level1_Name, '未匹配')  AS level1,
                COALESCE(j.match_Level1_Code, '00')      AS level1_code,
                m.onet_code,
                m.onet_title,
                m.similarity_score    AS onet_similarity_score,
                m.confidence          AS onet_confidence
            FROM jobs AS j
            LEFT JOIN mapping AS m
                ON j.{dadian_code_field} = m.dadian_code
        """

    # --- 写出 Hive 分区 Parquet ---
    # 分区列：year + level1_code（match_Level1_Code 的短数字编码，如 "01"）
    # 注意：不用 level1（完整中文名），因 DuckDB URL 编码后超出 Linux 255字节文件名限制
    # 输出结构：output_dir/year=2021/level1_code=01/data.parquet
    logger.info("正在执行 DuckDB JOIN 并写出 Hive 分区 Parquet...")

    copy_sql = f"""
        COPY (
            {select_sql}
        )
        TO '{output_dir}'
        (
            FORMAT PARQUET,
            PARTITION_BY (year, level1_code),
            OVERWRITE_OR_IGNORE true,
            COMPRESSION SNAPPY
        )
    """
    con.execute(copy_sql)
    con.close()

    logger.info(f"✓ 年份 {year} 写出完成: {output_dir}/year={year}/level1_code=*/")


def _validate_join(con: duckdb.DuckDBPyConnection, dadian_code_field: str, logger: logging.Logger):
    """验证 JOIN 成功率，若失败率 > 5% 则警告（可能是 code 格式不一致）"""
    try:
        total = con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        if total == 0:
            logger.warning("jobs 表为空，跳过验证")
            return

        # 有 match_Level4_Code 但 join 失败的行数
        unmatched = con.execute(f"""
            SELECT COUNT(*) FROM jobs j
            LEFT JOIN mapping m ON j.{dadian_code_field} = m.dadian_code
            WHERE j.{dadian_code_field} IS NOT NULL
              AND j.{dadian_code_field} != ''
              AND j.{dadian_code_field} != '未匹配'
              AND m.dadian_code IS NULL
        """).fetchone()[0]

        rate = unmatched / total if total > 0 else 0
        logger.info(f"JOIN 验证: 总记录 {total:,} | JOIN 失败 {unmatched:,} ({rate:.1%})")

        if rate > 0.05:
            logger.warning(
                f"JOIN 失败率 {rate:.1%} > 5%！"
                f"请检查 match_Level4_Code 与 dadian_code 的格式是否一致（如零填充、连字符格式等）"
            )
    except Exception as e:
        logger.warning(f"JOIN 验证失败（可忽略）: {e}")


# =============================================================================
# 命令行入口
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="O*NET 映射 - 云端 DuckDB 版")
    parser.add_argument("-c", "--config", required=True, help="配置文件路径")
    parser.add_argument("--year", required=True, help="处理年份，如 2021")
    return parser.parse_args()


def main():
    args = parse_args()
    year = args.year
    cfg = load_config(args.config)
    logger = setup_logger(year)

    logger.info("=" * 60)
    logger.info(f"云端 Step 4 | 年份: {year}")
    logger.info("=" * 60)

    # 断点检查
    base_dir = os.environ.get("PIPELINE_BASE_DIR", "/root/autodl-tmp")
    ckpt_dir = Path(base_dir) / "data" / "output" / "checkpoints" / "step4"
    ckpt = CheckpointManager(str(ckpt_dir / f"step4_{year}.json"))

    if ckpt.is_complete:
        logger.info(f"年份 {year} Step4 已完成，跳过")
        return

    (
        input_parquet,
        output_dir,
        mapping_csv,
        onet_detail_csv,
        dadian_code_field,
        include_onet_details,
    ) = get_paths(cfg, year)

    run_step4(
        year=year,
        input_parquet=input_parquet,
        output_dir=output_dir,
        mapping_csv=mapping_csv,
        onet_detail_csv=onet_detail_csv,
        dadian_code_field=dadian_code_field,
        include_onet_details=include_onet_details,
        logger=logger,
    )

    ckpt.mark_complete()


if __name__ == "__main__":
    main()
