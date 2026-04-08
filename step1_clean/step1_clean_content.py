#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗工具 - Step1

功能：
- 按 rec_id 去重
- 剔除 position_responsibilities 有效字符数过短的记录
- 输出清洗后的 CSV + 统计报告

使用方式：
    # 通过 pipeline 调度（推荐）
    python step1_clean_content.py --config config.yaml --year 2021

    # 独立运行（兼容旧模式，处理所有年份）
    python step1_clean_content.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from pathlib import Path

import yaml

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# =============================================================================
# 日志
# =============================================================================

def setup_logging(year: str = "") -> logging.Logger:
    logger = logging.getLogger(f"job2occ.step1_{year}")
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
    fh = logging.FileHandler(log_dir / f"step1_{year}.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


# =============================================================================
# 工具函数
# =============================================================================

def effective_len(text):
    """计算"有效字符数"：去除空白后再计数"""
    if text is None:
        return 0
    return len(re.sub(r"\s+", "", text))


def count_rows_csv(in_path):
    """统计 CSV 数据行数（不含表头），用于 tqdm 显示总进度"""
    with open(in_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def extract_year(name):
    """从文件名中提取年份（如 2015/2024），找不到则返回 unknown"""
    match = re.search(r"(19|20)\d{2}", name)
    return match.group(0) if match else "unknown"


# =============================================================================
# 核心清洗逻辑
# =============================================================================

def clean_file(in_path: str, out_path: str, min_effective_len: int = 30) -> dict:
    """单文件清洗：去重 + 删除短描述，并写出新文件"""
    base = os.path.splitext(os.path.basename(in_path))[0]
    seen = set()
    total = 0
    kept = 0
    dropped_dup = 0
    dropped_short = 0
    total_rows = count_rows_csv(in_path)

    with open(in_path, "r", encoding="utf-8", errors="replace", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError(f"Missing header in {in_path}")

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
            writer.writeheader()

            iter_rows = reader
            if tqdm:
                iter_rows = tqdm(reader, total=total_rows, desc=base,
                                 unit="rows", miniters=1000)

            for row in iter_rows:
                total += 1
                rec_id = row.get("rec_id", "")
                if rec_id in seen:
                    dropped_dup += 1
                    continue
                seen.add(rec_id)

                text = row.get("position_responsibilities", "")
                if effective_len(text) < min_effective_len:
                    dropped_short += 1
                    continue

                writer.writerow(row)
                kept += 1

    print(
        f"{os.path.basename(in_path)} -> {os.path.basename(out_path)} | "
        f"kept={kept}, dropped_dup={dropped_dup}, dropped_short={dropped_short}"
    )
    return {
        "year": extract_year(base),
        "total": total,
        "kept": kept,
        "dropped_dup": dropped_dup,
        "dropped_short": dropped_short,
    }


def write_stats(stats: list, stats_path: str):
    """输出每年清洗统计到 CSV"""
    file_exists = os.path.exists(stats_path)
    with open(stats_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "year", "dropped_total", "kept",
                "dropped_dup", "dropped_short", "total",
            ])
        for item in stats:
            dropped_total = item["dropped_dup"] + item["dropped_short"]
            writer.writerow([
                item["year"], dropped_total, item["kept"],
                item["dropped_dup"], item["dropped_short"], item["total"],
            ])


# =============================================================================
# 命令行入口
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="数据清洗 - Step1")
    parser.add_argument("-c", "--config", help="配置文件路径（pipeline 模式）")
    parser.add_argument("--year", help="处理年份，如 2021（pipeline 模式）")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Pipeline 模式：通过 --config + --year 运行 ----
    if args.config and args.year:
        year = args.year
        logger = setup_logging(year)

        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # 环境变量替换路径前缀
        base_dir = os.environ.get("PIPELINE_BASE_DIR", "")
        input_dir = cfg.get("input_dir", ".")
        output_dir = cfg.get("output_dir", ".")
        if base_dir:
            input_dir = input_dir.replace("/root/autodl-tmp", base_dir)
            output_dir = output_dir.replace("/root/autodl-tmp", base_dir)

        min_len = cfg.get("min_effective_len", 30)
        in_template = cfg.get("input_filename_template", "recruitment_infos_content_{year}.csv")
        out_template = cfg.get("output_filename_template",
                               "recruitment_infos_content_{year}_step1_clean.csv")

        in_path = os.path.join(input_dir, in_template.format(year=year))
        out_path = os.path.join(output_dir, out_template.format(year=year))

        if not os.path.exists(in_path):
            raise FileNotFoundError(f"输入文件不存在: {in_path}")

        logger.info("=" * 60)
        logger.info(f"云端 Step 1 | 年份: {year}")
        logger.info(f"输入: {in_path}")
        logger.info(f"输出: {out_path}")
        logger.info("=" * 60)

        stats = clean_file(in_path, out_path, min_len)

        stats_path = os.path.join(output_dir, "step1_clean_counts.csv")
        write_stats([stats], stats_path)

        logger.info(
            f"✓ 年份 {year} 清洗完成: "
            f"保留 {stats['kept']:,} | 去重 {stats['dropped_dup']:,} | "
            f"过短 {stats['dropped_short']:,}"
        )
        return

    # ---- 独立模式（兼容旧用法）：处理当前目录下所有年份 ----
    import glob as glob_mod

    out_dir = "step1_clean"
    os.makedirs(out_dir, exist_ok=True)

    inputs = sorted(glob_mod.glob("recruitment_infos_content_*.csv"))
    if not inputs:
        print("No input files matched recruitment_infos_content_*.csv")
        return

    stats = []
    for path in inputs:
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, f"{base}_step1_clean.csv")
        stats.append(clean_file(path, out_path))

    stats_path = os.path.join(out_dir, "step1_clean_counts.csv")
    write_stats(stats, stats_path)


if __name__ == "__main__":
    main()
