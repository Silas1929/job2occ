#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parquet 分区合并工具

将 Step4 输出的 Hive 分区 Parquet 合并为单个年度文件。

使用方式：
    # 合并单个年份
    python merge.py --year 2021

    # 合并全部年份
    python merge.py --all-years

    # 指定自定义路径
    PIPELINE_BASE_DIR=/data python merge.py --year 2021
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import duckdb


BASE_DIR = os.environ.get("PIPELINE_BASE_DIR", "/root/autodl-tmp")
FINAL_DIR = os.path.join(BASE_DIR, "data", "output", "final")
MERGED_DIR = os.path.join(BASE_DIR, "data", "output")

YEARS = list(range(2015, 2025))


def merge_year(year: int):
    """合并单个年份的分区 Parquet"""
    pattern = os.path.join(FINAL_DIR, f"year={year}", "level1_code=*", "*.parquet")
    output_path = os.path.join(MERGED_DIR, f"{year}_final.parquet")

    # 检查输入是否存在
    input_dir = os.path.join(FINAL_DIR, f"year={year}")
    if not Path(input_dir).exists():
        print(f"跳过 {year}: 目录不存在 {input_dir}")
        return

    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{pattern}')
        )
        TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)
    n = con.execute(
        f"SELECT count(*) FROM read_parquet('{output_path}')"
    ).fetchone()[0]
    con.close()

    print(f"✓ {year}: {n:,} rows → {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parquet 分区合并工具")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--year", type=int, choices=YEARS, help="合并单个年份")
    group.add_argument("--all-years", action="store_true", help="合并全部年份")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.year:
        merge_year(args.year)
    else:
        for year in YEARS:
            merge_year(year)

    print("\n✓ 合并完成")


if __name__ == "__main__":
    main()
