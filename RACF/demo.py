#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RACF import RACF


def demo_basic_usage():
    """
    演示1: 基本用法 - 基因数量不同、顺序不同的数据
    """
    print("\n" + "=" * 60)
    print("演示1: 基本用法")
    print("=" * 60)
    print("场景：")
    print("  - 数据库包含100个基因")
    print("  - Spot数据包含50个基因（数据库的子集）")
    print("  - 两者的基因顺序完全不同")
    print("=" * 60)

    demo_dir = os.path.join(os.path.dirname(__file__), "demo_data")
    db_path = os.path.join(demo_dir, "database.tsv")
    spot_path = os.path.join(demo_dir, "spot_data.tsv")
    db_gene_order_path = os.path.join(demo_dir, "database_gene_order.txt")
    spot_gene_order_path = os.path.join(demo_dir, "spot_gene_order.txt")

    print(f"\n数据库路径: {db_path}")
    print(f"Spot数据路径: {spot_path}")
    print(f"数据库基因顺序文件: {db_gene_order_path}")
    print(f"Spot基因顺序文件: {spot_gene_order_path}")

    recommender = RACF(
        database_path=db_path,
        database_gene_order=db_gene_order_path,
        k=5,
        n_candidates=100,
        n_precise=50
    )

    recommendations = recommender.run(
        spot_data=spot_path,
        spot_gene_order=spot_gene_order_path
    )

    print("\n推荐结果:")
    for spot_idx, rec_indices in recommendations.items():
        print(f"  Spot {spot_idx}: 推荐数据库样本 {rec_indices}")


def demo_with_final_expression():
    """
    演示2: 计算最终基因表达量
    """
    print("\n" + "=" * 60)
    print("演示2: 计算最终基因表达量")
    print("=" * 60)

    demo_dir = os.path.join(os.path.dirname(__file__), "demo_data")
    db_path = os.path.join(demo_dir, "database.tsv")
    spot_path = os.path.join(demo_dir, "spot_data.tsv")
    db_gene_order_path = os.path.join(demo_dir, "database_gene_order.txt")
    spot_gene_order_path = os.path.join(demo_dir, "spot_gene_order.txt")

    recommender = RACF(
        database_path=db_path,
        database_gene_order=db_gene_order_path,
        k=5,
        n_candidates=100,
        n_precise=50
    )

    recommendations, final_expressions = recommender.run(
        spot_path,
        spot_gene_order_path,
        calculate_final_expression=True
    )

    print(f"\n推荐结果:")
    print(f"  推荐数量: {len(recommendations)}")
    print(f"  最终表达量矩阵形状: {final_expressions.shape}")
    print(f"  (行数=spot数量, 列数=共同基因数量)")


def main():
    print("=" * 60)
    print("RACF 模块演示")
    print("=" * 60)
    print("场景设置:")
    print("  - 数据库: 1000个样本 x 100个基因")
    print("  - Spot: 10个样本 x 50个基因 (数据库的子集)")
    print("  - 数据库基因顺序: GENE_000, GENE_001, ..., GENE_099")
    print("  - Spot基因顺序: 打乱顺序的50个基因")
    print("=" * 60)

    print("\n[步骤1] 运行演示1: 基本用法...")
    demo_basic_usage()

    print("\n[步骤2] 运行演示2: 计算最终表达量...")
    demo_with_final_expression()

    print("\n" + "=" * 60)
    print("所有演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
