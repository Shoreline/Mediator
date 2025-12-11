#!/usr/bin/env python3
"""
伪随机选择器 - 确定性采样工具

用于对数据集进行可重复的下采样，支持：
1. 基于种子的确定性采样
2. 按类别独立采样（保持各类别比例）

使用示例：
    # 基础采样
    mask = generate_sample_mask(seed=42, data_size=109, sampling_rate=0.12)
    # 返回: [0, 0, 1, 0, ..., 1, 0]，共109个元素，其中13个为1
    
    # 按类别采样（用于MMSB数据集）
    sampled_records, stats = sample_by_category(
        records=all_records,
        seed=42,
        sampling_rate=0.5,
        category_field='category'
    )
"""

import random
from typing import List, Dict, Tuple, Any, Optional


def generate_sample_mask(seed: int, data_size: int, sampling_rate: float) -> List[int]:
    """
    生成确定性的采样掩码
    
    Args:
        seed: 随机种子（整数）
        data_size: 数据大小
        sampling_rate: 采样率 (0.0-1.0)
    
    Returns:
        长度为data_size的列表，包含0和1
        其中1的数量为 round(data_size * sampling_rate)
    
    Example:
        >>> mask = generate_sample_mask(42, 109, 0.12)
        >>> len(mask)
        109
        >>> sum(mask)
        13
        >>> # 相同种子产生相同结果
        >>> mask2 = generate_sample_mask(42, 109, 0.12)
        >>> mask == mask2
        True
    """
    if not 0.0 <= sampling_rate <= 1.0:
        raise ValueError(f"sampling_rate must be between 0.0 and 1.0, got {sampling_rate}")
    
    if data_size <= 0:
        raise ValueError(f"data_size must be positive, got {data_size}")
    
    # 创建确定性的随机数生成器
    rng = random.Random(seed)
    
    # 计算要选择的样本数
    sample_count = round(data_size * sampling_rate)
    
    # 生成所有索引并随机打乱
    indices = list(range(data_size))
    rng.shuffle(indices)
    
    # 选择前sample_count个索引
    selected_indices = set(indices[:sample_count])
    
    # 生成掩码
    mask = [1 if i in selected_indices else 0 for i in range(data_size)]
    
    return mask


def apply_mask_to_records(records: List[Any], mask: List[int]) -> List[Any]:
    """
    根据掩码过滤记录列表
    
    Args:
        records: 记录列表
        mask: 二进制掩码列表（0或1）
    
    Returns:
        过滤后的记录列表
    
    Raises:
        ValueError: 如果records和mask长度不匹配
    
    Example:
        >>> records = ['a', 'b', 'c', 'd', 'e']
        >>> mask = [1, 0, 1, 0, 1]
        >>> apply_mask_to_records(records, mask)
        ['a', 'c', 'e']
    """
    if len(records) != len(mask):
        raise ValueError(f"Length mismatch: records={len(records)}, mask={len(mask)}")
    
    return [record for record, selected in zip(records, mask) if selected == 1]


def sample_records(records: List[Any], seed: int, sampling_rate: float) -> List[Any]:
    """
    对记录列表进行采样
    
    Args:
        records: 记录列表
        seed: 随机种子
        sampling_rate: 采样率 (0.0-1.0)
    
    Returns:
        采样后的记录子集
    
    Example:
        >>> records = list(range(100))
        >>> sampled = sample_records(records, seed=42, sampling_rate=0.1)
        >>> len(sampled)
        10
    """
    # 处理空记录列表
    if len(records) == 0:
        return []
    
    if sampling_rate >= 1.0:
        return records.copy()
    
    if sampling_rate <= 0.0:
        return []
    
    mask = generate_sample_mask(seed, len(records), sampling_rate)
    return apply_mask_to_records(records, mask)


def sample_by_category(
    records: List[Dict[str, Any]], 
    seed: int, 
    sampling_rate: float,
    category_field: str = 'category'
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """
    按类别独立采样，确保每个类别保留相同比例
    
    Args:
        records: 记录列表，每个记录是包含类别字段的字典
        seed: 随机种子
        sampling_rate: 采样率 (0.0-1.0)
        category_field: 类别字段名（默认'category'）
    
    Returns:
        (sampled_records, stats)
        - sampled_records: 采样后的记录列表
        - stats: 每个类别的统计信息
          {
              'category1': {'original': 100, 'sampled': 12},
              'category2': {'original': 150, 'sampled': 18},
              ...
          }
    
    Example:
        >>> records = [
        ...     {'category': 'A', 'data': 1},
        ...     {'category': 'A', 'data': 2},
        ...     {'category': 'B', 'data': 3},
        ...     {'category': 'B', 'data': 4},
        ... ]
        >>> sampled, stats = sample_by_category(records, seed=42, sampling_rate=0.5)
        >>> stats['A']['sampled']
        1
        >>> stats['B']['sampled']
        1
    """
    if sampling_rate >= 1.0:
        # 不需要采样，返回原始数据和统计信息
        stats = {}
        for record in records:
            category = record.get(category_field, 'Unknown')
            if category not in stats:
                stats[category] = {'original': 0, 'sampled': 0}
            stats[category]['original'] += 1
            stats[category]['sampled'] += 1
        return records.copy(), stats
    
    if sampling_rate <= 0.0:
        # 采样率为0，返回空列表
        stats = {}
        for record in records:
            category = record.get(category_field, 'Unknown')
            if category not in stats:
                stats[category] = {'original': 0, 'sampled': 0}
            stats[category]['original'] += 1
        return [], stats
    
    # 按类别分组
    category_records = {}
    for record in records:
        category = record.get(category_field, 'Unknown')
        if category not in category_records:
            category_records[category] = []
        category_records[category].append(record)
    
    # 对每个类别独立采样
    sampled_records = []
    stats = {}
    
    for category, cat_records in sorted(category_records.items()):
        original_count = len(cat_records)
        
        # 为每个类别生成唯一的种子（基于原始种子和类别名）
        # 使用hash确保确定性
        category_seed = seed + hash(category) % 1000000
        
        # 生成该类别的采样掩码
        mask = generate_sample_mask(category_seed, original_count, sampling_rate)
        
        # 应用掩码
        sampled_cat_records = apply_mask_to_records(cat_records, mask)
        sampled_records.extend(sampled_cat_records)
        
        # 记录统计信息
        stats[category] = {
            'original': original_count,
            'sampled': len(sampled_cat_records)
        }
    
    return sampled_records, stats


def print_sampling_stats(stats: Dict[str, Dict[str, int]], sampling_rate: float):
    """
    打印采样统计信息
    
    Args:
        stats: sample_by_category返回的统计信息
        sampling_rate: 采样率
    """
    print(f"\n{'='*80}")
    print(f"📊 采样统计 (sampling_rate={sampling_rate:.2%})")
    print(f"{'='*80}")
    print(f"{'类别':<30} {'原始数量':<12} {'采样数量':<12} {'实际比例':<12}")
    print(f"{'-'*80}")
    
    total_original = 0
    total_sampled = 0
    
    for category in sorted(stats.keys()):
        original = stats[category]['original']
        sampled = stats[category]['sampled']
        actual_rate = (sampled / original * 100) if original > 0 else 0
        
        print(f"{category:<30} {original:<12} {sampled:<12} {actual_rate:.1f}%")
        
        total_original += original
        total_sampled += sampled
    
    print(f"{'-'*80}")
    overall_rate = (total_sampled / total_original * 100) if total_original > 0 else 0
    print(f"{'总计':<30} {total_original:<12} {total_sampled:<12} {overall_rate:.1f}%")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # 测试基础功能
    print("测试 1: 基础采样掩码生成")
    print("-" * 50)
    mask = generate_sample_mask(seed=42, data_size=109, sampling_rate=0.12)
    print(f"数据大小: 109")
    print(f"采样率: 0.12")
    print(f"采样数量: {sum(mask)} (期望: {round(109 * 0.12)})")
    
    # 测试确定性
    print("\n测试 2: 确定性验证")
    print("-" * 50)
    mask2 = generate_sample_mask(seed=42, data_size=109, sampling_rate=0.12)
    print(f"相同种子产生相同结果: {mask == mask2}")
    
    mask3 = generate_sample_mask(seed=99, data_size=109, sampling_rate=0.12)
    print(f"不同种子产生不同结果: {mask != mask3}")
    
    # 测试按类别采样
    print("\n测试 3: 按类别采样")
    print("-" * 50)
    test_records = []
    categories = ['A', 'B', 'C']
    for cat in categories:
        for i in range(100):
            test_records.append({'category': cat, 'index': i})
    
    sampled, stats = sample_by_category(test_records, seed=42, sampling_rate=0.12)
    print_sampling_stats(stats, 0.12)
    
    print(f"✅ 所有测试完成！")

