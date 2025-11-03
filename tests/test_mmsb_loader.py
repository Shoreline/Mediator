"""
测试 MM-SafetyBench 数据加载器
用于验证 load_mm_safety_items() 是否正确加载 MM-SafetyBench 数据

============================================================
📖 使用方法
============================================================

1. 测试数据加载（使用 SD + Changed Question，最常用的配对）
   python test_mmsb_loader.py

2. 测试不同的图片类型（问题字段会自动匹配）
   python test_mmsb_loader.py --image_type SD_TYPO
   python test_mmsb_loader.py --image_type TYPO

3. 显示更多样本
   python test_mmsb_loader.py --max_display 10

4. 使用自定义路径
   python test_mmsb_loader.py \
     --json_glob "~/code/MM-SafetyBench/data/processed_questions/*.json" \
     --image_base "~/Downloads/MM-SafetyBench_imgs/"

5. 验证配对关系（spot check）
   python test_mmsb_loader.py --test_pairing --num_samples 10

============================================================
🔍 MM-SafetyBench 配对关系
============================================================
- SD        → Changed Question           (修改后的问题)
- SD_TYPO   → Rephrased Question         (改写问题，引用图片底部)
- TYPO      → Rephrased Question(SD)     (SD版本改写问题)

============================================================
"""

import os
import sys
import json
import glob
import random

# 添加项目根目录到路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from request import load_mm_safety_items, Item, MMSB_IMAGE_QUESTION_MAP

def test_load_data(
    json_pattern: str, 
    image_base: str, 
    image_type: str = "SD",
    max_display: int = 5
):
    """
    测试数据加载功能
    
    Args:
        json_pattern: JSON 文件的 glob 模式
        image_base: 图片基础路径
        image_type: 图片类型（SD/SD_TYPO/TYPO），自动匹配对应的 question_field
        max_display: 最多显示多少条数据
    """
    # 从映射表获取对应的 question_field
    question_field = MMSB_IMAGE_QUESTION_MAP[image_type]
    
    print("=" * 70)
    print("📦 测试 MM-SafetyBench 数据加载器")
    print("=" * 70)
    print(f"JSON 文件模式: {json_pattern}")
    print(f"图片基础路径: {image_base}")
    print(f"图片类型:     {image_type}")
    print(f"问题字段:     {question_field} (自动匹配)")
    print()
    
    try:
        # 加载数据
        print("⏳ 正在加载数据...")
        items = list(load_mm_safety_items(json_pattern, image_base, image_type))
        
        print(f"✅ 成功加载 {len(items)} 条数据\n")
        
        # 显示前几条数据
        print(f"📋 显示前 {min(max_display, len(items))} 条数据:")
        print("-" * 60)
        
        for i, item in enumerate(items[:max_display]):
            print(f"\n[{i+1}] 数据项:")
            print(f"  Index:    {item.index}")
            print(f"  Category: {item.category}")
            print(f"  Question: {item.question}")
            print(f"  Image:    {item.image_path}")
            
            # 检查图片是否存在
            if os.path.exists(item.image_path):
                size = os.path.getsize(item.image_path)
                print(f"  ✅ 图片存在 ({size:,} bytes)")
            else:
                print(f"  ❌ 图片不存在!")
        
        # 统计信息
        print("\n" + "=" * 60)
        print("📊 统计信息")
        print("=" * 60)
        print(f"总数据量: {len(items)}")
        
        # 按类别统计
        categories = {}
        for item in items:
            categories[item.category] = categories.get(item.category, 0) + 1
        
        print(f"\n按类别分布:")
        for category, count in sorted(categories.items()):
            print(f"  {category}: {count}")
        
        # 检查图片存在性
        existing_images = sum(1 for item in items if os.path.exists(item.image_path))
        missing_images = len(items) - existing_images
        print(f"\n图片检查:")
        print(f"  存在: {existing_images}")
        print(f"  缺失: {missing_images}")
        
        if missing_images > 0:
            print(f"\n⚠️  警告: 有 {missing_images} 张图片缺失!")
            print("缺失的图片路径（前10个）:")
            count = 0
            for item in items:
                if not os.path.exists(item.image_path) and count < 10:
                    print(f"  - {item.image_path}")
                    count += 1
        
        print("\n" + "=" * 60)
        print("✅ 测试完成!")
        print("=" * 60)
        
        return items
        
    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到文件")
        print(f"   {e}")
        return None
    except Exception as e:
        print(f"❌ 错误: {type(e).__name__}")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return None

def test_mmsafety_pairing(
    json_pattern: str = "~/code/MM-SafetyBench/data/processed_questions/*.json",
    image_base_dir: str = "~/Downloads/MM-SafetyBench_imgs/",
    num_samples: int = 10
):
    """
    测试 MM-SafetyBench 的图片类型和问题文本配对
    通过调用 load_mm_safety_items() 三次来验证三种配对关系
    
    验证配对关系：
    - SD → Changed Question
    - SD_TYPO → Rephrased Question
    - TYPO → Rephrased Question(SD)
    
    Args:
        json_pattern: JSON 文件的 glob 模式
        image_base_dir: 图片基础目录
        num_samples: 抽查样本数量
    """
    print("=" * 70)
    print("🔍 MM-SafetyBench 配对关系验证（使用 load_mm_safety_items）")
    print("=" * 70)
    print("验证配对:")
    print("  SD        → Changed Question")
    print("  SD_TYPO   → Rephrased Question")
    print("  TYPO      → Rephrased Question(SD)")
    print()
    
    # 配对关系
    pairings = [
        ("SD", "Changed Question"),
        ("SD_TYPO", "Rephrased Question"),
        ("TYPO", "Rephrased Question(SD)")
    ]
    
    try:
        # 使用 load_mm_safety_items 加载三种配对的数据
        all_datasets = {}
        for img_type, question_field in pairings:
            print(f"⏳ 加载 {img_type} + {question_field}...")
            items = list(load_mm_safety_items(json_pattern, image_base_dir, img_type))
            all_datasets[(img_type, question_field)] = items
            print(f"   ✅ 成功加载 {len(items)} 条数据")
        
        print()
        
        # 验证三个数据集的数量应该相同
        dataset_sizes = [len(items) for items in all_datasets.values()]
        if len(set(dataset_sizes)) != 1:
            print(f"⚠️  警告: 三个数据集大小不一致: {dataset_sizes}")
        else:
            print(f"📊 三个数据集大小一致: {dataset_sizes[0]} 条")
        
        # 随机抽取样本进行验证
        first_dataset = list(all_datasets.values())[0]
        num_samples = min(num_samples, len(first_dataset))
        
        # 随机选择索引
        sample_indices = random.sample(range(len(first_dataset)), num_samples)
        print(f"🎲 随机抽取 {num_samples} 个样本进行验证\n")
        
        # 验证每个样本
        all_pass = True
        for i, idx in enumerate(sample_indices, 1):
            # 获取三个数据集中相同索引的 item
            items_dict = {}
            for (img_type, question_field), items in all_datasets.items():
                items_dict[img_type] = items[idx]
            
            # 获取第一个 item 作为参考
            ref_item = items_dict["SD"]
            
            print(f"[{i}/{num_samples}] 验证: {ref_item.category} / index {ref_item.index}")
            print("-" * 70)
            
            # 检查每种配对
            for img_type, question_field in pairings:
                item = items_dict[img_type]
                
                # 验证 category 和 index 一致
                if item.category != ref_item.category or item.index != ref_item.index:
                    print(f"  ❌ {img_type:10s} - 数据不一致!")
                    all_pass = False
                    continue
                
                # 检查图片是否存在
                img_exists = os.path.exists(item.image_path)
                
                # 检查问题是否为空
                question_valid = bool(item.question.strip())
                
                # 显示结果
                status = "✅" if (img_exists and question_valid) else "❌"
                print(f"  {status} {img_type:10s} + {question_field:25s}")
                print(f"     图片: {'存在' if img_exists else '缺失'} - {item.image_path}")
                print(f"     问题: {item.question}")
                
                if not img_exists:
                    print(f"     ⚠️  图片文件不存在!")
                    all_pass = False
                
                if not question_valid:
                    print(f"     ⚠️  问题文本为空!")
                    all_pass = False
            
            print()
        
        # 总结
        print("=" * 70)
        if all_pass:
            print("✅ 所有抽查样本配对验证通过!")
        else:
            print("❌ 部分样本配对验证失败，请检查!")
        print("=" * 70)
        
        return all_pass
        
    except Exception as e:
        print(f"❌ 错误: {type(e).__name__}")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 MM-SafetyBench 数据加载器")
    parser.add_argument("--json_glob", 
                       default="~/code/MM-SafetyBench/data/processed_questions/*.json",
                       help="JSON 文件的 glob 模式")
    parser.add_argument("--image_base", 
                       default="~/Downloads/MM-SafetyBench_imgs/",
                       help="图片基础目录")
    parser.add_argument("--image_type",
                       default="SD",
                       choices=["SD", "SD_TYPO", "TYPO"],
                       help="图片类型（问题字段会自动匹配）")
    parser.add_argument("--max_display", type=int, default=5,
                       help="最多显示多少条数据")
    parser.add_argument("--test_pairing", action="store_true",
                       help="验证配对关系（spot check）")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="配对验证的抽查样本数量")
    
    args = parser.parse_args()
    
    if args.test_pairing:
        # 验证配对关系
        test_mmsafety_pairing(
            args.json_glob,
            args.image_base, 
            args.num_samples
        )
    else:
        # 测试数据加载
        test_load_data(
            args.json_glob, 
            args.image_base,
            args.image_type,
            args.max_display
        )

