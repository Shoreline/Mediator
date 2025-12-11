#!/usr/bin/env python3
"""
伪随机采样器单元测试

测试 pseudo_random_sampler.py 中的所有函数功能：
- generate_sample_mask: 采样掩码生成
- apply_mask_to_records: 掩码应用
- sample_records: 记录采样
- sample_by_category: 按类别采样
- print_sampling_stats: 统计信息打印
"""

import unittest
import sys
import os
from io import StringIO

# 添加父目录到路径以导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pseudo_random_sampler import (
    generate_sample_mask,
    apply_mask_to_records,
    sample_records,
    sample_by_category,
    print_sampling_stats
)


class TestGenerateSampleMask(unittest.TestCase):
    """测试 generate_sample_mask 函数"""
    
    def test_basic_functionality(self):
        """测试基本功能"""
        mask = generate_sample_mask(seed=42, data_size=109, sampling_rate=0.12)
        
        # 检查长度
        self.assertEqual(len(mask), 109)
        
        # 检查采样数量
        expected_count = round(109 * 0.12)  # 13
        self.assertEqual(sum(mask), expected_count)
        
        # 检查只包含0和1
        self.assertTrue(all(x in [0, 1] for x in mask))
    
    def test_determinism(self):
        """测试确定性 - 相同种子产生相同结果"""
        mask1 = generate_sample_mask(seed=42, data_size=109, sampling_rate=0.12)
        mask2 = generate_sample_mask(seed=42, data_size=109, sampling_rate=0.12)
        
        self.assertEqual(mask1, mask2)
    
    def test_different_seeds(self):
        """测试不同种子产生不同结果"""
        mask1 = generate_sample_mask(seed=42, data_size=109, sampling_rate=0.12)
        mask2 = generate_sample_mask(seed=99, data_size=109, sampling_rate=0.12)
        
        # 不同种子应该产生不同结果（概率极高）
        self.assertNotEqual(mask1, mask2)
    
    def test_sampling_rate_zero(self):
        """测试采样率为0"""
        mask = generate_sample_mask(seed=42, data_size=100, sampling_rate=0.0)
        
        self.assertEqual(len(mask), 100)
        self.assertEqual(sum(mask), 0)
    
    def test_sampling_rate_one(self):
        """测试采样率为1.0"""
        mask = generate_sample_mask(seed=42, data_size=100, sampling_rate=1.0)
        
        self.assertEqual(len(mask), 100)
        self.assertEqual(sum(mask), 100)
    
    def test_various_sampling_rates(self):
        """测试各种采样率"""
        data_size = 1000
        test_rates = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        
        for rate in test_rates:
            with self.subTest(rate=rate):
                mask = generate_sample_mask(seed=42, data_size=data_size, sampling_rate=rate)
                expected_count = round(data_size * rate)
                self.assertEqual(sum(mask), expected_count, 
                               f"Rate {rate}: expected {expected_count}, got {sum(mask)}")
    
    def test_invalid_sampling_rate_negative(self):
        """测试无效的负采样率"""
        with self.assertRaises(ValueError):
            generate_sample_mask(seed=42, data_size=100, sampling_rate=-0.1)
    
    def test_invalid_sampling_rate_too_large(self):
        """测试无效的过大采样率"""
        with self.assertRaises(ValueError):
            generate_sample_mask(seed=42, data_size=100, sampling_rate=1.5)
    
    def test_invalid_data_size_zero(self):
        """测试无效的数据大小（0）"""
        with self.assertRaises(ValueError):
            generate_sample_mask(seed=42, data_size=0, sampling_rate=0.5)
    
    def test_invalid_data_size_negative(self):
        """测试无效的数据大小（负数）"""
        with self.assertRaises(ValueError):
            generate_sample_mask(seed=42, data_size=-10, sampling_rate=0.5)
    
    def test_small_data_sizes(self):
        """测试小数据集"""
        for size in [1, 2, 5, 10]:
            with self.subTest(size=size):
                mask = generate_sample_mask(seed=42, data_size=size, sampling_rate=0.5)
                self.assertEqual(len(mask), size)
                expected = round(size * 0.5)
                self.assertEqual(sum(mask), expected)


class TestApplyMaskToRecords(unittest.TestCase):
    """测试 apply_mask_to_records 函数"""
    
    def test_basic_functionality(self):
        """测试基本功能"""
        records = ['a', 'b', 'c', 'd', 'e']
        mask = [1, 0, 1, 0, 1]
        
        result = apply_mask_to_records(records, mask)
        
        self.assertEqual(result, ['a', 'c', 'e'])
    
    def test_all_selected(self):
        """测试全部选中"""
        records = [1, 2, 3, 4, 5]
        mask = [1, 1, 1, 1, 1]
        
        result = apply_mask_to_records(records, mask)
        
        self.assertEqual(result, records)
    
    def test_none_selected(self):
        """测试全部不选"""
        records = [1, 2, 3, 4, 5]
        mask = [0, 0, 0, 0, 0]
        
        result = apply_mask_to_records(records, mask)
        
        self.assertEqual(result, [])
    
    def test_empty_records(self):
        """测试空记录列表"""
        records = []
        mask = []
        
        result = apply_mask_to_records(records, mask)
        
        self.assertEqual(result, [])
    
    def test_dict_records(self):
        """测试字典类型记录"""
        records = [
            {'id': 1, 'name': 'A'},
            {'id': 2, 'name': 'B'},
            {'id': 3, 'name': 'C'},
        ]
        mask = [1, 0, 1]
        
        result = apply_mask_to_records(records, mask)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['id'], 1)
        self.assertEqual(result[1]['id'], 3)
    
    def test_length_mismatch(self):
        """测试长度不匹配"""
        records = [1, 2, 3]
        mask = [1, 0]
        
        with self.assertRaises(ValueError) as context:
            apply_mask_to_records(records, mask)
        
        self.assertIn("Length mismatch", str(context.exception))


class TestSampleRecords(unittest.TestCase):
    """测试 sample_records 函数"""
    
    def test_basic_sampling(self):
        """测试基本采样"""
        records = list(range(100))
        sampled = sample_records(records, seed=42, sampling_rate=0.1)
        
        # 检查采样数量
        self.assertEqual(len(sampled), 10)
        
        # 检查所有元素都来自原始列表
        self.assertTrue(all(x in records for x in sampled))
    
    def test_determinism(self):
        """测试确定性"""
        records = list(range(100))
        
        sampled1 = sample_records(records, seed=42, sampling_rate=0.2)
        sampled2 = sample_records(records, seed=42, sampling_rate=0.2)
        
        self.assertEqual(sampled1, sampled2)
    
    def test_sampling_rate_zero(self):
        """测试采样率为0"""
        records = list(range(100))
        sampled = sample_records(records, seed=42, sampling_rate=0.0)
        
        self.assertEqual(len(sampled), 0)
    
    def test_sampling_rate_one(self):
        """测试采样率为1.0"""
        records = list(range(100))
        sampled = sample_records(records, seed=42, sampling_rate=1.0)
        
        self.assertEqual(len(sampled), len(records))
        # 应该返回副本，不是原列表
        self.assertIsNot(sampled, records)
    
    def test_sampling_rate_greater_than_one(self):
        """测试采样率大于1.0"""
        records = list(range(100))
        sampled = sample_records(records, seed=42, sampling_rate=1.5)
        
        # 应该返回全部
        self.assertEqual(len(sampled), len(records))
    
    def test_empty_records(self):
        """测试空记录列表"""
        records = []
        sampled = sample_records(records, seed=42, sampling_rate=0.5)
        
        self.assertEqual(len(sampled), 0)
    
    def test_string_records(self):
        """测试字符串记录"""
        records = ['apple', 'banana', 'cherry', 'date', 'elderberry']
        sampled = sample_records(records, seed=42, sampling_rate=0.6)
        
        expected_count = round(len(records) * 0.6)
        self.assertEqual(len(sampled), expected_count)
        self.assertTrue(all(x in records for x in sampled))


class TestSampleByCategory(unittest.TestCase):
    """测试 sample_by_category 函数"""
    
    def setUp(self):
        """设置测试数据"""
        self.test_records = []
        categories = ['A', 'B', 'C']
        for cat in categories:
            for i in range(100):
                self.test_records.append({
                    'category': cat,
                    'index': i,
                    'data': f'{cat}_{i}'
                })
    
    def test_basic_functionality(self):
        """测试基本功能"""
        sampled, stats = sample_by_category(
            self.test_records, 
            seed=42, 
            sampling_rate=0.12
        )
        
        # 检查统计信息
        self.assertEqual(len(stats), 3)
        self.assertIn('A', stats)
        self.assertIn('B', stats)
        self.assertIn('C', stats)
        
        # 每个类别原始数量应该是100
        for cat in ['A', 'B', 'C']:
            self.assertEqual(stats[cat]['original'], 100)
            # 采样数量应该接近12
            self.assertEqual(stats[cat]['sampled'], round(100 * 0.12))
    
    def test_determinism(self):
        """测试确定性"""
        sampled1, stats1 = sample_by_category(
            self.test_records, seed=42, sampling_rate=0.2
        )
        sampled2, stats2 = sample_by_category(
            self.test_records, seed=42, sampling_rate=0.2
        )
        
        self.assertEqual(sampled1, sampled2)
        self.assertEqual(stats1, stats2)
    
    def test_different_seeds(self):
        """测试不同种子"""
        sampled1, _ = sample_by_category(
            self.test_records, seed=42, sampling_rate=0.2
        )
        sampled2, _ = sample_by_category(
            self.test_records, seed=99, sampling_rate=0.2
        )
        
        # 不同种子应该产生不同结果
        self.assertNotEqual(sampled1, sampled2)
    
    def test_category_proportions(self):
        """测试每个类别保持相同比例"""
        sampled, stats = sample_by_category(
            self.test_records, seed=42, sampling_rate=0.15
        )
        
        # 检查每个类别的采样比例应该接近设定值
        for cat in ['A', 'B', 'C']:
            original = stats[cat]['original']
            sampled_count = stats[cat]['sampled']
            expected = round(original * 0.15)
            self.assertEqual(sampled_count, expected)
    
    def test_sampling_rate_zero(self):
        """测试采样率为0"""
        sampled, stats = sample_by_category(
            self.test_records, seed=42, sampling_rate=0.0
        )
        
        self.assertEqual(len(sampled), 0)
        
        # 统计信息应该显示原始数量但采样为0
        for cat in ['A', 'B', 'C']:
            self.assertEqual(stats[cat]['original'], 100)
            self.assertEqual(stats[cat]['sampled'], 0)
    
    def test_sampling_rate_one(self):
        """测试采样率为1.0"""
        sampled, stats = sample_by_category(
            self.test_records, seed=42, sampling_rate=1.0
        )
        
        self.assertEqual(len(sampled), len(self.test_records))
        
        # 统计信息应该显示全部保留
        for cat in ['A', 'B', 'C']:
            self.assertEqual(stats[cat]['original'], 100)
            self.assertEqual(stats[cat]['sampled'], 100)
    
    def test_custom_category_field(self):
        """测试自定义类别字段"""
        records = [
            {'type': 'X', 'value': 1},
            {'type': 'X', 'value': 2},
            {'type': 'Y', 'value': 3},
            {'type': 'Y', 'value': 4},
        ]
        
        sampled, stats = sample_by_category(
            records, seed=42, sampling_rate=0.5, category_field='type'
        )
        
        self.assertIn('X', stats)
        self.assertIn('Y', stats)
        self.assertEqual(stats['X']['original'], 2)
        self.assertEqual(stats['Y']['original'], 2)
    
    def test_missing_category_field(self):
        """测试缺失类别字段"""
        records = [
            {'data': 1},
            {'data': 2},
            {'category': 'A', 'data': 3},
        ]
        
        sampled, stats = sample_by_category(
            records, seed=42, sampling_rate=0.5
        )
        
        # 缺失类别字段的记录应该归入'Unknown'
        self.assertIn('Unknown', stats)
        self.assertEqual(stats['Unknown']['original'], 2)
    
    def test_uneven_category_sizes(self):
        """测试不均匀的类别大小"""
        records = []
        # 类别A: 10条
        for i in range(10):
            records.append({'category': 'A', 'index': i})
        # 类别B: 50条
        for i in range(50):
            records.append({'category': 'B', 'index': i})
        # 类别C: 200条
        for i in range(200):
            records.append({'category': 'C', 'index': i})
        
        sampled, stats = sample_by_category(
            records, seed=42, sampling_rate=0.2
        )
        
        # 验证每个类别独立采样
        self.assertEqual(stats['A']['sampled'], round(10 * 0.2))   # 2
        self.assertEqual(stats['B']['sampled'], round(50 * 0.2))   # 10
        self.assertEqual(stats['C']['sampled'], round(200 * 0.2))  # 40
    
    def test_single_category(self):
        """测试单一类别"""
        records = [{'category': 'A', 'index': i} for i in range(100)]
        
        sampled, stats = sample_by_category(
            records, seed=42, sampling_rate=0.15
        )
        
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats['A']['original'], 100)
        self.assertEqual(stats['A']['sampled'], 15)
    
    def test_many_categories(self):
        """测试多个类别"""
        records = []
        for cat_idx in range(13):  # MMSB有13个类别
            cat_name = f'Category_{cat_idx}'
            for i in range(100):
                records.append({'category': cat_name, 'index': i})
        
        sampled, stats = sample_by_category(
            records, seed=42, sampling_rate=0.12
        )
        
        self.assertEqual(len(stats), 13)
        
        # 验证每个类别都被正确采样
        for cat_idx in range(13):
            cat_name = f'Category_{cat_idx}'
            self.assertEqual(stats[cat_name]['original'], 100)
            self.assertEqual(stats[cat_name]['sampled'], 12)
    
    def test_different_categories_different_masks(self):
        """测试不同类别使用相同数据集但产生不同的采样mask
        
        关键要求：
        1. 所有类别应该有相同的 sampling rate
        2. 但是被选中的具体元素（mask中'1'的位置）必须不同
        3. 即使数据集完全相同
        """
        # 创建3个类别，每个类别都有完全相同的数据集
        # 这模拟了每个类别都有相同数量和内容的场景
        identical_data = list(range(100))
        records = []
        
        for category in ['A', 'B', 'C']:
            for value in identical_data:
                records.append({
                    'category': category,
                    'value': value,
                    'data': f'item_{value}'  # 完全相同的数据
                })
        
        # 执行按类别采样
        sampled, stats = sample_by_category(
            records, seed=42, sampling_rate=0.2
        )
        
        # 1. 验证每个类别的采样率相同
        expected_count = round(100 * 0.2)  # 20
        for cat in ['A', 'B', 'C']:
            self.assertEqual(stats[cat]['original'], 100)
            self.assertEqual(stats[cat]['sampled'], expected_count,
                           f"Category {cat} should have {expected_count} samples")
        
        # 2. 提取每个类别被选中的元素位置
        sampled_values_A = set()
        sampled_values_B = set()
        sampled_values_C = set()
        
        for record in sampled:
            if record['category'] == 'A':
                sampled_values_A.add(record['value'])
            elif record['category'] == 'B':
                sampled_values_B.add(record['value'])
            elif record['category'] == 'C':
                sampled_values_C.add(record['value'])
        
        # 3. 验证每个类别被选中的具体位置不同
        # A 和 B 应该选中不同的元素
        self.assertNotEqual(sampled_values_A, sampled_values_B,
                          "Category A and B should select different items")
        
        # A 和 C 应该选中不同的元素
        self.assertNotEqual(sampled_values_A, sampled_values_C,
                          "Category A and C should select different items")
        
        # B 和 C 应该选中不同的元素
        self.assertNotEqual(sampled_values_B, sampled_values_C,
                          "Category B and C should select different items")
        
        # 4. 验证确定性：相同种子产生相同的类别特定mask
        sampled2, _ = sample_by_category(
            records, seed=42, sampling_rate=0.2
        )
        
        sampled_values_A2 = {r['value'] for r in sampled2 if r['category'] == 'A'}
        sampled_values_B2 = {r['value'] for r in sampled2 if r['category'] == 'B'}
        sampled_values_C2 = {r['value'] for r in sampled2 if r['category'] == 'C'}
        
        self.assertEqual(sampled_values_A, sampled_values_A2,
                        "Same seed should produce same mask for category A")
        self.assertEqual(sampled_values_B, sampled_values_B2,
                        "Same seed should produce same mask for category B")
        self.assertEqual(sampled_values_C, sampled_values_C2,
                        "Same seed should produce same mask for category C")


class TestPrintSamplingStats(unittest.TestCase):
    """测试 print_sampling_stats 函数"""
    
    def test_basic_output(self):
        """测试基本输出"""
        stats = {
            'A': {'original': 100, 'sampled': 12},
            'B': {'original': 150, 'sampled': 18},
            'C': {'original': 200, 'sampled': 24},
        }
        
        # 捕获输出
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            print_sampling_stats(stats, sampling_rate=0.12)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        
        # 验证输出包含关键信息
        self.assertIn('采样统计', output)
        self.assertIn('sampling_rate=12.00%', output)
        self.assertIn('A', output)
        self.assertIn('B', output)
        self.assertIn('C', output)
        self.assertIn('100', output)
        self.assertIn('12', output)
        self.assertIn('总计', output)
    
    def test_empty_stats(self):
        """测试空统计"""
        stats = {}
        
        # 不应该抛出异常
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            print_sampling_stats(stats, sampling_rate=0.5)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        self.assertIn('采样统计', output)


class TestIntegration(unittest.TestCase):
    """集成测试 - 测试完整工作流"""
    
    def test_full_workflow_mmsb_like(self):
        """测试类似MMSB的完整工作流"""
        # 模拟MMSB数据集：13个类别，每个类别约130个问题
        records = []
        mmsb_categories = [
            '01-Illegal_Activity', '02-HateSpeech', '03-Malware_Generation',
            '04-Physical_Harm', '05-EconomicHarm', '06-Fraud',
            '07-Pornography', '08-Political_Lobbying', '09-Privacy_Violence',
            '10-Legal_Opinion', '11-Financial_Advice', '12-Health_Consultation',
            '13-Gov_Decision'
        ]
        
        items_per_category = 129  # 13 * 129 ≈ 1677
        for category in mmsb_categories:
            for i in range(items_per_category):
                records.append({
                    'category': category,
                    'question_id': i,
                    'question': f'Question {i} in {category}'
                })
        
        # 执行12%采样
        sampled, stats = sample_by_category(
            records, seed=42, sampling_rate=0.12
        )
        
        # 验证结果
        total_original = sum(s['original'] for s in stats.values())
        total_sampled = sum(s['sampled'] for s in stats.values())
        
        self.assertEqual(total_original, 13 * items_per_category)
        self.assertEqual(len(sampled), total_sampled)
        
        # 验证每个类别都被采样
        self.assertEqual(len(stats), 13)
        
        # 验证每个类别的采样比例一致
        for category in mmsb_categories:
            expected = round(items_per_category * 0.12)
            self.assertEqual(stats[category]['sampled'], expected)
    
    def test_reproducibility_across_runs(self):
        """测试跨多次运行的可重复性"""
        records = [{'category': 'A', 'id': i} for i in range(1000)]
        
        results = []
        for _ in range(5):
            sampled, _ = sample_by_category(records, seed=12345, sampling_rate=0.25)
            sampled_ids = [r['id'] for r in sampled]
            results.append(sampled_ids)
        
        # 所有结果应该完全相同
        for i in range(1, 5):
            self.assertEqual(results[0], results[i])


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestGenerateSampleMask))
    suite.addTests(loader.loadTestsFromTestCase(TestApplyMaskToRecords))
    suite.addTests(loader.loadTestsFromTestCase(TestSampleRecords))
    suite.addTests(loader.loadTestsFromTestCase(TestSampleByCategory))
    suite.addTests(loader.loadTestsFromTestCase(TestPrintSamplingStats))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回测试是否成功
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

