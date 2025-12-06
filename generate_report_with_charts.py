"""
生成包含图表的评估报告

对每个模型系列，生成柱状图显示不同类别的攻击率
"""

import os
import csv
import json
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 13个类别
CATEGORIES = [
    '01-Illegal_Activitiy',
    '02-HateSpeech',
    '03-Malware_Generation',
    '04-Physical_Harm',
    '05-EconomicHarm',
    '06-Fraud',
    '07-Sex',
    '08-Political_Lobbying',
    '09-Privacy_Violence',
    '10-Legal_Opinion',
    '11-Financial_Advice',
    '12-Health_Consultation',
    '13-Gov_Decision'
]

# 类别简称（用于图表）
CATEGORY_LABELS = [
    'Illegal',
    'Hate',
    'Malware',
    'Physical',
    'Economic',
    'Fraud',
    'Sex',
    'Political',
    'Privacy',
    'Legal',
    'Financial',
    'Health',
    'Gov'
]

def read_csv_file(filepath):
    """读取 CSV 文件并返回数据"""
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 处理不同的列名格式
            category = row.get('Category') or row.get('category')
            
            # 两种格式：
            # 1. Attack_Rate(%) - 已经是百分比（6.19 = 6.19%）
            # 2. attack_rate - 是小数（0.0206 = 2.06%，需要×100）
            if 'Attack_Rate(%)' in row:
                attack_rate_str = row['Attack_Rate(%)']
                is_percentage = True
            elif 'attack_rate' in row:
                attack_rate_str = row['attack_rate']
                is_percentage = False  # 小数格式，需要×100
            else:
                continue
            
            if not category or not attack_rate_str:
                continue
            
            try:
                attack_rate = float(str(attack_rate_str).replace('%', '').strip())
                # 如果是小数格式，转换为百分比
                if not is_percentage and attack_rate < 1.0:
                    attack_rate *= 100
            except ValueError:
                print(f"⚠️  无法解析攻击率: {filepath}, {category}, {attack_rate_str}")
                attack_rate = 0.0
            
            data[category] = attack_rate
    return data

def parse_filename(filename):
    """
    从文件名提取模型信息和时间
    例如: eval_qwen_qwen3-vl-235b-a22b-thinking_2025-11-16_08-06-28_tasks_1680.csv
    返回: (brand, model_display_name, timestamp)
    """
    # 移除 eval_ 前缀和 .csv 后缀
    name = filename.replace('eval_', '').replace('.csv', '')
    
    # 提取时间戳（如果存在）
    parts = name.split('_')
    timestamp = None
    if 'tasks' in name:
        # 找到包含日期的部分
        for i, part in enumerate(parts):
            if '-' in part and len(part) == 10:  # 日期格式 YYYY-MM-DD
                timestamp = f"{parts[i]}_{parts[i+1]}"
                break
    
    # 识别品牌和生成显示名称
    if 'google_gemini' in name or 'gemini' in name:
        brand = 'Gemini'
        if 'vsp' in name:
            model_display_name = 'Gemini-2.5-Flash + VSP'
        elif name.startswith('mini_'):
            model_display_name = 'Gemini-2.5-Flash (Mini-Eval)'
        else:
            model_display_name = 'Gemini-2.5-Flash'
    
    elif 'gpt-5' in name or 'gpt5' in name:
        brand = 'OpenAI'
        if 'vsp' in name:
            model_display_name = 'GPT-5 + VSP'
        else:
            model_display_name = 'GPT-5'
    
    elif 'qwen' in name:
        # 先检查是否是 VSP + Qwen
        is_vsp = 'vsp' in name
        
        # 检查是否是 Thinking 模式
        is_thinking = 'thinking' in name
        
        # 根据 Thinking 分组
        if is_thinking:
            brand = 'Qwen (Thinking)'
        else:
            brand = 'Qwen'
        
        # 区分不同的Qwen模型
        if 'qwen3-vl-235b' in name or 'qwen_qwen3-vl-235b' in name:
            if is_thinking:
                base_name = 'Qwen3-VL-235B-Thinking'
            else:
                base_name = 'Qwen3-VL-235B-Instruct'
        elif 'qwen3-vl-30b' in name or 'qwen_qwen3-vl-30b' in name:
            if is_thinking:
                base_name = 'Qwen3-VL-30B-Thinking'
            else:
                base_name = 'Qwen3-VL-30B-Instruct'
        elif 'qwen3-vl-8b' in name or 'qwen_qwen3-vl-8b' in name:
            if is_thinking:
                base_name = 'Qwen3-VL-8B-Thinking'
            else:
                base_name = 'Qwen3-VL-8B-Instruct'
        else:
            base_name = 'Qwen3-VL (Unknown)'
        
        # 如果是 VSP，添加后缀
        if is_vsp:
            model_display_name = f'{base_name} + VSP'
        else:
            model_display_name = base_name
    
    elif 'internvl' in name:
        brand = 'InternVL'
        if 'vsp' in name:
            model_display_name = 'InternVL3-78B + VSP'
        else:
            model_display_name = 'InternVL3-78B'
    
    elif 'vsp' in name:
        brand = 'VSP'
        model_display_name = 'VSP (Unknown Model)'
    
    else:
        brand = 'Other'
        model_display_name = 'Unknown Model'
    
    return brand, model_display_name, timestamp

def load_all_data():
    """加载所有 1680 任务的评估数据，按品牌分组"""
    output_dir = 'output'
    all_data = defaultdict(list)  # {brand: [(model_display_name, timestamp, data), ...]}
    
    for filename in os.listdir(output_dir):
        if filename.startswith('eval_') and filename.endswith('.csv') and 'tasks_1680' in filename:
            filepath = os.path.join(output_dir, filename)
            
            # 解析文件名
            brand, model_display_name, timestamp = parse_filename(filename)
            
            # 读取数据
            data = read_csv_file(filepath)
            
            all_data[brand].append({
                'model_display_name': model_display_name,
                'timestamp': timestamp,
                'data': data,
                'filename': filename
            })
    
    # 同时检查没有 tasks_1680 标记但是有 14 行（13个类别+表头）或 15 行（+空行）的文件
    for filename in os.listdir(output_dir):
        if filename.startswith('eval_') and filename.endswith('.csv') and 'tasks_1680' not in filename:
            filepath = os.path.join(output_dir, filename)
            
            # 检查行数
            with open(filepath, 'r') as f:
                line_count = sum(1 for _ in f)
            
            if line_count == 14 or line_count == 15:  # 13 类别 + 1 表头 (+ 可能的空行)
                brand, model_display_name, timestamp = parse_filename(filename)
                data = read_csv_file(filepath)
                
                all_data[brand].append({
                    'model_display_name': model_display_name,
                    'timestamp': timestamp,
                    'data': data,
                    'filename': filename
                })
    
    return all_data

def average_multiple_runs(models_data):
    """
    对同一模型的多次运行取平均值
    
    Args:
        models_data: [{model_display_name, timestamp, data}, ...]
    
    Returns:
        {display_name: {category: avg_attack_rate}}
    """
    # 按完整模型名分组
    model_groups = defaultdict(list)
    
    for item in models_data:
        # 直接使用model_display_name作为分组键
        model_display_name = item['model_display_name']
        model_groups[model_display_name].append(item)
    
    # 计算平均值（如果同一个模型有多次运行，取平均）
    averaged_data = {}
    for model_name, items in model_groups.items():
        averaged_data[model_name] = {}
        for category in CATEGORIES:
            rates = []
            for item in items:
                if category in item['data']:
                    rates.append(item['data'][category])
            
            if rates:
                averaged_data[model_name][category] = np.mean(rates)
            else:
                averaged_data[model_name][category] = 0.0
    
    return averaged_data

def create_bar_chart(brand, averaged_data, output_file):
    """
    创建柱状图
    
    Args:
        brand: 品牌名称（如 Qwen, Gemini）
        averaged_data: {model_name: {category: attack_rate}}
        output_file: 输出文件路径
    """
    # 限制图表大小
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 准备数据 - 按模型名排序
    variants = sorted(list(averaged_data.keys()))
    x = np.arange(len(CATEGORY_LABELS))
    width = 0.8 / len(variants) if len(variants) > 0 else 0.8
    
    # 使用高对比度的颜色方案
    # 定义专业的配色：蓝、红、绿、橙、紫、青、粉
    color_palette = [
        '#1f77b4',  # 深蓝
        '#ff7f0e',  # 橙色
        '#2ca02c',  # 绿色
        '#d62728',  # 红色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#7f7f7f',  # 灰色
        '#17becf',  # 青色
        '#bcbd22',  # 黄绿
    ]
    # 根据需要循环使用颜色
    colors = [color_palette[i % len(color_palette)] for i in range(len(variants))]
    
    for i, (variant, color) in enumerate(zip(variants, colors)):
        data = averaged_data[variant]
        attack_rates = [data[cat] for cat in CATEGORIES]
        
        offset = (i - len(variants)/2 + 0.5) * width
        bars = ax.bar(x + offset, attack_rates, width, 
                     label=variant, color=color, alpha=0.9, edgecolor='white', linewidth=0.5)
        
        # 在所有柱子上显示数值
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # 只要不是0就显示
                # 根据高度调整字体大小和位置
                if height > 5:
                    fontsize = 7
                    va = 'bottom'
                    y_offset = 0
                else:
                    fontsize = 6
                    va = 'bottom'
                    y_offset = 1
                
                ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                       f'{height:.1f}%',
                       ha='center', va=va, fontsize=fontsize, rotation=0)
    
    # 设置标签和标题
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attack Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{brand} Models - Attack Rate by Category', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_LABELS, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)
    
    # 添加水平参考线
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(len(CATEGORY_LABELS)-0.5, 51, '50%', color='red', fontsize=9)
    
    # 保存图表
    plt.subplots_adjust(bottom=0.15, top=0.92, left=0.08, right=0.98)
    try:
        plt.savefig(output_file, dpi=100)  # 进一步降低 dpi
    except Exception as e:
        print(f"⚠️  生成图表失败 {output_file}: {e}")
    finally:
        plt.close()
    
    print(f"✅ 生成图表: {output_file}")

def generate_html_report(all_data, output_file='output/evaluation_report.html'):
    """生成包含所有图表的 HTML 报告"""
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MM-SafetyBench Evaluation Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 40px;
            border-left: 5px solid #3498db;
            padding-left: 15px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chart-container img {
            width: 100%;
            height: auto;
        }
        .summary {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        .stat-label {
            color: #7f8c8d;
            font-size: 14px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .timestamp {
            color: #7f8c8d;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <h1>📊 MM-SafetyBench Evaluation Report</h1>
    <div class="summary">
        <p><strong>Generated:</strong> """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        <p><strong>Total Brands:</strong> """ + str(len(all_data)) + """</p>
        <p><strong>Total Models:</strong> """ + str(sum(len(models) for models in all_data.values())) + """</p>
    </div>
"""
    
    # 为每个品牌生成图表和统计
    for i, (brand, models_data) in enumerate(sorted(all_data.items()), 1):
        # 计算平均值
        averaged_data = average_multiple_runs(models_data)
        
        # 生成图表
        chart_file = f'output/chart_{i}_{brand.replace(" ", "_").replace("+", "")}.png'
        create_bar_chart(brand, averaged_data, chart_file)
        
        # 添加到 HTML
        html_content += f"""
    <h2>{i}. {brand}</h2>
    <div class="chart-container">
        <img src="{os.path.basename(chart_file)}" alt="{brand} Chart">
    </div>
    
    <div class="stats-grid">
"""
        
        # 添加统计卡片
        for model_name, data in averaged_data.items():
            avg_attack_rate = np.mean(list(data.values()))
            html_content += f"""
        <div class="stat-card">
            <div class="stat-label">{model_name}</div>
            <div class="stat-value">{avg_attack_rate:.1f}%</div>
            <div class="timestamp">Average Attack Rate</div>
        </div>
"""
        
        html_content += """
    </div>
    
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Runs</th>
                <th>CSV Files</th>
            </tr>
        </thead>
        <tbody>
"""
        
        # 添加详细信息表格
        model_info = defaultdict(list)
        for item in models_data:
            model_name = item['model_display_name']
            model_info[model_name].append({
                'timestamp': item['timestamp'],
                'filename': item['filename']
            })
        
        for model_name, info_list in model_info.items():
            filenames = [info['filename'] for info in info_list]
            html_content += f"""
            <tr>
                <td><strong>{model_name}</strong></td>
                <td>{len(filenames)}</td>
                <td class="timestamp" style="max-width: 600px; word-break: break-all;">{', '.join(filenames)}</td>
            </tr>
"""
        
        html_content += """
        </tbody>
    </table>
"""
    
    html_content += """
    <div class="summary" style="margin-top: 40px;">
        <h3>📝 Notes</h3>
        <ul>
            <li>Models are grouped by brand (e.g., all Qwen models in one chart)</li>
            <li>Attack rates are averaged across multiple runs of the same model</li>
            <li>All evaluations are based on 1680 tasks from MM-SafetyBench</li>
            <li>Lower attack rate indicates better safety performance</li>
        </ul>
    </div>
</body>
</html>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 生成 HTML 报告: {output_file}")

def load_specific_data(eval_files: list):
    """
    加载指定的评估文件数据，按品牌分组
    
    Args:
        eval_files: 评估文件路径列表
        
    Returns:
        {brand: [(model_display_name, timestamp, data), ...]}
    """
    all_data = defaultdict(list)
    
    for filepath in eval_files:
        if not os.path.exists(filepath):
            print(f"⚠️  文件不存在: {filepath}")
            continue
        
        filename = os.path.basename(filepath)
        
        # 解析文件名
        brand, model_display_name, timestamp = parse_filename(filename)
        
        # 读取数据
        data = read_csv_file(filepath)
        
        if data:
            all_data[brand].append({
                'model_display_name': model_display_name,
                'timestamp': timestamp,
                'data': data,
                'filename': filename
            })
    
    return all_data


def main(eval_files: list = None, output_file: str = None):
    """
    主函数
    
    Args:
        eval_files: 指定的评估文件列表，如果为 None 则使用默认逻辑加载所有符合条件的文件
        output_file: 输出报告文件路径，如果为 None 则使用默认路径
    """
    print("📊 开始生成评估报告...\n")
    
    # 加载数据
    print("📖 加载评估数据...")
    if eval_files:
        print(f"   指定了 {len(eval_files)} 个评估文件")
        all_data = load_specific_data(eval_files)
    else:
        all_data = load_all_data()
    
    if not all_data:
        print("⚠️  没有找到有效的评估数据")
        return
    
    print(f"✅ 找到 {len(all_data)} 个品牌")
    for brand, models in all_data.items():
        print(f"  - {brand}: {len(models)} 个模型")
    
    print("\n🎨 生成图表和报告...")
    
    # 生成 HTML 报告
    report_output = output_file or 'output/evaluation_report.html'
    generate_html_report(all_data, output_file=report_output)
    
    print("\n🎉 完成！")
    print(f"📄 HTML 报告: {report_output}")
    print("🖼️  图表文件: output/chart_*.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成包含图表的评估报告")
    parser.add_argument("--files", nargs='+', default=None,
                       help="指定要处理的评估 CSV 文件列表。不指定则使用默认逻辑加载所有符合条件的文件")
    parser.add_argument("--output", default=None,
                       help="输出报告文件路径（默认: output/evaluation_report.html）")
    
    args = parser.parse_args()
    
    main(eval_files=args.files, output_file=args.output)

