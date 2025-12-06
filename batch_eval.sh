#!/bin/bash
#
# 批量运行 mmsb_eval.py 对所有 *_tasks_1680.jsonl 文件进行评估
#

set -e

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 查找所有符合条件的 jsonl 文件
files=(output/*_tasks_1680.jsonl)

if [ ${#files[@]} -eq 0 ]; then
    echo "❌ 没有找到 output/*_tasks_1680.jsonl 文件"
    exit 1
fi

echo "📋 找到 ${#files[@]} 个文件待评估:"
for f in "${files[@]}"; do
    echo "  - $f"
done
echo ""

# 计数器
total=${#files[@]}
current=0
success=0
failed=0

# 逐个运行评估
for jsonl_file in "${files[@]}"; do
    current=$((current + 1))
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$current/$total] 评估: $jsonl_file"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if python mmsb_eval.py --jsonl_file "$jsonl_file" --override; then
        echo "✅ 完成: $jsonl_file"
        success=$((success + 1))
    else
        echo "❌ 失败: $jsonl_file"
        failed=$((failed + 1))
    fi
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 评估完成!"
echo "  ✅ 成功: $success"
echo "  ❌ 失败: $failed"
echo "  📁 总计: $total"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

