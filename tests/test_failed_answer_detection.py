#!/usr/bin/env python3
"""
测试失败答案检测功能
"""
import sys
import os
# 添加项目根目录到路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from request import is_failed_answer

def test_failed_answer_detection():
    """测试各种失败答案的检测"""
    
    print("="*80)
    print("测试失败答案检测功能")
    print("="*80)
    
    # 测试用例
    test_cases = [
        # (答案文本, 预期结果, 描述)
        ("<your answer> and ends with", True, "VSP LLM 调用失败"),
        ("Please generate the next THOUGHT and ACTION", True, "VSP 未完成"),
        ("If you can get the answer, please also reply with ANSWER", True, "VSP 提示文本"),
        ("VSP completed but no clear answer found in debug log", True, "VSP 没有找到答案"),
        ("VSP Error: debug log not found", True, "VSP 执行错误"),
        ("", True, "空答案"),
        ("short", False, "短答案（现在允许）"),
        ("This is a valid answer with more than 20 characters.", False, "正常答案"),
        ("Here's a detailed response...\n\n1) First step\n2) Second step", False, "长答案"),
        ("[ERROR] API调用超时（120秒）", True, "明确的错误标志"),
    ]
    
    passed = 0
    failed = 0
    
    for answer, expected, description in test_cases:
        result = is_failed_answer(answer)
        status = "✅" if result == expected else "❌"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{status} {description}")
        print(f"   答案: {repr(answer[:50])}")
        print(f"   预期: {'失败' if expected else '成功'}, 实际: {'失败' if result else '成功'}")
    
    print("\n" + "="*80)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("="*80)
    
    return failed == 0

if __name__ == "__main__":
    success = test_failed_answer_detection()
    sys.exit(0 if success else 1)

