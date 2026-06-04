"""
Agent 评估脚本
==============

用于测试 Agent 的意图识别准确率和回答质量。

用法：
    python eval_runner.py              # 运行所有测试用例
    python eval_runner.py --intent     # 只测意图识别（不调用 Agent，但是调用大模型）
    python eval_runner.py --case rag_001  # 只跑指定用例
"""

import argparse
import json
import sys
import time
from typing import Any
from pathlib import Path

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    print("错误: 缺少 langchain-openai 依赖")
    print("请运行: pip install langchain-openai")
    sys.exit(1)

from config import DASHSCOPE_API_KEY, CHAT_MODEL_NAME
from workflow import create_router, route_query


def load_cases(path: str = "eval_cases.json") -> list[dict[str, Any]]:
    """加载测试用例文件。"""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"测试用例文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_eval(cases: list[dict[str, Any]], intent_only: bool = False) -> dict[str, Any]:
    """执行评估并返回结果统计。"""
    llm = ChatOpenAI(
        model=CHAT_MODEL_NAME,
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=60,
    )
    router = create_router(llm)

    agent = None
    if not intent_only:
        from agent_service import AnimeAgent
        agent = AnimeAgent(session_id="eval_session")

    total = len(cases)
    intent_correct = 0
    keyword_hit = 0
    failures = []

    for i, case in enumerate(cases):
        if i > 0:
            time.sleep(2)
        case_id = case["id"]
        user_input = case["input"]
        expected_intent = case["expected_intent"]
        must_include = case.get("must_include", [])

        print(f"\n[{case_id}] 输入: {user_input}")

        # === Step 1: 意图识别测试 ===
        try:
            actual_intent = route_query(router, user_input)
        except Exception as e:
            print(f"  ✗ 意图识别异常: {e}")
            failures.append(f"{case_id}: 意图识别异常 - {e}")
            continue

        intent_ok = actual_intent == expected_intent

        if intent_ok:
            intent_correct += 1
            print(f"  ✓ 意图: {actual_intent}")
        else:
            print(f"  ✗ 意图: expected={expected_intent}, got={actual_intent}")
            failures.append(f"{case_id}: expected {expected_intent}, got {actual_intent}")

        # === Step 2: 关键词命中测试 ===
        if not intent_only and agent:
            try:
                response = agent.chat(user_input)["output"]
            except Exception as e:
                print(f"  ✗ Agent 调用异常: {e}")
                failures.append(f"{case_id}: Agent 调用异常 - {e}")
                agent.clear_history()
                continue

            print(f"  回答: {response[:80]}...")

            missing = [kw for kw in must_include if kw not in response]
            if not missing:
                keyword_hit += 1
                print(f"  ✓ 关键词全部命中: {must_include}")
            else:
                print(f"  ✗ 缺少关键词: {missing}")
                if intent_ok:
                    failures.append(f"{case_id}: 回答缺少关键词 {missing}")

            agent.clear_history()

    return {
        "total": total,
        "intent_correct": intent_correct,
        "keyword_hit": keyword_hit if not intent_only else None,
        "failures": failures,
        "intent_only": intent_only,
    }


def print_report(result: dict[str, Any]):
    """打印格式化的评估报告。"""
    total = result["total"]
    intent_correct = result["intent_correct"]
    intent_only = result["intent_only"]

    print("\n" + "=" * 50)
    print("评估报告")
    print("=" * 50)
    print(f"总用例数：{total}")
    print(f"意图识别正确：{intent_correct} / {total}")
    
    if total > 0:
        print(f"意图准确率：{intent_correct / total * 100:.1f}%")

    if not intent_only:
        keyword_hit = result["keyword_hit"]
        print(f"\n回答关键词命中：{keyword_hit} / {total}")
        if total > 0:
            print(f"回答命中率：{keyword_hit / total * 100:.1f}%")

    if result["failures"]:
        print("\n失败用例：")
        for f in result["failures"]:
            print(f"  - {f}")
    else:
        print("\n全部通过！")


def main():
    """评估脚本入口函数。"""
    parser = argparse.ArgumentParser(description="Agent 评估工具")
    parser.add_argument("--intent", action="store_true", help="只测试意图识别")
    parser.add_argument("--case", type=str, help="只运行指定的测试用例 ID")
    parser.add_argument("--file", type=str, default="eval_cases.json", help="测试用例文件路径")
    args = parser.parse_args()

    try:
        cases = load_cases(args.file)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: 测试用例文件 JSON 格式错误 - {e}")
        sys.exit(1)

    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"未找到用例: {args.case}")
            sys.exit(1)

    result = run_eval(cases, intent_only=args.intent)
    print_report(result)


if __name__ == "__main__":
    main()
