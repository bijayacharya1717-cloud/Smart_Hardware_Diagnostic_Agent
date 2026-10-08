"""Evaluate end-to-end diagnostics with DeepEval contextual recall/faithfulness.

Run: python -m evals.evaluate_fault_scenarios
DeepEval's configured judge model/API credentials are required for scoring.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import ContextualRecallMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from diagnostic_agent import diagnose

ROOT = Path(__file__).resolve().parents[1]


def main():
    scenarios = json.loads((ROOT / "golden" / "fault_scenarios.json").read_text(encoding="utf-8"))
    cases = []
    for scenario in scenarios:
        result = asyncio.run(diagnose(scenario["description"], scenario.get("failure_time")))
        context = [
            entry.get("content") or json.dumps(entry.get("result", entry), ensure_ascii=False)
            for entry in result["evidence"].values()
        ]
        cases.append(LLMTestCase(
            input=scenario["description"],
            actual_output=result["diagnosis"],
            expected_output=scenario["ideal_diagnosis"],
            retrieval_context=context,
        ))
        print(f"\n=== {scenario['id']} ===\n{result['diagnosis']}\n")

    metrics = [
        ContextualRecallMetric(threshold=0.7, include_reason=True),
        FaithfulnessMetric(threshold=0.7, include_reason=True),
    ]
    evaluate(test_cases=cases, metrics=metrics)


if __name__ == "__main__":
    main()
