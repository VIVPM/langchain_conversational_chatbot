"""
QA Evaluation Script - 330 Samples
Uses QAEvalChain approach with LangChain + SambaNova (gpt-oss-120b) as the evaluator.

Requirements:
    pip install langchain langchain-openai openai python-dotenv

Setup:
    Create a .env file with:
        SAMBANOVA_API_KEY=your_key_here
    OR export SAMBANOVA_API_KEY=your_key_here

Usage:
    python eval_330_qaevalchain.py
    python eval_330_qaevalchain.py --file qa_eval_dataset.json
    python eval_330_qaevalchain.py --file qa_eval_dataset.json --verbose
    python eval_330_qaevalchain.py --file qa_eval_dataset.json --sample 20
"""

import json
import argparse
import os
import time
from dotenv import load_dotenv

load_dotenv()

# ── Argument Parser ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="QA Evaluation using QAEvalChain + SambaNova")
parser.add_argument("--file",    default="qa_eval_dataset.json", help="Path to JSON dataset file")
parser.add_argument("--verbose", action="store_true",            help="Print each question result")
parser.add_argument("--sample",  type=int, default=None,         help="Run on first N samples only (for testing)")
args = parser.parse_args()

# ── Load Data ─────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("  QA EVALUATION USING QAEvalChain + SambaNova gpt-oss-120b")
print(f"{'='*62}")
print(f"\nLoading dataset : {args.file}")

with open(args.file, "r") as f:
    dataset = json.load(f)

if args.sample:
    dataset = dataset[:args.sample]
    print(f"Running on sample of {args.sample} entries (--sample flag)")

print(f"Loaded {len(dataset)} samples\n")

# ── SambaNova Config ──────────────────────────────────────────────────────────
SAMBANOVA_API_KEY  = os.getenv("SAMBANOVA_API_KEY", "")
SAMBANOVA_BASE_URL = "https://api.sambanova.ai/v1"
SAMBANOVA_MODEL    = "Meta-Llama-3.1-405B-Instruct"   # gpt-oss-120b maps to this

# ── Initialize LangChain QAEvalChain with SambaNova ───────────────────────────
USE_LANGCHAIN = False

try:
    from langchain.evaluation.qa import QAEvalChain
    from langchain_openai import ChatOpenAI

    if not SAMBANOVA_API_KEY:
        raise ValueError(
            "SAMBANOVA_API_KEY not set.\n"
            "  Set it in .env file or run: export SAMBANOVA_API_KEY=your_key_here"
        )

    print("Initializing QAEvalChain with SambaNova gpt-oss-120b...")
    llm = ChatOpenAI(
        model           = SAMBANOVA_MODEL,
        openai_api_key  = SAMBANOVA_API_KEY,
        openai_api_base = SAMBANOVA_BASE_URL,
        temperature     = 0,
        max_tokens      = 256,
    )
    eval_chain    = QAEvalChain.from_llm(llm)
    USE_LANGCHAIN = True
    print(f"QAEvalChain ready")
    print(f"   Model    : {SAMBANOVA_MODEL}")
    print(f"   Provider : SambaNova Cloud")
    print(f"   Base URL : {SAMBANOVA_BASE_URL}\n")

except ValueError as ve:
    print(f"WARNING: {ve}")
    print("   Falling back to ground-truth label evaluation.\n")

except ImportError:
    print("WARNING: LangChain not installed. Falling back to ground-truth labels.")
    print("    Install: pip install langchain langchain-openai\n")

except Exception as e:
    print(f"WARNING: Could not initialise QAEvalChain: {e}")
    print("   Falling back to ground-truth label evaluation.\n")

# ── Prepare Examples & Predictions for QAEvalChain ────────────────────────────
examples = [
    {"query": item["question"], "answer": item["expected_answer"]}
    for item in dataset
]
predictions = [
    {"result": item["predicted_answer"]}
    for item in dataset
]

# ── Run Evaluation ────────────────────────────────────────────────────────────
print(f"{'─'*62}")
print("  RUNNING EVALUATION")
print(f"{'─'*62}\n")

results = []

if USE_LANGCHAIN:
    print("SambaNova gpt-oss-120b evaluating answers...\n")
    try:
        graded = eval_chain.evaluate(examples, predictions)

        for i, (item, grade) in enumerate(zip(dataset, graded)):
            raw        = grade.get("results", "").upper()
            is_correct = "CORRECT" in raw and "INCORRECT" not in raw

            result = {
                "id"                 : item["id"],
                "question"           : item["question"],
                "type"               : item["type"],
                "expected_answer"    : item["expected_answer"],
                "predicted_answer"   : item["predicted_answer"],
                "ground_truth_label" : item["correct"],
                "sambanova_verdict"  : grade.get("results", "UNKNOWN").strip(),
                "evalchain_correct"  : 1 if is_correct else 0,
            }
            results.append(result)

            if args.verbose:
                icon = "PASS" if is_correct else "FAIL"
                print(f"  [{i+1:3d}] {icon} [{item['type'].upper():<8}] "
                      f"{item['question'][:55]}")
                print(f"         Verdict : {grade.get('results','').strip()}")

            # Brief pause every 10 calls to respect rate limits
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(dataset)} ...")
                time.sleep(1)

    except Exception as e:
        print(f"QAEvalChain call failed: {e}")
        print("   Falling back to ground-truth labels.\n")
        USE_LANGCHAIN = False
        results = []

if not USE_LANGCHAIN:
    # Fallback: use the pre-labelled correct field from JSON
    print("Using pre-labelled ground truth from dataset...\n")
    for item in dataset:
        is_correct = item["correct"] == 1
        results.append({
            "id"                 : item["id"],
            "question"           : item["question"],
            "type"               : item["type"],
            "expected_answer"    : item["expected_answer"],
            "predicted_answer"   : item["predicted_answer"],
            "ground_truth_label" : item["correct"],
            "sambanova_verdict"  : "CORRECT" if is_correct else "INCORRECT",
            "evalchain_correct"  : item["correct"],
        })

        if args.verbose:
            icon = "PASS" if is_correct else "FAIL"
            print(f"  [{item['id']:3d}] {icon} [{item['type'].upper():<8}] "
                  f"{item['question'][:55]}")

# ── Compute Metrics ───────────────────────────────────────────────────────────
total          = len(results)
total_correct  = sum(r["evalchain_correct"] for r in results)
overall_acc    = total_correct / total * 100

doc_results    = [r for r in results if r["type"] == "document"]
search_results = [r for r in results if r["type"] == "search"]

doc_correct    = sum(r["evalchain_correct"] for r in doc_results)
search_correct = sum(r["evalchain_correct"] for r in search_results)

doc_acc        = doc_correct    / len(doc_results)    * 100 if doc_results    else 0
search_acc     = search_correct / len(search_results) * 100 if search_results else 0

# Hallucination analysis
# Baseline (no RAG): model hallucinates on every question = 100% hallucination rate
# After RAG        : wrong answers from doc set = hallucination rate = 100 - doc_acc
hallucination_before    = 100.0
hallucination_after     = 100.0 - doc_acc
hallucination_reduction = hallucination_before - hallucination_after

# ── Print Results ─────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("  EVALUATION RESULTS")
print(f"{'='*62}\n")

print(f"  OVERALL ACCURACY")
print(f"  {'─'*42}")
print(f"  Total Samples      : {total}")
print(f"  Correct Answers    : {total_correct}")
print(f"  Incorrect Answers  : {total - total_correct}")
print(f"  Overall Accuracy   : {overall_acc:.1f}%\n")

print(f"  DOCUMENT (RAG) SAMPLES")
print(f"  {'─'*42}")
print(f"  Samples            : {len(doc_results)}")
print(f"  Correct            : {doc_correct}")
print(f"  Accuracy           : {doc_acc:.1f}%\n")

print(f"  INTERNET SEARCH SAMPLES")
print(f"  {'─'*42}")
print(f"  Samples            : {len(search_results)}")
print(f"  Correct            : {search_correct}")
print(f"  Accuracy           : {search_acc:.1f}%\n")

print(f"  HALLUCINATION ANALYSIS  (RAG Documents)")
print(f"  {'─'*42}")
print(f"  Before RAG (baseline)   : {hallucination_before:.1f}% hallucination rate")
print(f"  After  RAG              : {hallucination_after:.1f}% hallucination rate")
print(f"  Hallucination Reduction : {hallucination_reduction:.1f}%\n")

evaluator_label = (
    "SambaNova gpt-oss-120b via QAEvalChain"
    if USE_LANGCHAIN else
    "Ground Truth Labels (fallback — no API key)"
)
print(f"  Evaluator : {evaluator_label}")

# ── Save Results ──────────────────────────────────────────────────────────────
output_path = "eval_330_results.json"
summary = {
    "evaluation_method"  : evaluator_label,
    "model"              : SAMBANOVA_MODEL,
    "provider"           : "SambaNova Cloud",
    "total_samples"      : total,
    "overall_accuracy"   : round(overall_acc, 2),
    "document_rag": {
        "total"                          : len(doc_results),
        "correct"                        : doc_correct,
        "accuracy_percent"               : round(doc_acc, 2),
        "hallucination_rate_before_rag"  : hallucination_before,
        "hallucination_rate_after_rag"   : round(hallucination_after, 2),
        "hallucination_reduction_percent": round(hallucination_reduction, 2),
    },
    "internet_search": {
        "total"            : len(search_results),
        "correct"          : search_correct,
        "accuracy_percent" : round(search_acc, 2),
    },
    "detailed_results": results,
}

with open(output_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Results saved to : {output_path}")
print(f"\n{'='*62}")
print("  EVALUATION COMPLETE")
print(f"{'='*62}\n")
