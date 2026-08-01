"""
QuickLoan -- Session 6: Baseline Evaluation (US-05)
====================================================

What you build this session
  A structured evaluation pipeline that runs 40 questions from a golden
  dataset through the Session 5 QuickLoan agent, scores each response,
  and produces a pass-rate report broken down by question category.

  Routing is evaluated deterministically (COMPLEX and OUT_OF_SCOPE have
  canned responses with known keywords). SIMPLE responses are scored 1-5
  by an LLM judge. A response passes if it is correctly routed, scores
  >= 3 out of 5, and contains no forbidden content.

What is NOT here yet
  - Regression tests comparing scores across commits (Session 12)

Run
  python s06/solution/evaluate.py

  The script imports the built QuickLoan graph from s05/solution/ and
  runs each question with a fresh MemorySaver checkpointer so no prior
  conversation state leaks between evaluation items.

Golden dataset
  s06/data/golden_dataset.json  --  40 Q&A items:
    20 SIMPLE  (loan rates, eligibility, loan products, loan policy)
    10 COMPLEX (personalised eligibility, comparisons, EMI calculations)
    10 OUT_OF_SCOPE (off-topic, competitor comparisons)
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

MODEL_NAME  = "llama-3.3-70b-versatile"  # judge stays off the agent's model to avoid self-grading bias; no tool calls needed here
PASS_SCORE  = 3   # minimum judge score (out of 5) for a SIMPLE question to pass

DATA_DIR     = Path(__file__).parent.parent.parent / "data"
DATASET_PATH = DATA_DIR / "golden_dataset.json"

# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

judge_llm = ChatGroq(
    api_key=GROQ_API_KEY, # type: ignore
    model=MODEL_NAME,
    temperature=0.0,
    max_tokens=100,
    max_retries=6,
)

JUDGE_PROMPT = """You are evaluating a loan AI assistant's response to a customer question.

Customer question:
{question}

The response should cover these points:
{criteria_list}

Assistant response:
{response}

Score the response on a scale of 1 to 5:
  5 = Excellent: all required points covered, factually accurate, professional
  4 = Good: most points covered, minor gaps
  3 = Acceptable: the key information is present but incomplete
  2 = Poor: missing important information or contains inaccuracies
  1 = Fail: refuses to answer, wrong information, or off-topic

Reply in exactly this format (two lines, no other text):
SCORE: <integer 1-5>
REASON: <one sentence explaining the score>"""


def parse_judge_response(output: str) -> tuple[int, str]:
    """Extract the integer score and one-line reason from the judge LLM output.

    Expected format:
      SCORE: 4
      REASON: The response correctly states the personal loan rate.

    If parsing fails, returns (0, "Could not parse judge output").
    Score is clamped to [1, 5] even if the LLM returns an out-of-range value.
    """
    score  = 0
    reason = "Could not parse judge output"
    for line in output.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                raw = int(line.split(":", 1)[1].strip())
                score = max(1, min(5, raw))
            except ValueError:
                pass
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return score, reason


def llm_judge(question: str, criteria: list[str], response: str) -> tuple[int, str]:
    """Ask the judge LLM to score a SIMPLE response.

    Returns (score, reason) where score is 1-5 and reason is a one-line string.
    Falls back to (0, error_message) on any exception.
    """
    criteria_list = "\n".join(f"  - {c}" for c in criteria) if criteria else "  - (none specified)"
    prompt = JUDGE_PROMPT.format(
        question=question,
        criteria_list=criteria_list,
        response=response,
    )
    try:
        result = judge_llm.invoke([
            SystemMessage(content="You are a strict but fair evaluation judge."),
            HumanMessage(content=prompt),
        ])
        return parse_judge_response(result.content) # type: ignore
    except Exception as e:
        return 0, f"Judge error: {e}"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {"id", "query", "expected_route", "category", "criteria"}


def load_dataset(path: Path) -> list[dict]:
    """Load and validate the golden dataset from a JSON file.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if any item is missing required fields.
    """
    with open(path, encoding="utf-8") as f:
        dataset = json.load(f)

    for i, item in enumerate(dataset):
        missing = REQUIRED_FIELDS - set(item.keys())
        if missing:
            raise ValueError(
                f"Golden dataset item {i} (id={item.get('id', '?')}) "
                f"is missing fields: {missing}"
            )
    return dataset


# ---------------------------------------------------------------------------
# Response evaluation
# ---------------------------------------------------------------------------

def evaluate_response(item: dict, result: dict) -> dict:
    """Score a single graph result against its golden dataset entry.

    Returns a dict with:
      id, query, category, expected_route, actual_route,
      route_correct, score, reason, forbidden_found, passed, response

    Scoring logic:
      - COMPLEX / OUT_OF_SCOPE: deterministic -- check that canned keywords
        appear in the response. Score is 5 if all criteria present, 1 otherwise.
      - SIMPLE: LLM-as-judge scores 1-5. Passes if score >= PASS_SCORE,
        route is correct, and no forbidden content is found.
    """
    actual_route  = result.get("query_type", "UNKNOWN")
    response      = result.get("response", "")
    route_correct = (actual_route == item["expected_route"])
    criteria      = item.get("criteria", [])
    must_not      = item.get("must_not_contain", [])

    criteria_met    = all(c.lower() in response.lower() for c in criteria)
    forbidden_found = [f for f in must_not if f.lower() in response.lower()]

    if item["expected_route"] in ("COMPLEX", "OUT_OF_SCOPE"):
        score  = 5 if criteria_met else 1
        reason = "Canned response criteria met." if criteria_met else "Canned response keyword missing."
        passed = route_correct and criteria_met and not forbidden_found
    else:
        score, reason = llm_judge(item["query"], criteria, response)
        passed = route_correct and score >= PASS_SCORE and not forbidden_found

    return {
        "id":             item["id"],
        "query":          item["query"],
        "category":       item["category"],
        "expected_route": item["expected_route"],
        "actual_route":   actual_route,
        "route_correct":  route_correct,
        "score":          score,
        "reason":         reason,
        "forbidden_found": forbidden_found,
        "passed":         passed,
        "response":       response,
    }


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(graph, dataset: list[dict]) -> list[dict]:
    """Invoke the graph on every dataset item and return a list of eval results.

    Each item gets a fresh thread_id so conversation history from one question
    does not bleed into the next. Uses the graph's existing checkpointer
    (MemorySaver in evaluation mode).
    """
    results = []
    for item in dataset:
        config = {"configurable": {"thread_id": f"eval-{item['id']}"}}
        try:
            graph_result = graph.invoke(
                {"customer_message": item["query"], "response": ""},
                config=config,
            )
        except Exception as e:
            graph_result = {
                "query_type": "ERROR",
                "response": f"Graph error: {e}",
            }
        eval_result = evaluate_response(item, graph_result)
        status = "PASS" if eval_result["passed"] else "FAIL"
        print(f"  [{status}] {item['id']}: {item['query'][:60]}")
        results.append(eval_result)
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: list[dict]) -> dict:
    """Aggregate evaluation results into a structured report.

    Returns:
      total, passed, failed, pass_rate (0.0-1.0),
      average_score (SIMPLE only),
      by_category: {category: {total, passed, pass_rate}},
      failures: list of failed items (id + query + reason + score)
    """
    total  = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    simple_scores = [r["score"] for r in results if r["category"] not in ("complex", "oos") and r["score"] > 0]
    avg_score = sum(simple_scores) / len(simple_scores) if simple_scores else 0.0

    by_category: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"total": 0, "passed": 0}
        by_category[cat]["total"] += 1
        if r["passed"]:
            by_category[cat]["passed"] += 1
    for cat, data in by_category.items():
        data["pass_rate"] = data["passed"] / data["total"] if data["total"] else 0.0

    failures = [
        {
            "id":     r["id"],
            "query":  r["query"],
            "reason": r["reason"],
            "score":  r["score"],
            "actual_route": r["actual_route"],
        }
        for r in results if not r["passed"]
    ]

    return {
        "total":        total,
        "passed":       passed,
        "failed":       failed,
        "pass_rate":    passed / total if total else 0.0,
        "average_score": round(avg_score, 2),
        "by_category":  by_category,
        "failures":     failures,
    }


def print_report(report: dict) -> None:
    """Print the evaluation report to stdout in a readable format."""
    print("\n" + "=" * 60)
    print("  QuickLoan Baseline Evaluation Report")
    print("=" * 60)
    print(f"  Total questions : {report['total']}")
    print(f"  Passed          : {report['passed']}")
    print(f"  Failed          : {report['failed']}")
    print(f"  Pass rate       : {report['pass_rate']:.0%}")
    print(f"  Avg SIMPLE score: {report['average_score']} / 5")
    print()
    print("  By category:")
    for cat, data in sorted(report["by_category"].items()):
        bar = "#" * data["passed"] + "-" * (data["total"] - data["passed"])
        print(f"    {cat:<15} [{bar}] {data['passed']}/{data['total']} ({data['pass_rate']:.0%})")

    if report["failures"]:
        print()
        print(f"  Failed items ({len(report['failures'])}):")
        for f in report["failures"]:
            print(f"    {f['id']}: (route={f['actual_route']}, score={f['score']}) {f['query'][:55]}")
            print(f"         {f['reason']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Load the golden dataset, build the QuickLoan graph, run evaluation, print report."""
    from langgraph.checkpoint.memory import MemorySaver
    from quickloan.agent import build_graph

    graph = build_graph(checkpointer=MemorySaver())

    dataset = load_dataset(DATASET_PATH)
    print(f"\nRunning evaluation on {len(dataset)} questions...")
    print("-" * 60)

    results = run_evaluation(graph, dataset)
    report  = generate_report(results)
    print_report(report)


if __name__ == "__main__":
    main()