import numpy as np
from math_verify import parse, verify


def math_reward_function(solution_str, ground_truth):
    if "</think>" in solution_str:
        solution_str = solution_str.split("</think>")[1]
    try:
        math_verify_parsed = parse(solution_str, parsing_timeout=5)
    except Exception:
        return 0.0
    if len(math_verify_parsed) < 2:
        return 0.0
    if math_verify_parsed[1] in ground_truth:
        return 1.0
    for valid_answer in ground_truth:
        try:
            if verify(
                parse(f"\\boxed{{{valid_answer}}}", parsing_timeout=5),
                math_verify_parsed,
                timeout_seconds=5,
            ):
                return 1.0
        except Exception:
            continue
    return 0.0


def compute_score(solution_str, ground_truth):
    # assert len(ground_truth) == 1, f"ground_truth must be a single value, but got {ground_truth}"
    if isinstance(ground_truth, (str, float, int)):
        ground_truth = [ground_truth]
    elif isinstance(ground_truth, list) and isinstance(ground_truth[0], np.ndarray):
        ground_truth = ground_truth[0].tolist()
    score = math_reward_function(solution_str, ground_truth)
    if isinstance(score, (int, float, bool)):
        return float(score)
    else:
        return float(score[0])
