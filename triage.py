# file: triage.py
# Triage logic — maps stress score to priority level


def classify(stress_score):
    """
    Map a numeric stress score (0–100) to a triage priority.

    Thresholds:
      0–30  → Low
      31–70 → Medium
      71–100 → High

    Parameters
    ----------
    stress_score : int   Score from model inference (0–100)

    Returns
    -------
    dict  {"stress_score": int, "priority": str}
    """
    score = max(0, min(100, int(stress_score)))

    if score <= 30:
        priority = "Low"
    elif score <= 70:
        priority = "Medium"
    else:
        priority = "High"

    return {
        "stress_score": score,
        "priority": priority,
    }
