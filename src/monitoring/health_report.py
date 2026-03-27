"""
src/monitoring/health_report.py
--------------------------------
Sentinel Suite — Model Health Report generator.

Parses the structured result dict returned by drift_detector.run_drift_report()
and emits a tiered alert report with actionable recommendations.

Alert tiers
-----------
  CRITICAL  — dataset-level drift detected  → retrain immediately
  WARNING   — ≥ 1 column drifted            → scheduled retraining review
  INFO      — no drift                      → model stable, continue monitoring

Usage (standalone):
    python src/monitoring/health_report.py

Usage (programmatic):
    from src.monitoring.health_report import generate_health_report
    report = generate_health_report(result_dict)
"""

import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.monitoring.drift_detector import run_drift_report

# ------------------------------------------------------------------
# Clinical thresholds — features where drift has direct patient risk
# ------------------------------------------------------------------
_HIGH_STAKES_FEATURES = {"glucose", "hba1c"}


# ------------------------------------------------------------------
# Core report generator
# ------------------------------------------------------------------

def generate_health_report(drift_result: dict) -> dict:
    """
    Parse a drift result dict and return a structured health report.

    Parameters
    ----------
    drift_result : dict
        The dict returned by drift_detector.run_drift_report()

    Returns
    -------
    dict with keys:
        timestamp       : ISO-8601 string
        overall_status  : "CRITICAL" | "WARNING" | "INFO"
        alerts          : list of alert dicts
        recommendations : list of action strings
        column_summary  : dict col → {drift_score, drift_detected, stake}
    """
    alerts: list[dict] = []
    recommendations: list[str] = []

    dataset_drift     = drift_result.get("dataset_drift")
    drifted_count     = drift_result.get("drifted_columns_count")
    column_drift      = drift_result.get("column_drift", {})
    classification    = drift_result.get("classification", {})

    # ------------------------------------------------------------------
    # Evaluate dataset-level drift
    # ------------------------------------------------------------------
    if dataset_drift is True:
        alerts.append({
            "level":   "CRITICAL",
            "message": "Dataset-level drift detected between NHANES 2015-16 and 2021-23.",
            "detail":  f"{drifted_count} feature(s) drifted beyond threshold.",
        })
        recommendations.append(
            "Trigger retraining pipeline on NHANES 2021-23 as new reference data."
        )
        recommendations.append(
            "Review feature engineering pipeline — biomarker collection protocols "
            "may have changed between cycles."
        )
    elif dataset_drift is False:
        alerts.append({
            "level":   "INFO",
            "message": "No statistically significant dataset drift detected.",
            "detail":  "Model appears stable on NHANES 2021-23 population.",
        })

    # ------------------------------------------------------------------
    # Evaluate column-level drift
    # ------------------------------------------------------------------
    column_summary: dict[str, dict] = {}
    for col, info in column_drift.items():
        is_drifted  = info.get("drift_detected", False)
        score        = info.get("drift_score")
        stake        = "HIGH" if col in _HIGH_STAKES_FEATURES else "STANDARD"
        column_summary[col] = {
            "drift_score":    score,
            "drift_detected": is_drifted,
            "stake":          stake,
        }
        if is_drifted:
            level = "CRITICAL" if stake == "HIGH" else "WARNING"
            alerts.append({
                "level":   level,
                "message": f"Column drift detected: '{col}' (stake={stake})",
                "detail":  f"p-value / score = {score:.4f}" if score is not None else "",
            })
            if stake == "HIGH":
                recommendations.append(
                    f"Investigate '{col}' shift — this directly affects diabetes "
                    "risk scoring. Verify lab protocol consistency across cycles."
                )
            else:
                recommendations.append(
                    f"Monitor '{col}' trend — consider updating input scaling "
                    "or recalibrating the model threshold."
                )

    # ------------------------------------------------------------------
    # Classification performance shift
    # ------------------------------------------------------------------
    roc_auc = classification.get("roc_auc")
    if roc_auc is not None:
        if roc_auc < 0.75:
            alerts.append({
                "level":   "WARNING",
                "message": f"Model ROC-AUC on 2021-23 population: {roc_auc:.4f}",
                "detail":  "Performance degradation versus training baseline (0.9722).",
            })
            recommendations.append(
                "Evaluate whether concept drift (changing diabetes prevalence / "
                "diagnostic criteria) is causing ROC-AUC degradation."
            )
        else:
            alerts.append({
                "level":   "INFO",
                "message": f"Model ROC-AUC on 2021-23 population: {roc_auc:.4f}",
                "detail":  "Performance remains within acceptable range.",
            })

    # ------------------------------------------------------------------
    # Default recommendation if nothing raised
    # ------------------------------------------------------------------
    if not recommendations:
        recommendations.append(
            "Continue scheduled monitoring. Next comparison: NHANES 2023-25 when available."
        )

    # ------------------------------------------------------------------
    # Overall status
    # ------------------------------------------------------------------
    severity_order = ["CRITICAL", "WARNING", "INFO"]
    levels_present = {a["level"] for a in alerts}
    overall = next(
        (s for s in severity_order if s in levels_present),
        "INFO",
    )

    report = {
        "timestamp":       datetime.now(tz=timezone.utc).isoformat(),
        "overall_status":  overall,
        "alerts":          alerts,
        "recommendations": recommendations,
        "column_summary":  column_summary,
        "classification":  classification,
    }

    _print_report(report)
    return report


# ------------------------------------------------------------------
# Console renderer
# ------------------------------------------------------------------

def _print_report(report: dict) -> None:
    status_bar = {
        "CRITICAL": "!! CRITICAL !!",
        "WARNING":  "-- WARNING   --",
        "INFO":     "   INFO       ",
    }

    print()
    print("╔" + "═" * 58 + "╗")
    print("║  P.U.L.S.E.  MODEL HEALTH REPORT" + " " * 24 + "║")
    print(f"║  Generated : {report['timestamp'][:19]} UTC" + " " * 21 + "║")
    print(f"║  Status    : {status_bar[report['overall_status']]}" + " " * 29 + "║")
    print("╠" + "═" * 58 + "╣")

    print("║  ALERTS:" + " " * 49 + "║")
    for a in report["alerts"]:
        prefix = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🟢"}.get(a["level"], "  ")
        msg = f"  {prefix} [{a['level']}] {a['message']}"
        print(f"║{msg[:58]:<58}║")
        if a.get("detail"):
            detail = f"       {a['detail']}"
            print(f"║{detail[:58]:<58}║")

    print("╠" + "═" * 58 + "╣")
    print("║  RECOMMENDATIONS:" + " " * 40 + "║")
    for i, rec in enumerate(report["recommendations"], 1):
        # Word-wrap at 54 chars
        words, line = rec.split(), ""
        lines = []
        for w in words:
            if len(line) + len(w) + 1 <= 54:
                line = (line + " " + w).strip()
            else:
                lines.append(line)
                line = w
        if line:
            lines.append(line)
        for j, ln in enumerate(lines):
            prefix = f"  {i}. " if j == 0 else "     "
            print(f"║{(prefix + ln)[:58]:<58}║")

    print("╚" + "═" * 58 + "╝")
    print()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    result = run_drift_report()
    generate_health_report(result)
