from __future__ import annotations

import json
from pathlib import Path

from brain_sidecar.eval.models import EvalReport


def write_json_report(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_report(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Brain Sidecar Eval Report: {report.session_id}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Passed | {str(report.report_passed).lower()} |",
        f"| Input segments | {report.input_segment_count} |",
        f"| Consolidated segments | {report.consolidated_segment_count} |",
        f"| Collapsed segments | {report.collapsed_segment_count} |",
        f"| Generated cards | {report.generated_card_count} |",
        f"| Accepted cards | {report.accepted_card_count} |",
        f"| Suppressed cards | {report.suppressed_card_count} |",
        f"| Memory cards | {report.memory_card_count} |",
        f"| Cards per 5 min | {report.cards_per_5min:.2f} |",
        f"| Cards per 100 segments | {report.cards_per_100_segments:.2f} |",
        f"| Source id coverage | {report.percent_cards_with_source_ids:.2%} |",
        f"| Evidence quote coverage | {report.percent_cards_with_evidence_quote:.2%} |",
        "",
        "## Accepted Cards",
        "",
    ]
    if report.accepted_cards:
        for card in report.accepted_cards:
            lines.extend(
                [
                    f"- **{card.get('title', '')}** ({card.get('category', '')}, {card.get('priority', '')})",
                    f"  - Evidence: {card.get('evidence_quote', '')}",
                ]
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Suppressed Reasons", ""])
    if report.suppression_reasons:
        for reason, count in sorted(report.suppression_reasons.items()):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- None")
    lines.extend(["", "## Expected Topics", ""])
    lines.append(f"- Hits: {', '.join(report.expected_topic_hits) or 'none'}")
    lines.append(f"- Misses: {', '.join(report.expected_topic_misses) or 'none'}")
    lines.extend(["", "## Unsupported Violations", ""])
    lines.extend(_violation_lines(report.unsupported_topic_violations))
    lines.extend(["", "## Memory Leakage", ""])
    lines.extend(_violation_lines(report.memory_leakage_violations))
    lines.extend(["", "## Recommendations", ""])
    if report.recommendations:
        lines.extend(f"- {item}" for item in report.recommendations)
    else:
        lines.append("- No immediate recommendation.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _violation_lines(violations: list[dict]) -> list[str]:
    if not violations:
        return ["- None"]
    return [f"- `{item.get('term', item.get('reason', 'violation'))}` in {item.get('card_title', 'card')}" for item in violations]
