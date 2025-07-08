"""
Utilities to emit Markdown and CSV summary reports for modal‐analysis outputs.
"""

import csv
from typing import Mapping, Union


def write_summary_csv(path: str, metrics: Mapping[str, Union[str, int, float]]) -> None:
    """
    Write out a two‐column CSV of key→value.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])


def write_modal_markdown_report(
    path: str,
    model_name: str,
    taxonomy: str,
    metrics: Mapping[str, Union[str, int, float]],
    file_links: Mapping[str, str],
) -> None:
    """
    Write a Markdown report that:
      - Lists the metrics in prose
      - References generated HTML/CSV artifacts
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Modal Analysis Report: {model_name} ({taxonomy})\n\n")
        f.write("## Summary Metrics\n")
        for k, v in metrics.items():
            f.write(f"- **{k.replace('_', ' ').title()}**: {v}\n")
        f.write("\n## Artifacts\n")
        for label, link in file_links.items():
            nice = label.replace("_", " ").title()
            f.write(f"- **{nice}**: [{link}]({link})\n")
