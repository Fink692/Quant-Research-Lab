"""Report serialization helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_dataframe(frame: pd.DataFrame | pd.Series, path: str | Path) -> Path:
    """Save a DataFrame or Series to CSV and return the path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frame, pd.Series):
        frame.to_frame().to_csv(output)
    else:
        frame.to_csv(output)
    return output


def write_markdown_report(title: str, sections: dict[str, str], path: str | Path) -> Path:
    """Write a simple Markdown report from section text."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for heading, body in sections.items():
        lines.extend([f"## {heading}", "", body.strip(), ""])
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
