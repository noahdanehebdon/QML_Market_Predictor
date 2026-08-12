import re
from pathlib import Path

METHODOLOGY = Path("docs/methodology.md")


def test_methodology_covers_issue_sections_and_claim_boundaries():
    content = METHODOLOGY.read_text(encoding="utf-8")

    for section in (
        "## Purpose and research question",
        "## Prediction target",
        "## Research universe and data",
        "## Feature construction and information timing",
        "## Classical baselines",
        "## Quantum models",
        "### Quantum convolutional neural network",
        "## Chronological experimental design",
        "## Portfolio simulation",
        "## Regime analysis",
        "## Interpretation rules and limitations",
    ):
        assert section in content

    assert "does not demonstrate quantum advantage" in content
    assert "does not support claims of future profitability" in content
    assert "IBM Quantum Runtime" in content
    assert "fixed, locally trained VQC parameters" in content
    assert "must not be described as hardware validation" in content


def test_methodology_relative_links_resolve():
    content = METHODOLOGY.read_text(encoding="utf-8")
    links = re.findall(r"\[[^]]+\]\(([^)]+)\)", content)

    assert links
    for link in links:
        assert not link.startswith(("http://", "https://"))
        assert (METHODOLOGY.parent / link).exists(), link
