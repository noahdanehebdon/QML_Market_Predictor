import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTATION_INDEX = REPOSITORY_ROOT / "docs" / "README.md"


def test_documentation_index_links_resolve():
    content = DOCUMENTATION_INDEX.read_text(encoding="utf-8")
    relative_links = re.findall(r"\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)", content)

    assert relative_links
    for link in relative_links:
        assert (DOCUMENTATION_INDEX.parent / link).is_file(), link


def test_repository_metadata_and_professional_entry_points_exist():
    for path in (
        ".editorconfig",
        ".gitattributes",
        ".github/PULL_REQUEST_TEMPLATE.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "docs/README.md",
    ):
        assert (REPOSITORY_ROOT / path).is_file(), path

    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/README.md" in readme
    assert "## Milestone" not in readme
