"""Keep research workflows out of the installable library surface."""

from pathlib import Path


def test_models_only_contains_reusable_building_blocks():
    root = Path(__file__).parents[1]
    models = root / "src" / "festim_microstructure" / "models"
    assert {path.stem for path in models.glob("*.py")} == {
        "__init__",
        "fisher",
        "properties",
        "resolved",
    }


def test_study_drivers_are_runnable_examples():
    root = Path(__file__).parents[1]
    examples = root / "examples"
    for name in (
        "gb_homogenisation.py",
        "gb_validation.py",
        "gb_figures.py",
        "li2022_fig4.py",
        "li2022_fig4_codim.py",
    ):
        assert (examples / name).is_file()
