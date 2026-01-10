from __future__ import annotations

import re
import unittest
from pathlib import Path


def _repo_root() -> Path:
    # tests/ -> repo file
    return Path(__file__).resolve().parents[1]


def _has_gradience_console_script_in_pyproject(pyproject: Path) -> bool:
    try:
        import tomllib  # Python 3.11+
    except Exception:
        return False

    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    # PEP 621
    scripts = (data.get("project") or {}).get("scripts") or {}
    if scripts.get("gradience") == "gradience.cli:main":
        return True

    # Poetry
    poetry_scripts = ((data.get("tool") or {}).get("poetry") or {}).get("scripts") or {}
    if poetry_scripts.get("gradience") == "gradience.cli:main":
        return True

    # PDM (supports either string or dict forms)
    pdm_scripts = ((data.get("tool") or {}).get("pdm") or {}).get("scripts") or {}
    val = pdm_scripts.get("gradience")
    if val == "gradience.cli:main":
        return True
    if isinstance(val, dict) and val.get("call") == "gradience.cli:main":
        return True

    return False


def _has_gradience_console_script_in_setup_cfg(setup_cfg: Path) -> bool:
    import configparser

    cfg = configparser.ConfigParser()
    cfg.read(setup_cfg, encoding="utf-8")

    section = None
    for candidate in ("options.entry_points", "entry_points"):
        if cfg.has_section(candidate):
            section = candidate
            break
    if section is None:
        return False

    console_scripts = cfg.get(section, "console_scripts", fallback="")
    # Accept either "gradience=..." or "gradience = ..."
    return re.search(r"(?m)^\s*gradience\s*=\s*gradience\.cli:main\s*$", console_scripts) is not None


def _has_gradience_console_script_in_setup_py(setup_py: Path) -> bool:
    txt = setup_py.read_text(encoding="utf-8", errors="ignore")
    if "gradience=gradience.cli:main" in txt:
        return True
    if "gradience = gradience.cli:main" in txt:
        return True
    # Slightly more flexible regex
    return re.search(r"gradience\s*=\s*gradience\.cli:main", txt) is not None


class TestConsoleScriptDeclared(unittest.TestCase):
    def test_console_script_entrypoint_declared(self):
        root = _repo_root()
        pyproject = root / "pyproject.toml"
        setup_cfg = root / "setup.cfg"
        setup_py = root / "setup.py"

        found_packaging_file = pyproject.exists() or setup_cfg.exists() or setup_py.exists()

        found = False
        if pyproject.exists():
            found |= _has_gradience_console_script_in_pyproject(pyproject)
        if setup_cfg.exists():
            found |= _has_gradience_console_script_in_setup_cfg(setup_cfg)
        if setup_py.exists():
            found |= _has_gradience_console_script_in_setup_py(setup_py)

        self.assertTrue(
            found_packaging_file,
            "No packaging config found (pyproject.toml / setup.cfg / setup.py). "
            "Add one and declare a console script entrypoint for gradience.",
        )

        self.assertTrue(
            found,
            (
                "Console script 'gradience' is not declared in packaging config.\n"
                "Add one of:\n"
                "  pyproject.toml (PEP 621):\n"
                "    [project.scripts]\n"
                "    gradience = \"gradience.cli:main\"\n"
                "\n"
                "  pyproject.toml (Poetry):\n"
                "    [tool.poetry.scripts]\n"
                "    gradience = \"gradience.cli:main\"\n"
                "\n"
                "  setup.cfg:\n"
                "    [options.entry_points]\n"
                "    console_scripts =\n"
                "      gradience=gradience.cli:main\n"
                "\n"
                "  setup.py:\n"
                "    entry_points={'console_scripts': ['gradience=gradience.cli:main']}\n"
            ),
        )


if __name__ == "__main__":
    unittest.main()
