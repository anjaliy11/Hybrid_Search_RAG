"""Unit tests for rubric loading and scoring logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yaml


class TestRubricConfig:
    """Tests for evaluation rubric YAML files."""

    RUBRIC_DIR = Path("config/eval_rubrics")

    def test_rubrics_exist(self):
        """Required rubric files are present."""
        required = ["faithfulness.yaml", "relevance.yaml", "completeness.yaml"]
        for name in required:
            path = self.RUBRIC_DIR / name
            assert path.exists(), f"Missing rubric: {name}"

    def test_rubrics_valid_yaml(self):
        """All rubrics parse as valid YAML."""
        for path in self.RUBRIC_DIR.glob("*.yaml"):
            with open(path) as f:
                data = yaml.safe_load(f)
            assert isinstance(data, dict), f"{path.name} is not a dict"

    def test_rubrics_have_required_keys(self):
        """Each rubric has name, description, criteria, scoring."""
        required_keys = ["name", "description", "criteria", "scoring"]

        for path in self.RUBRIC_DIR.glob("*.yaml"):
            with open(path) as f:
                data = yaml.safe_load(f)

            for key in required_keys:
                assert key in data, f"{path.name} missing key: {key}"

    def test_rubric_criteria_is_list(self):
        """Criteria field must be a list."""
        for path in self.RUBRIC_DIR.glob("*.yaml"):
            with open(path) as f:
                data = yaml.safe_load(f)
            assert isinstance(data["criteria"], list), f"{path.name}: criteria must be list"

    def test_rubric_scoring_has_numeric_keys(self):
        """Scoring guide uses numeric keys (0.0, 0.5, 1.0, etc)."""
        for path in self.RUBRIC_DIR.glob("*.yaml"):
            with open(path) as f:
                data = yaml.safe_load(f)

            scoring = data["scoring"]
            assert isinstance(scoring, dict)
            for key in scoring:
                assert isinstance(key, (int, float)), f"{path.name}: scoring key '{key}' not numeric"


if __name__ == "__main__":
    t = TestRubricConfig()
    t.test_rubrics_exist()
    t.test_rubrics_valid_yaml()
    t.test_rubrics_have_required_keys()
    t.test_rubric_criteria_is_list()
    t.test_rubric_scoring_has_numeric_keys()
    print(" All rubric tests passed")