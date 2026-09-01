"""Configuration loading fails early and says why."""

from __future__ import annotations

import pytest

from alpinemet.config import (
    NAMED_DOMAINS,
    ConfigError,
    load_config,
    load_paths,
)
from alpinemet.io.subset import ALPINE_DOMAIN

MINIMAL = """
name: test-analysis
datasets:
  - key: era5
    paths: "ERA5/era5_2020.nc"
"""


def _write(tmp_path, text, name="analysis.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Analysis configuration
# --------------------------------------------------------------------------


def test_a_minimal_configuration_loads(tmp_path):
    config = load_config(_write(tmp_path, MINIMAL))
    assert config.name == "test-analysis"
    assert len(config.datasets) == 1
    assert config.datasets[0].key == "era5"


def test_the_domain_defaults_to_the_alpine_box(tmp_path):
    config = load_config(_write(tmp_path, MINIMAL))
    assert config.domain == ALPINE_DOMAIN


def test_a_named_domain_resolves(tmp_path):
    text = MINIMAL + "\ndomain: alpine\n"
    assert load_config(_write(tmp_path, text)).domain == NAMED_DOMAINS["alpine"]


def test_explicit_bounds_are_accepted(tmp_path):
    text = MINIMAL + """
domain:
  lon_min: 9.0
  lon_max: 18.0
  lat_min: 46.0
  lat_max: 49.5
  name: Austria
"""
    domain = load_config(_write(tmp_path, text)).domain
    assert (domain.lon_min, domain.lat_max) == (9.0, 49.5)
    assert domain.name == "Austria"


def test_the_period_is_parsed(tmp_path):
    text = MINIMAL + """
period:
  start: "2020-01-01"
  end: "2020-12-31"
"""
    config = load_config(_write(tmp_path, text))
    assert config.start == "2020-01-01"
    assert config.end == "2020-12-31"


def test_options_pass_through_untouched(tmp_path):
    text = MINIMAL + """
options:
  dunkelflaute:
    wind_threshold: 0.15
"""
    config = load_config(_write(tmp_path, text))
    assert config.options["dunkelflaute"]["wind_threshold"] == 0.15


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_a_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="No configuration file"):
        load_config(tmp_path / "absent.yaml")


def test_a_missing_name_is_reported(tmp_path):
    text = """
datasets:
  - key: era5
    paths: "x.nc"
"""
    with pytest.raises(ConfigError, match="missing required key 'name'"):
        load_config(_write(tmp_path, text))


def test_an_unknown_dataset_key_is_caught_at_load_time(tmp_path):
    """The point of eager validation: fail now, not after a two-hour load."""
    text = """
name: broken
datasets:
  - key: era6
    paths: "x.nc"
"""
    with pytest.raises(ConfigError) as excinfo:
        load_config(_write(tmp_path, text))

    message = str(excinfo.value)
    assert "Unknown dataset" in message
    assert "datasets[0]" in message, "the offending entry must be identified"
    assert "era5" in message, "the available keys must be listed"


def test_an_empty_dataset_list_is_rejected(tmp_path):
    text = "name: empty\ndatasets: []\n"
    with pytest.raises(ConfigError, match="at least one product"):
        load_config(_write(tmp_path, text))


def test_a_dataset_without_paths_is_rejected(tmp_path):
    text = """
name: broken
datasets:
  - key: era5
"""
    with pytest.raises(ConfigError, match="missing required key 'paths'"):
        load_config(_write(tmp_path, text))


def test_an_unknown_named_domain_lists_the_alternatives(tmp_path):
    text = MINIMAL + "\ndomain: pyrenees\n"
    with pytest.raises(ConfigError, match="unknown domain"):
        load_config(_write(tmp_path, text))


def test_an_inverted_domain_is_rejected(tmp_path):
    text = MINIMAL + """
domain:
  lon_min: 18.0
  lon_max: 4.0
  lat_min: 43.0
  lat_max: 49.0
"""
    with pytest.raises(ConfigError, match="invalid domain"):
        load_config(_write(tmp_path, text))


def test_an_inverted_period_is_rejected(tmp_path):
    text = MINIMAL + """
period:
  start: "2020-12-31"
  end: "2020-01-01"
"""
    with pytest.raises(ConfigError, match="is after end"):
        load_config(_write(tmp_path, text))


def test_malformed_yaml_names_the_file(tmp_path):
    path = _write(tmp_path, "name: [unclosed\n")
    with pytest.raises(ConfigError, match="could not parse YAML"):
        load_config(path)


def test_a_non_mapping_document_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="expected a mapping"):
        load_config(_write(tmp_path, "- just\n- a list\n"))


# --------------------------------------------------------------------------
# Paths file
# --------------------------------------------------------------------------


def test_paths_are_loaded(tmp_path):
    path = _write(tmp_path, "data_root: /data\noutput_root: /out\n", name="paths.yaml")
    paths = load_paths(path)
    assert paths.data_root.as_posix().endswith("/data")
    assert paths.output_root.as_posix().endswith("/out")


def test_dataset_paths_resolve_against_the_data_root(tmp_path):
    path = _write(tmp_path, "data_root: /data\n", name="paths.yaml")
    resolved = load_paths(path).resolve("ERA5/era5_2020.nc")
    assert resolved.as_posix().endswith("/data/ERA5/era5_2020.nc")


def test_absolute_dataset_paths_are_left_alone(tmp_path):
    path = _write(tmp_path, "data_root: /data\n", name="paths.yaml")
    absolute = tmp_path / "elsewhere.nc"
    assert load_paths(path).resolve(absolute) == absolute


def test_a_missing_paths_file_points_at_the_example(tmp_path):
    with pytest.raises(FileNotFoundError, match="paths.example.yaml"):
        load_paths(tmp_path / "paths.yaml")


def test_a_paths_file_without_data_root_is_rejected(tmp_path):
    path = _write(tmp_path, "output_root: /out\n", name="paths.yaml")
    with pytest.raises(ConfigError, match="missing required key 'data_root'"):
        load_paths(path)


def test_the_shipped_example_is_valid(tmp_path):
    """The example must actually load, or a fresh clone starts broken."""
    from pathlib import Path

    example = Path("configs/paths.example.yaml")
    if not example.exists():  # pragma: no cover - only when run from elsewhere
        pytest.skip("run from the repository root")

    paths = load_paths(example)
    assert paths.data_root is not None


# --------------------------------------------------------------------------
# The configurations shipped with the repository
# --------------------------------------------------------------------------


def _shipped_configs():
    from pathlib import Path

    directory = Path("configs")
    if not directory.exists():  # pragma: no cover - only when run from elsewhere
        return []
    # paths*.yaml holds machine-specific locations, not an analysis specification.
    return sorted(p for p in directory.glob("*.yaml") if not p.name.startswith("paths"))


@pytest.mark.parametrize("path", _shipped_configs(), ids=lambda p: p.name)
def test_every_shipped_configuration_is_valid(path):
    """A broken example config is a broken first impression of the repository."""
    config = load_config(path)
    assert config.name
    assert config.datasets
