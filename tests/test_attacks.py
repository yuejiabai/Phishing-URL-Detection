from urllib.parse import urlsplit

import pytest

from src.attacks import METHODS, perturb


@pytest.mark.parametrize("method", METHODS)
def test_perturbation_is_deterministic_and_path_only(method):
    original = "https://example.org/login?foo=bar#fragment"
    edited, status = perturb(original, method)
    assert (edited, status) == perturb(original, method)
    assert status == "valid"
    before, after = urlsplit(original), urlsplit(edited)
    assert before._replace(path=after.path) == after
    assert before.path != after.path


def test_substitution_reports_no_eligible_edit():
    assert perturb("https://example.org/%20", "Character substitution") == (
        None, "no_eligible_edit"
    )


def test_invalid_perturbation_input():
    assert perturb("https://[broken", "Path extension") == (None, "invalid_original")


def test_extension_preserves_original_prefix_and_empty_component_markers():
    original = "HTTPS://example.org/login//?#"
    edited, status = perturb(original, "Path extension")
    assert status == "valid"
    assert edited == "HTTPS://example.org/login//research/example/archive/2024?#"
