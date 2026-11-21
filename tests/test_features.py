import pytest

from src.features import lexical_features, registered_group


def test_lexical_features():
    url = "https://a.b.example.co.uk/login_2?verify=&x=1"
    features = lexical_features(url)
    assert features["url_length"] == len(url)
    assert features["hostname_length"] == len("a.b.example.co.uk")
    assert features["path_length"] == len("/login_2")
    assert features["digit_count"] == 2
    assert features["query_parameter_count"] == 2
    assert features["has_keyword"] == features["has_https"] == 1
    assert features["entropy"] > 0
    assert len(features) == 19


@pytest.mark.parametrize("url", ["http://192.0.2.1/", "https://[2001:db8::1]:443/"])
def test_ip_detection(url):
    assert lexical_features(url)["has_ip_host"] == 1
    assert lexical_features(url)["subdomain_count"] == 0
    assert registered_group(url).startswith("ip:")


def test_subdomains_for_multilevel_suffix():
    url = "https://a.b.example.co.uk/"
    assert lexical_features(url)["subdomain_count"] == 2
    assert registered_group(url) == registered_group("http://EXAMPLE.CO.UK./")


@pytest.mark.parametrize("url", ["", None, "https://[bad", "https://example.com:99999", "bad"])
def test_invalid_input(url):
    features = lexical_features(url)
    assert features["hostname_length"] == features["has_ip_host"] == 0
    assert registered_group(url) == "unresolved"
