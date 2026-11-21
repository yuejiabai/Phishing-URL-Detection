"""Offline URL parsing, domain grouping and lexical features."""

import ipaddress
import math
import re
from collections import Counter
from functools import lru_cache
from urllib.parse import parse_qsl, urlsplit

import tldextract


EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=(), cache_dir=None, include_psl_private_domains=False
)
KEYWORDS = re.compile(r"login|verify|account|secure|update", re.IGNORECASE)
HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


def ip_address(host):
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def parse_url(url):
    """Return parsed HTTP(S) URLs only, without resolving the hostname."""
    if not isinstance(url, str) or any(c.isspace() or ord(c) < 32 for c in url):
        return None
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if parsed.scheme not in {"http", "https"} or not host:
            return None
        # Reading the port also validates its range and syntax.
        parsed.port
        return parsed if normalize_host(host) else None
    except (ValueError, UnicodeError):
        return None


@lru_cache(maxsize=65536)
def normalize_host(host):
    host = host.removesuffix(".")
    address = ip_address(host)
    if address is not None:
        return str(address)
    ascii_host = host.encode("idna").decode("ascii").lower()
    if len(ascii_host) <= 253 and all(HOST_LABEL.fullmatch(p) for p in ascii_host.split(".")):
        return ascii_host
    return None


@lru_cache(maxsize=65536)
def domain_parts(host):
    normalized = normalize_host(host)
    if ip_address(normalized) is not None:
        return f"ip:{normalized}", 0
    extracted = EXTRACT(normalized)
    if extracted.domain and extracted.suffix:
        count = len(extracted.subdomain.split(".")) if extracted.subdomain else 0
        return f"domain:{extracted.top_domain_under_public_suffix}", count
    return "unresolved", 0


def registered_group(url):
    parsed = parse_url(url)
    return domain_parts(parsed.hostname)[0] if parsed else "unresolved"


def lexical_features(url):
    """Measure lengths, URL structure, character composition and token cues."""
    url = url if isinstance(url, str) else ""
    parsed = parse_url(url)
    host = parsed.hostname if parsed else ""
    counts = Counter(url)
    length = len(url)
    # Shannon entropy in bits per character, with zero for empty strings.
    entropy = -sum((n / length) * math.log2(n / length) for n in counts.values())
    return {
        "url_length": length,
        "hostname_length": len(host),
        "path_length": len(parsed.path) if parsed else 0,
        "query_length": len(parsed.query) if parsed else 0,
        "subdomain_count": domain_parts(host)[1] if host else 0,
        "has_ip_host": int(ip_address(host.removesuffix(".")) is not None),
        "has_at": int("@" in counts),
        "has_https": int(parsed is not None and parsed.scheme == "https"),
        "dot_count": counts["."],
        "hyphen_count": counts["-"],
        "underscore_count": counts["_"],
        "digit_count": sum(n for c, n in counts.items() if c.isdigit()),
        "special_count": sum(n for c, n in counts.items() if not c.isalnum()),
        "slash_count": counts["/"],
        "percent_count": counts["%"],
        # Repeated keys and empty values each count as separate parameters.
        "query_parameter_count": len(parse_qsl(parsed.query, keep_blank_values=True)) if parsed else 0,
        "entropy": entropy,
        "has_keyword": int(bool(KEYWORDS.search(url))),
        "has_fragment": int(parsed is not None and bool(parsed.fragment)),
    }
