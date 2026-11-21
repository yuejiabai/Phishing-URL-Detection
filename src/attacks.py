"""Deterministic path-only edits for paired offline robustness evaluation."""

from src.features import parse_url


def substitute(path):
    replacements = {"l": "1", "o": "0", "i": "1"}
    for index, char in enumerate(path):
        # Preserve percent escapes, including the two characters after '%'.
        if char in replacements and "%" not in path[max(0, index - 2):index]:
            return path[:index] + replacements[char] + path[index + 1:]
    return path


def append_segments(path, segments):
    separator = "" if path.endswith("/") else "/"
    return path + separator + segments


METHODS = {
    "Character substitution": substitute,
    "Path extension": lambda path: append_segments(path, "research/example/archive/2024"),
    "Benign keyword injection": lambda path: append_segments(path, "about/help/contact"),
}


def perturb(url, method):
    parsed = parse_url(url)
    if parsed is None:
        return None, "invalid_original"
    path = METHODS[method](parsed.path)
    if path == parsed.path:
        return None, "no_eligible_edit"
    # Preserve the original scheme spelling and any empty query or fragment markers.
    start = url.index("://") + 3 + len(parsed.netloc)
    end = start + len(parsed.path)
    candidate = url[:start] + path + url[end:]
    edited = parse_url(candidate)
    if edited != parsed._replace(path=path):
        return None, "invalid_result"
    return candidate, "valid"
