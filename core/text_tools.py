"""Small, dependency-free helpers shared by text-oriented utility nodes."""

from __future__ import annotations


def decode_separator(value: str) -> str:
    """Decode common visible escape sequences without corrupting Unicode text."""

    text = str(value)
    escapes = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\"}
    output: list[str] = []
    index = 0
    while index < len(text):
        if text[index] == "\\" and index + 1 < len(text):
            replacement = escapes.get(text[index + 1])
            if replacement is not None:
                output.append(replacement)
                index += 2
                continue
        output.append(text[index])
        index += 1
    return "".join(output)
