import re


def postprocess_vlm_text(vlm_text: str) -> tuple[str, bool]:
    """Post-processes VLM OCR text: dehyphenation, heading normalization, token loop removal."""
    if not vlm_text:
        return ("", False)

    original = vlm_text
    text = vlm_text

    text = re.sub(r"-\s*\n\s*(?=[a-zA-Z])", "", text)
    text = _glue_multiline_headings(text)
    text = _prepend_heading_marker(text)
    text = re.sub(r"([.\-_=*~#>|])\1{29,}", "", text)
    text = re.sub(r"\b(\w+)\b(?:\s+\1\b){9,}", r"\1", text, flags=re.IGNORECASE)

    return (text, text != original)


def _is_all_caps_line(line: str) -> bool:
    has_letter = any(c.isalpha() for c in line)
    return has_letter and all(not c.isalpha() or c.isupper() for c in line)


def _prepend_heading_marker(text: str) -> str:
    lines = text.split("\n")
    first_nonempty_idx = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first_nonempty_idx is None:
        return text

    first_line = lines[first_nonempty_idx].strip()
    if _is_all_caps_line(first_line) and not first_line.startswith("#"):
        lines[first_nonempty_idx] = "# " + lines[first_nonempty_idx]
        return "\n".join(lines)

    return text


def _glue_multiline_headings(text: str) -> str:
    """Joins consecutive all-caps lines into single lines."""
    lines = text.split("\n")
    result_lines: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if _is_all_caps_line(line.strip()) and line.strip():
            heading_parts = [line.strip()]
            j = i + 1

            while j < len(lines):
                if lines[j].strip() == "":
                    k = j + 1
                    while k < len(lines) and lines[k].strip() == "":
                        k += 1
                    if k < len(lines) and _is_all_caps_line(lines[k].strip()):
                        j = k
                        heading_parts.append(lines[j].strip())
                        j += 1
                    else:
                        break
                elif _is_all_caps_line(lines[j].strip()):
                    heading_parts.append(lines[j].strip())
                    j += 1
                else:
                    break

            if len(heading_parts) > 1:
                result_lines.append(" ".join(heading_parts))
            else:
                result_lines.append(line)
            i = j
        else:
            result_lines.append(line)
            i += 1

    return "\n".join(result_lines)
