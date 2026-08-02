"""Output schema and parsers for the targeted-extraction task.

The model is asked for three things per speech: a boolean ``mentions_topic``,
a list of free-text ``subthemes`` (inductive sub-topics, in the model's own
words), and an ``evidence_quote``. The schema is deliberately small — every
extra free-form field is a new source of variance unrelated to the factors
under study.

Sub-themes are an *open* vocabulary: we do not constrain or canonicalise them
here. Comparability across prompt variants is recovered downstream by
embedding the phrases (see metrics.py), not by string matching. Parsing only
does light hygiene (trim, de-dup, cap), preserving the surface form so the
embedding step sees what the model actually produced.

Two parsers keep the "output format" factor honest:

    parse_json  - strict-ish JSON extraction (the JSON-format condition)
    parse_free  - tolerant natural-language extraction (the no-format condition)

Both return the same ``Extraction`` and record ``parse_ok`` / ``parse_error``
so parser failures are measurable — a real cost of dropping the format
instruction, not silently hidden.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Extraction:
    """Parsed model output for one speech under one prompt variant.

    ``presence_inferred`` marks free-format rows where ``mentions_topic`` was
    not stated by the model but inferred from the fact that it listed themes.
    Analyses comparing formats must be able to exclude these rows: counting
    them as positives is a *parser* decision, not a model output, and it
    inflated the free-vs-JSON presence gap in the pilot by roughly half.

    ``subthemes_raw`` keeps every theme the parser recovered regardless of the
    presence verdict; ``subthemes`` remains the verdict-gated list (empty when
    ``mentions_topic`` is falsy) that downstream metrics consume.
    """

    mentions_topic: bool | None = None
    subthemes: list[str] = field(default_factory=list)
    evidence_quote: str = ""
    parse_ok: bool = False
    parse_error: str | None = None
    presence_inferred: bool = False
    subthemes_raw: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_themes(raw: Any, max_n: int) -> list[str]:
    """Light hygiene on free-text sub-themes: trim, drop empties, de-duplicate
    case-insensitively, cap to ``max_n``. Surface form is otherwise preserved
    for the downstream embedding step.

    Accepts a list of strings, or a single delimited string (the free-format
    parser may hand us either).
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        items = re.split(r"[\n;|]+|,(?![^(]*\))", raw)
    elif isinstance(raw, (list, tuple)):
        items = [str(x) for x in raw]
    else:
        items = [str(raw)]

    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        phrase = re.sub(r"\s+", " ", str(item)).strip().strip("\"'.-*•").strip()
        if not phrase or phrase.lower() in _NONE_WORDS:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(phrase)
        if len(out) >= max_n:
            break
    return out


# Phrases a model uses to say "no sub-themes" — must not become a theme.
_NONE_WORDS = {
    "none", "none.", "n/a", "na", "none identified", "not applicable",
    "no sub-topics", "no subthemes", "no specific sub-topics", "none found",
}


_TRUE_WORDS = {"true", "yes", "y", "1"}
_FALSE_WORDS = {"false", "no", "n", "0"}


def _coerce_bool(raw: Any) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in _TRUE_WORDS:
        return True
    if s in _FALSE_WORDS:
        return False
    return None


def _strip_code_fence(text: str) -> str:
    """Remove a leading ```json / ``` fence if the model wrapped its JSON."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t


def _find_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` block, or None.

    Models sometimes prepend prose before the JSON; a brace-matching scan is
    more robust than a greedy regex.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def parse_json(text: str, max_n: int) -> Extraction:
    """Parse the JSON-format condition output."""
    if not text or not text.strip():
        return Extraction(parse_ok=False, parse_error="empty response")

    blob = _find_json_object(_strip_code_fence(text))
    if blob is None:
        return Extraction(parse_ok=False, parse_error="no JSON object found")
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError as e:
        return Extraction(parse_ok=False, parse_error=f"json decode: {e}")
    if not isinstance(obj, dict):
        return Extraction(parse_ok=False, parse_error="JSON is not an object")

    mt = _coerce_bool(obj.get("mentions_topic"))
    subs = normalize_themes(obj.get("subthemes"), max_n)
    quote = obj.get("evidence_quote") or ""
    err = None if mt is not None else "missing/uninterpretable mentions_topic"
    return Extraction(
        mentions_topic=mt,
        subthemes=subs if mt else [],
        evidence_quote=str(quote)[:500],
        parse_ok=mt is not None,
        parse_error=err,
        subthemes_raw=subs,
    )


# A labelled answer: "Substantive ...: Yes" / "mentions_topic: false" — the
# value may be separated by markdown noise (** , spaces) after the colon.
_LABEL_ANSWER = re.compile(
    r"(?:substantive|substantively|mentions?[ _]topic)[^:\n]*:\W*"
    r"(yes|no|true|false)\b", re.IGNORECASE)

# Negated / affirmed discussion of the topic, by verb or by "(not) a
# substantive subject/topic" phrasing.
_NEG_VERB = re.compile(
    r"\bnot\s+(?:a\s+)?substantive\b"
    r"|\b(?:does not|do not|is not|are not|not)\s+(?:substantively\s+)?"
    r"(?:discuss|address|mention|concern|relate|cover|about|relevant)", re.IGNORECASE)
_POS_VERB = re.compile(
    r"\bis\s+a\s+substantive\b"
    r"|\b(?:substantively\s+)?(?:discusses|addresses|concerns|covers)\b", re.IGNORECASE)

# A leading list item: "- ...", "* ...", "1. ...", "1) ..."
_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)\s*$")


def _presence_from_text(text: str) -> bool | None:
    """Recover the yes/no presence judgement from unstructured prose.

    Order of evidence, strongest first: an explicit labelled answer, a leading
    yes/no, then negated/affirmed discussion verbs (earliest wins). Returns None
    if no signal is found; the caller may then infer presence from whether any
    sub-themes were produced.
    """
    m = _LABEL_ANSWER.search(text)
    if m:
        return m.group(1).lower() in ("yes", "true")

    head = text.lstrip().lower()
    if re.match(r"yes\b", head):
        return True
    if re.match(r"no\b", head):
        return False

    neg = _NEG_VERB.search(text)
    pos = _POS_VERB.search(text)
    if neg and pos:
        return pos.start() < neg.start()
    if neg:
        return False
    if pos:
        return True
    return None


def _extract_free_themes(text: str, max_n: int) -> list[str]:
    """Best-effort theme recovery from unstructured prose.

    Prefers an explicit bulleted/numbered list; falls back to a comma list that
    follows a cue word ("sub-topics", "aspects", "themes", "covers"). When the
    model writes a pure paragraph with no list, we return nothing — that miss
    is the genuine cost of the no-format condition and is recorded as such.
    """
    items = [m.group(1) for line in text.splitlines()
             if (m := _LIST_ITEM.match(line))]
    if items:
        return normalize_themes(items, max_n)

    cue = re.search(
        r"(?:sub-?topics?|aspects?|themes?|covers?|discusses?)\s*[:\-]\s*(.+)",
        text, flags=re.IGNORECASE)
    if cue:
        return normalize_themes(cue.group(1).split(".")[0], max_n)
    return []


def parse_free(text: str, max_n: int) -> Extraction:
    """Tolerant parser for the no-format condition.

    With no structure imposed, we recover the signal heuristically: scan for an
    explicit yes/no for presence, pull a bulleted/cued list for themes, and grab
    the first quoted span as evidence. This parser is intentionally part of the
    "format" factor — its failure rate *is* the cost of removing the format
    instruction, and we record it rather than hide it.
    """
    if not text or not text.strip():
        return Extraction(parse_ok=False, parse_error="empty response")

    mt = _presence_from_text(text)
    themes = _extract_free_themes(text, max_n)

    # Inference fallback: no explicit yes/no, but the model listed sub-themes —
    # producing themes is an affirmative signal, but it is the PARSER's
    # inference, not the model's statement. It is flagged so analyses can
    # exclude these rows: in the pilot the fallback alone accounted for ~half
    # of the apparent free-vs-JSON presence gap.
    inferred = False
    if mt is None and themes:
        mt = True
        inferred = True

    subs = themes if mt else []
    quote = ""
    m = re.search(r'"([^"]{8,400})"', text) or re.search(r"'([^']{8,400})'", text)
    if m:
        quote = m.group(1)

    err = None if mt is not None else "no yes/no signal in free text"
    return Extraction(
        mentions_topic=mt,
        subthemes=subs,
        evidence_quote=quote[:500],
        parse_ok=mt is not None,
        parse_error=err,
        presence_inferred=inferred,
        subthemes_raw=themes,
    )


def parse(text: str, output_format: str, max_n: int) -> Extraction:
    """Dispatch to the parser matching the prompt's output-format factor."""
    if output_format == "json":
        return parse_json(text, max_n)
    if output_format == "free":
        return parse_free(text, max_n)
    raise ValueError(f"unknown output_format {output_format!r}")
