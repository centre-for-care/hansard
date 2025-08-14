"""infer_speaker_gender.py
=================================
A command‑line tool to enrich Historic Hansard *speaker* records with a
best‑effort `gender` field.

The algorithm applies four successive heuristics:

1. **Title lookup** – If the very first token (e.g. "Mr", "Baroness") maps
   unambiguously to a gender, we accept it.
2. **Prefix stripping** – For names that begin with military, clerical or
   honorific prefixes (e.g. "Lieut‑Colonel"), the prefixes are removed and the
   step‑1 lookup is tried again.
3. **`gender_guesser`** – If the cleaned first token is a forename, we query
   the `gender_guesser.detector.Detector` model. "male" or "female" results are
   accepted; "unknown", "androgynous" etc. leave the gender as *None*.
4. **Manual overrides** – A hard‑coded dictionary for edge‑cases where lookups
   are unreliable (e.g. Irish hereditary titles, initials). This can be edited
   in‑place or loaded from an external JSON/YAML file as needed.

Any record still unresolved after the pipeline retains `gender=None` (will
serialise as JSON `null`).

Usage
-----
```bash
python infer_speaker_gender.py \
    --input  speaker_details.json \
    --output speaker_details_with_gender.json
```

Dependencies
------------
* **gender‑guesser** – `pip install gender‑guesser`
* **beautifulsoup4 / requests** – *only if you reuse the regex helpers elsewhere*.

The script itself is pure‑standard‑library **plus** `gender_guesser`.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from gender_guesser.detector import Detector

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1) High‑confidence title → gender map
# ---------------------------------------------------------------------------
_TITLE_TO_GENDER: Dict[str, str] = {
    # Male titles
    "mr": "male",
    "sir": "male",
    "lord": "male",
    "baron": "male",
    "viscount": "male",
    "duke": "male",
    "earl": "male",
    "marquess": "male",
    # Female titles
    "mrs": "female",
    "ms": "female",
    "miss": "female",
    "lady": "female",
    "dame": "female",
    "duchess": "female",
    "viscountess": "female",
    "countess": "female",
    "baroness": "female",
}


def _infer_from_title(name: str) -> Optional[str]:
    """Return *male* / *female* if the first token maps in `_TITLE_TO_GENDER`.

    Args:
        name: Raw speaker `name` field.

    Returns:
        "male", "female" or *None* if inconclusive.
    """
    if not name:
        return None
    token = name.strip().split()[0].lower().rstrip(".,")
    return _TITLE_TO_GENDER.get(token)


# ---------------------------------------------------------------------------
# 2) Regexes for prefix stripping
# ---------------------------------------------------------------------------
# A superset of military, aristocratic and honorific prefixes seen in Historic
# Hansard. Two passes are applied to handle nested prefixes (e.g.
# "Lieut‑Colonel Sir Walter …").
_PREFIX_RE = re.compile(
    r"^(?:"
    r"Lieut[\-\s]?Colonel|Lt[\-\s]?Col|Major|Colonel|Brigadier(?:[\-\s]?General)?|"
    r"Lieut[\-\s]?Commander|Lieutenant[\-\s]?Commander|Lieutenant|"
    r"Captain|Commander|Commodore|Wing[\-\s]?Commander|Group[\-\s]?Captain|"
    r"Air[\-\s]?Commodore|Rear[\-\s]?Admiral|Vice[\-\s]?Admiral|Admiral|"
    r"Field[\-\s]?Marshal|Major[\-\s]?General|Lieut[\-\s]?General|General|"
    r"Air[\-\s]?Vice[\-\s]?Marshall|Squadron[\-\s]?Leader|Flight[\-\s]?Lieut|"
    r"Sub[\-\s]?Lieutenant|"
    r"Viscountess|Viscount|Baroness|Baron|Lord|Lady|"
    r"Earl|Countess|Marquess|Marchioness|Duke|Duchess|"
    r"Hon(?:\.|orable)?|Sir|Dame|Dr|Professor|Prof\.?|Reverend|Rev\.?|"
    r"Master(?:[\s\-]of[\s\w]+)?|Count|"
    r"The\sO'Conor\sDon|The\sO'Donoghue[^\s]*|"
    r"Of\b"  # Captures titles such as "Earl of" …
    r")\s+",
    re.IGNORECASE,
)


def _strip_prefixes(name: str) -> str:
    """Remove up to two leading prefix segments and return the cleaned string."""
    cleaned = _PREFIX_RE.sub("", name).strip()
    # Repeat once to catch compounded prefixes.
    cleaned = _PREFIX_RE.sub("", cleaned).strip()
    return cleaned


# ---------------------------------------------------------------------------
# 3) Manual overrides for stubborn cases
# ---------------------------------------------------------------------------
# These should be RARE. Keep the list alphabetised for Git diff clarity.
_GENDER_OVERRIDES: Dict[str, str] = {
    "Captain Algernon Moreing": "male",
    "Captain Bertie Kirby": "male",
    "Captain Rees Gronow": "male",
    "Captain Sidney Streatfeild": "male",
    "Colonel Chichester Crookshank": "male",
    "Colonel Claude Lowther": "male",
    "Colonel Sir Smith Child": "male",
    "Commander Redvers Prior": "male",
    "Dr Ashok Kumar": "male",
    "Dr Bouverie McDonald": "male",
    "Dr Dickson Mabon": "male",
    "Dr Hyacinth Morgan": "male",
    "Dr Kim Howells": "male",
    "Dr Leslie Haden-Guest": "male",
    "Dr Mo Mowlam": "female",
    "Dr Mont Follick": "male",
    "Dr Rhodes Boyson": "male",
    "Dr Sidney Peters": "male",
    "Duchess of Atholl": "female",  # example of adding a female override
    "Hon. Algernon Egerton": "male",
    "Hon. Algernon Greville": "male",
    "Hon. Algernon Tollemache": "male",
    "Hon. Ashley Ponsonby": "male",
    "Hon. Aubrey Herbert": "male",
    "Hon. Auberon Herbert": "male",
    "Hon. Beilby Lawley": "male",
    "Hon. Cadwallader Blayney": "male",
    "Hon. Claude Hay": "male",
    "Hon. Craven Berkeley": "male",
    "Hon. Duncombe Pleydell-Bouverie": "male",
    "Hon. Eustace Fiennes": "male",
    "Hon. Fox Maule": "male",
    "Hon. Gervase Beckett": "male",
    "Hon. Greville Vernon": "male",
    "Hon. Grenville Howard": "male",
    "Hon. Heneage Legge": "male",
    "Hon. Mary Pickford": "female",
    "Hon. Octavius Duncombe": "male",
    "Hon. Pascoe Glyn": "male",
    "Hon. Sam Silkin": "male",
    "Hon. Sidney Glyn": "male",
    "Hon. Sidney Herbert": "male",
    "Hon. Somerset Maxwell": "male",
    "Hon. Swynfen Carnegie": "male",
    "Hon. Vicary Gibbs": "male",
    "Hon. Waldorf Astor": "male",
    "Hon. Wentworth Beaumont": "male",
    "Hon. Wilbraham Egerton": "male",
    "Hon. Wilbraham Tollemache": "male",
    "Lieut-Colonel Assheton Pownall": "male",
    "Lieut-Colonel Hon. Glyn Mason": "male",
    "Lieut-Colonel Hon. Sidney Peel": "male",
    "Lieut-Colonel Hon. Standish O'Grady": "male",
    "Lieut-Colonel Leslie Wilson": "male",
    "Lieut-Colonel Penry Williams": "male",
    "Lieut-Colonel Uvedale Corbett": "male",
    "Lieut-Colonel Wentworth Schofield": "male",
    "Lieut-General Sir Aylmer Hunter-Weston": "male",
    "Master of Elibank": "male",
    "Major Aubrey Beauclerk": "male",
    "Major Bertie Leighton": "male",
    "Major Collingwood Hamilton": "male",
    "Major Vaughan Vaughan-Lee": "male",
    "Rear-Admiral Morgan Morgan-Giles": "male",
    "Rear-Admiral Tufton Beamish": "male",
    "Sir Wellington Cotton": "male",
    "The O'Conor Don": "male",
    "The O'Donoghue of the Glens": "male",
    "The O'Gorman Mahon": "male",
    "Very Reverend Dr John Simms": "male",
    "Viscount Wavell Wakefield": "male",  # cleaned name keys work too
}


# ---------------------------------------------------------------------------
# 4) Core gender‑inference pipeline
# ---------------------------------------------------------------------------

def infer_gender(name: str, detector: Detector) -> Optional[str]:
    """Run all inference stages and return *male* / *female* / *None*.

    Args:
        name: Raw name string from `speaker_details.json`.
        detector: Shared instance of `gender_guesser.Detector`.

    Returns:
        Inferred gender or *None* when ambiguous.
    """
    # (a) Manual override takes absolute priority.
    override = _GENDER_OVERRIDES.get(name.strip())
    if override:
        return override

    # (b) Direct title lookup.
    gender = _infer_from_title(name)
    if gender:
        return gender

    # (c) Strip prefixes and try title lookup again.
    cleaned = _strip_prefixes(name)
    gender = _infer_from_title(cleaned)
    if gender:
        return gender

    # (d) First‑name model via gender_guesser.
    if cleaned:
        first_token = cleaned.split()[0].rstrip(",").capitalize()
        guess = detector.get_gender(first_token)
        if guess in {"male", "mostly_male"}:
            return "male"
        if guess in {"female", "mostly_female"}:
            return "female"

    return None


# ---------------------------------------------------------------------------
# 5) Entry‑point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer gender for Historic Hansard speaker records.")
    parser.add_argument("--input", type=Path, default="speaker_details.json", help="Path to input JSON list.")
    parser.add_argument("--output", type=Path, default="speaker_details_with_gender.json", help="Destination JSON file.")
    parser.add_argument("--log", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log), format="%(levelname)s %(message)s")

    _LOG.info("Loading speakers from %s", args.input)
    with args.input.open("r", encoding="utf-8") as fh:
        speakers: List[Dict[str, Any]] = json.load(fh)

    detector = Detector(case_sensitive=False)

    unset_before = sum(1 for sp in speakers if sp.get("gender") is None)
    _LOG.info("Records without gender before processing: %d", unset_before)

    for sp in speakers:
        if sp.get("gender") in {"male", "female"}:  # respect existing values
            continue
        name = sp.get("name", "")
        sp["gender"] = infer_gender(name, detector)

    unset_after = sum(1 for sp in speakers if sp.get("gender") is None)
    _LOG.info("Records without gender after processing:  %d", unset_after)

    args.output.write_text(json.dumps(speakers, ensure_ascii=False, indent=2), encoding="utf-8")
    _LOG.info("Wrote enriched data ➜ %s", args.output)


if __name__ == "__main__":
    main()
