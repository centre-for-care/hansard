#!/usr/bin/env python3
"""infer_speaker_gender.py
====================================================
Single‑file replica of the *ad‑hoc notebook logic* you
shared, wrapped into a clean script.

Pipeline
--------
1. **Direct title lookup** via a small `TITLE_TO_GENDER` table.
2. **Prefix stripping** (three passes) and title lookup again.
3. **`gender_guesser`** strict‑mode (`male`/`female` only).
4. **Manual overrides** — *list generated with GPT at the end of the process*.

The script prints progress stats after each stage and writes the enriched list
back to disk.
"""
from __future__ import annotations

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional

from gender_guesser.detector import Detector

# ---------------------------------------------------------------------------
# 0) Configurable paths -------------------------------------------------------
# ---------------------------------------------------------------------------
INPUT_PATH = Path("speaker_details.json")
OUTPUT_PATH = Path("speaker_details_with_gender.json")

# ---------------------------------------------------------------------------
# 1) Title → gender table (high‑confidence)
# ---------------------------------------------------------------------------
TITLE_TO_GENDER: Dict[str, str] = {
    # male
    "mr": "male", "sir": "male", "lord": "male", "baron": "male",
    "viscount": "male", "duke": "male", "earl": "male", "marquess": "male",
    # female
    "mrs": "female", "ms": "female", "miss": "female", "lady": "female",
    "dame": "female", "duchess": "female", "viscountess": "female",
    "countess": "female", "baroness": "female",
}


def _title_lookup(name: str) -> Optional[str]:
    if not name:
        return None
    token = name.strip().split()[0].lower().rstrip(".,")
    return TITLE_TO_GENDER.get(token)


# ---------------------------------------------------------------------------
# 2) Prefix regex & stripping helper (three passes to match NB)
# ---------------------------------------------------------------------------
PREFIX_RE = re.compile(
    r'^(?:'
    r'Lieut[\-\s]?Colonel|Lt[\-\s]?Col|Major|Colonel|Brigadier(?:[\-\s]?General)?|'
    r'Lieut[\-\s]?Commander|Lieutenant[\-\s]?Commander|Lieutenant|'
    r'Captain|Commander|Commodore|Wing[\-\s]?Commander|Group[\-\s]?Captain|'
    r'Air[\-\s]?Commodore|Rear[\-\s]?Admiral|Vice[\-\s]?Admiral|Admiral|'
    r'Field[\-\s]?Marshal|Major[\-\s]?General|Lieut[\-\s]?General|General|'
    r'Air[\-\s]?Vice[\-\s]?Marshall|Squadron[\-\s]?Leader|Flight[\-\s]?Lieut|'
    r'Sub[\-\s]?Lieutenant|'
    r'Viscountess|Viscount|Baroness|Baron|Lord|Lady|'
    r'Earl|Countess|Marquess|Marchioness|Duke|Duchess|'
    r'Hon(?:\\.|orable)?|Sir|Dame|Dr|Professor|Prof\\.?|Reverend|Rev\\.?|'
    r'Master(?:[\s\-]of[\s\w]+)?|Count|'
    r'The\sO\'Conor\sDon|The\sO\'Donoghue[^\s]*|'
    r'Of\b'
    r')\s+', re.IGNORECASE
)


def _strip_prefixes(name: str, passes: int = 3) -> str:
    cleaned = name
    for _ in range(max(1, passes)):
        cleaned = PREFIX_RE.sub("", cleaned).strip()
    return cleaned


# ---------------------------------------------------------------------------
# 3) Manual overrides (generated via GPT in final stage)
# ---------------------------------------------------------------------------
GENDER_OVERRIDES: Dict[str, str] = {
    "Hon. Hedworth Jolliffe":                   "male",
    "Captain Bertie Kirby":                     "male",
    "Dr Ashok Kumar":                           "male",
    "Hon. Egremont Lascelles":                  "male",
    "Hon. Beilby Lawley":                       "male",  # applies to both 1818–1880 and 1849–1912 entries
    "Hon. Heneage Legge":                       "male",
    "Major Bertie Leighton":                    "male",
    "Colonel Claude Lowther":                   "male",
    "Dr Dickson Mabon":                         "male",
    "Lieut-Colonel Hon. Glyn Mason":            "male",
    "Hon. Fox Maule":                           "male",
    "Hon. Lauderdale Maule":                    "male",
    "Hon. Somerset Maxwell":                    "male",
    "Dr Bouverie McDonald":                     "male",
    "Captain Algernon Moreing":                 "male",
    "Dr Hyacinth Morgan":                       "male",
    "Rear-Admiral Morgan Morgan-Giles":         "male",
    "Dr Mo Mowlam":                             "female",
    "The O'Conor Don":                          "male",  # (1794–1847 and 1838–1906 share the same name key)
    "Master of Elibank":                        "male",
    "Hon. Eustace Fiennes":                     "male",
    "Dr Mont Follick":                          "male",
    "Hon. Vicary Gibbs":                        "male",
    "Hon. Pascoe Glyn":                         "male",
    "Hon. Sidney Glyn":                         "male",
    "Hon. Algernon Greville":                   "male",
    "Captain Rees Gronow":                      "male",
    "Dr Leslie Haden-Guest":                    "male",
    "Major Collingwood Hamilton":               "male",
    "Captain Villiers Hatton":                  "male",
    "Hon. Claude Hay":                          "male",
    "Hon. Auberon Herbert":                     "male",
    "Hon. Aubrey Herbert":                      "male",
    "Hon. Sidney Herbert":                      "male",  # covers both 1810–1861 and 1853–1913 entries
    "Mary Holt":                                "female",
    "Hon. Grenville Howard":                    "male",
    "Dr Kim Howells":                           "male",
    "Lieut-General Sir Aylmer Hunter-Weston":   "male",
    "The O'Donoghue of the Glens":              "male",
    "The O'Gorman Mahon":                       "male",
    "Lieut-Colonel Hon. Standish O'Grady":      "male",
    "Hon. Weetman Pearson":                     "male",
    "Lieut-Colonel Hon. Sidney Peel":           "male",
    "Dr Sidney Peters":                         "male",
    "Hon. Mary Pickford":                       "female",
    "Hon. Duncombe Pleydell-Bouverie":          "male",
    "Hon. Ashley Ponsonby":                     "male",
    "Lieut-Colonel Sir Assheton Pownall":       "male",
    "Commander Redvers Prior":                  "male",
    "Lieut-Colonel Wentworth Schofield":        "male",
    "Hon. Sam Silkin":                          "male",
    "Very Reverend Dr John Simms":              "male",
    "Captain Sidney Streatfeild":               "male",
    "Hon. Algernon Tollemache":                 "male",
    "Hon. Wilbraham Tollemache":                "male",
    "Major Vaughan Vaughan-Lee":                "male",
    "Hon. Greville Vernon":                     "male",
    "Flight Lieut Wavell Wakefield":            "male",
    "Lieut-Colonel Penry Williams":             "male",
    "Lieut-Colonel Leslie Wilson":              "male",
    "Hon. Waldorf Astor":               "male",
    "Rear-Admiral Tufton Beamish":      "male",
    "Major Aubrey Beauclerk":           "male",
    "Hon. Wentworth Beaumont":          "male",
    "Hon. Sir Gervase Beckett":         "male",
    "Hon. Craven Berkeley":             "male",
    "Hon. Cadwallader Blayney":         "male",
    "Dr Rhodes Boyson":                 "male",
    "Lieut-Colonel Campbell-Johnston":  "male",
    "Hon. Swynfen Carnegie":            "male",
    "Colonel Sir Smith Child":          "male",
    "Hon. Wenman Coke":                 "male",
    "Lieut-Colonel Uvedale Corbett":    "male",
    "Hon. Wellington Cotton":           "male",
    "Colonel Chichester Crookshank":    "male",
    "Hon. Montagu Curzon":              "male",
    "Hon. Vesey Dawson":                "male",
    "Hon. Octavius Duncombe":           "male",
    "Hon. Algernon Egerton":            "male",
    "Hon. Wilbraham Egerton":           "male"
}


# ---------------------------------------------------------------------------
# 4) Main logic ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
    log = logging.getLogger("gender-infer")

    speakers: List[dict]
    speakers = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    # Stage 0: init field
    for sp in speakers:
        sp["gender"] = None

    # Stage 1: direct title lookup
    for sp in speakers:
        sp["gender"] = _title_lookup(sp.get("name", ""))

    none_after_title = [sp for sp in speakers if sp["gender"] is None]
    log.info("After title lookup → unresolved: %d", len(none_after_title))

    # Stage 2: strip prefixes + title lookup again
    for sp in none_after_title:
        cleaned = _strip_prefixes(sp.get("name", ""))
        sp["gender"] = _title_lookup(cleaned)

    none_after_strip = [sp for sp in speakers if sp["gender"] is None]
    log.info("After prefix stripping → unresolved: %d", len(none_after_strip))

    # Stage 3: gender_guesser strict
    detector = Detector(case_sensitive=False)
    for sp in none_after_strip:
        cleaned = _strip_prefixes(sp.get("name", ""))
        if cleaned:
            first = cleaned.split()[0].rstrip(",").capitalize()
            guess = detector.get_gender(first)
            if guess == "male":
                sp["gender"] = "male"
            elif guess == "female":
                sp["gender"] = "female"

    none_after_guesser = [sp for sp in speakers if sp["gender"] is None]
    log.info("After gender_guesser → unresolved: %d", len(none_after_guesser))

    # Stage 4: GPT‑derived overrides
    for sp in none_after_guesser:
        override = GENDER_OVERRIDES.get(sp.get("name", "").strip())
        if override:
            sp["gender"] = override

    final_none = [sp for sp in speakers if sp["gender"] is None]
    log.info("After overrides → still unresolved: %d", len(final_none))

    OUTPUT_PATH.write_text(json.dumps(speakers, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote enriched data → %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
