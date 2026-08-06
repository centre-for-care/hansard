"""Configuration: environment, model registry, topic definition, paths.

Everything that varies between runs (which models, which topic, where files
live) is centralised here so the rest of the package stays declarative and the
experiment is reproducible from a single source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# The code lives in the scraper repo, but the large inputs (the ~3 GB enriched
# Parquet) and pilot outputs are kept OUTSIDE the repo. Locations are resolved
# from env vars so the package is portable; the fallbacks point at the sibling
# ``hansard_eda`` checkout used during the pilot (repo and hansard_eda side by
# side under the same parent). Set the env vars in ``.env`` to be explicit.
REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_HOME = REPO_ROOT.parent / "hansard_eda"


def _resolved(env_var: str, default: Path) -> Path:
    val = os.environ.get(env_var)
    return Path(val).expanduser() if val else default


ENV_PATH = _resolved("HANSARD_LLM_ENV", _DATA_HOME / ".env")

# Load .env first so the remaining path vars below can be overridden from it.
# python-dotenv's default search walks up from the *caller's* directory, which
# fails when scripts live elsewhere, so we point at the file explicitly.
load_dotenv(ENV_PATH)

DATA_DIR = _resolved("HANSARD_LLM_DATA_DIR", _DATA_HOME / "data")

# All pilot artifacts live under artifacts/llm/ (gitignored alongside data).
ARTIFACTS_DIR = _resolved("HANSARD_LLM_ARTIFACTS_DIR", _DATA_HOME / "artifacts" / "llm")
SAMPLE_PATH = ARTIFACTS_DIR / "pilot_sample.parquet"
RESULTS_PATH = ARTIFACTS_DIR / "pilot_results.parquet"

# Versioned results store (see provenance.py). New experiment runs write under
# runs/<experiment>/<run_id>/; the pre-provenance single log is frozen here.
RUNS_DIR = ARTIFACTS_DIR / "runs"
LEGACY_DIR = ARTIFACTS_DIR / "legacy"
LEGACY_RESULTS_LOG = LEGACY_DIR / "pilot_results.frozen-20260802.jsonl"
LEGACY_ANNOTATED_PATH = LEGACY_DIR / "pilot_results_annotated.parquet"


def _require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(
            f"Missing env var {key!r}. Copy env.txt to .env and fill it in "
            f"(expected at {ENV_PATH})."
        )
    return val


def base_url() -> str:
    return _require("LLM_BASE_URL")


def api_key() -> str:
    return _require("LLM_API_KEY")


def base_url_host() -> str:
    """Hostname of the serving endpoint (for provenance rows/manifests)."""
    from urllib.parse import urlparse
    try:
        return urlparse(base_url()).hostname or "unknown"
    except RuntimeError:
        return "unset"


def backend_name() -> str:
    """Label for which serving stack produced a row: ``LLM_BACKEND_NAME`` if
    set (e.g. ``vllm-cluster``), else derived from the endpoint host. The same
    model served by Nebius and by a local vLLM is not guaranteed to produce
    identical output, so rows must record which one they came from."""
    return os.environ.get("LLM_BACKEND_NAME") or base_url_host()


# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """A hosted model and how to call it.

    ``reasoning`` models emit a hidden reasoning trace before the visible
    answer (Nebius returns it in a non-standard ``reasoning`` field), so they
    need a larger token budget and are kept out of the core grid to avoid
    confounding "reasoning vs not" with "model family". See the smoke-test
    notes in the project history.
    """

    model_id: str
    family: str
    reasoning: bool = False
    max_tokens: int = 512
    # "production" = a real deploy candidate (must be feasible on BMRC);
    # "reference" = included in the grid only as a quality/size ceiling, to test
    # whether a much larger model would change the answer. Not for deployment.
    tier: str = "production"


# Active non-reasoning grid for new runs — aligned with the cluster panel.
# Servable on a single A100-80GB (Nemotron-49B needs FP8). Qwen3-235B is
# deferred in REFERENCE_MODELS and must not be pulled into default plans or
# download scripts until we deliberately schedule it.
CORE_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("Qwen/Qwen3-30B-A3B-Instruct-2507", family="qwen"),   # MoE, 3B active
    ModelSpec("google/gemma-3-27b-it", family="google"),            # 27B dense
    ModelSpec("nvidia/Llama-3_3-Nemotron-Super-49B-v1_5", family="nvidia"),  # was Llama-3.3-70B
    ModelSpec("Qwen/Qwen3-32B", family="qwen"),                    # dense sibling of 30B-A3B
)

# Production-feasible subset (used when picking what to actually deploy).
PRODUCTION_MODELS: tuple[ModelSpec, ...] = tuple(
    m for m in CORE_MODELS if m.tier == "production")

# Size-ceiling reference — not in CORE/PANEL, not downloaded by default.
REFERENCE_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("Qwen/Qwen3-235B-A22B-Instruct-2507", family="qwen",
              tier="reference"),
)

# Reasoning-class models, available but deferred to a separate axis. They need
# a much larger budget because the reasoning trace consumes tokens first.
REASONING_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("moonshotai/Kimi-K2.6", family="moonshot", reasoning=True, max_tokens=2048),
    ModelSpec("openai/gpt-oss-120b", family="gpt-oss", reasoning=True, max_tokens=2048),
)

# --------------------------------------------------------------------------
# Panel models (cluster vLLM; Workstream C2 — the unified model-sensitivity
# experiment whose labels double as retrieval gold via leave-one-out)
# --------------------------------------------------------------------------
# Same set as CORE_MODELS: Nemotron-Super-49B replaces Llama-3.3-70B
# (single 80GB GPU at FP8); Qwen3-32B pairs with 30B-A3B for dense-vs-MoE.
PANEL_MODELS: tuple[ModelSpec, ...] = CORE_MODELS

# Extended size/family axis: runs on a ~2k subsample only.
EXTENDED_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("Qwen/Qwen3-4B-Instruct-2507", family="qwen"),
    ModelSpec("Qwen/Qwen3-14B", family="qwen"),
    ModelSpec("mistralai/Mistral-Small-3.2-24B-Instruct-2506", family="mistral"),
)

# Panel labels under the definitions we actually care about. Retrieval eval
# must use leave-one-definition-out (panel.panel_gold(exclude_definition=…))
# so a query is never scored against gold produced from the same wording.
PANEL_DEFINITIONS: tuple[str, ...] = (
    "expert_hc_sc", "expert_sc_hc", "current", "name_only",
)

MODELS_BY_ID = {m.model_id: m for m in
                CORE_MODELS + EXTENDED_MODELS + REFERENCE_MODELS + REASONING_MODELS}


# --------------------------------------------------------------------------
# Topic definition
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Topic:
    """A fixed target topic for extraction.

    Only the *topic* is fixed; sub-themes are discovered inductively (free-text
    generation), because the research aim is exploratory — "what is there" —
    rather than scoring against a pre-set schema. Comparability of those
    free-text themes across prompt variants is recovered downstream via
    embedding similarity, not by constraining the model's vocabulary here.

    ``seed_regex`` is the existing heuristic used only to *stratify the sample*
    (so it is not ~97% negatives) — it is never used as a label.
    ``max_subthemes`` caps emitted phrases per speech to keep semantic
    comparison tractable and cost bounded.
    """

    name: str
    description: str
    seed_regex: str
    max_subthemes: int = 5
    # Label for which *construct definition* this Topic encodes. Only
    # ``description`` differs between definitions, so this is the provenance tag
    # that makes rows from different definitions separable in the results log.
    # It is deliberately NOT part of the rendered prompt, so adding it leaves
    # every existing ``prompt_hash`` (and therefore the whole cache) untouched.
    definition_id: str = "current"


# Seed vocabulary for *sample stratification only* (never a label). Recall is
# prioritised over precision: the LLM makes the real H&SC judgement, so a broad
# seed just ensures the sample contains plausible positives across every era and
# event type. Assembled from the validated patterns in the prior exploratory
# stage (scripts/hsc_vocab_comparison.py and scripts/covid_analysis.py): modern
# NHS-era terms, pre-NHS historical vocabulary (Poor Law, sanitary reform,
# epidemic disease), and the COVID/pandemic spike.

# Modern — NHS-era and welfare-state terms (post-WWII).
_HSC_MODERN = (
    r"social care|care home|domiciliary care|residential care|adult social care"
    r"|care worker|carer|\bnhs\b|national health service|public health"
    r"|mental health|health care|healthcare"
)

# Historical — pre-NHS/pre-welfare-state: Poor Law system + sanitary reform +
# 19th-century epidemic disease.
_HSC_HISTORICAL = (
    r"poor law|workhouse|\bpauper|board of guardians|relieving officer"
    r"|district nurs|lunatic asylum|feeble.minded|mental deficiency"
    r"|friendly society|\binfirmary\b|almshouse"
    r"|sanitar|board of health|fever hospital|\bvaccination\b|\bsmallpox\b"
    r"|\bcholera\b|\btuberculosis\b|medical officer of health|\bdispensary\b"
)

# COVID/pandemic spike plus broader epidemic/event vocabulary (the "epidemics
# and other events" the modern set misses across the timeline).
_HSC_EPIDEMIC = (
    r"\bcovid\b|\bcoronavirus\b|\bpandemic\b|long covid|\blockdown\b|\bfurlough\b"
    r"|\bepidemic\b|\binfluenza\b|spanish flu|\bquarantine\b|\boutbreak\b"
    r"|\btyphoid\b|\btyphus\b|\bdiphtheria\b|scarlet fever|\bplague\b"
)

_HSC_SEED_REGEX = "|".join((_HSC_MODERN, _HSC_HISTORICAL, _HSC_EPIDEMIC))

HEALTH_SOCIAL_CARE = Topic(
    name="health and social care",
    description=(
        "UK health and social care policy: the NHS, public and mental health, "
        "adult social care, care homes, and the people who provide or rely on "
        "that care"
    ),
    seed_regex=_HSC_SEED_REGEX,
    definition_id="current",
)

# --------------------------------------------------------------------------
# Alternative construct definitions (the definition-sensitivity arm)
# --------------------------------------------------------------------------
# Unlike role/task/format, the definition is NOT a nuisance factor: we expect it
# to move the result. The question is by how much, and in particular whether the
# era gradient in presence (32% pre-1900 vs 63% post-1948) survives a definition
# that does not name post-1948 institutions.
#
# ``current`` names the NHS, adult social care and care homes: three anchors
# that did not exist for the first ~145 years of the corpus. ``era_neutral`` is
# the candidate replacement, describing the *function* (caring for the sick and
# destitute) rather than the institution, so a workhouse infirmary and an NHS
# trust are in scope on the same terms. ``narrow_clinical`` and
# ``broad_determinants`` bracket the construct from either side so the headline
# prevalence can be reported as a sensitivity band rather than a point estimate.
#
# The expert arm replaces the pilot wordings with domain-expert definitions of
# healthcare and social care. Because the construct of interest is speeches
# about *both*, the two texts are concatenated; ``expert_hc_sc`` vs
# ``expert_sc_hc`` isolates whether presentation order moves the result.
#
# Only ``description`` varies. Name, seed regex and cap are held constant, so
# the contrast is the definition and nothing else.
_DEFINITION_TEXT = {
    "era_neutral": (
        "the health of the population and the care of people who are sick, "
        "injured, disabled, elderly or destitute, however that care is "
        "provided and paid for, whether by the state, local authorities, "
        "hospitals, charities, religious bodies or families"
    ),
    "narrow_clinical": (
        "the treatment of illness and injury and the organisation, funding and "
        "staffing of medical services such as hospitals, doctors, nurses and "
        "medicines, but not social care, welfare relief, or general living "
        "conditions"
    ),
    "broad_determinants": (
        "anything bearing on the health and wellbeing of the population, "
        "including medical treatment and care services and also the conditions "
        "that shape health such as housing, sanitation, clean water, poverty, "
        "nutrition and working conditions"
    ),
}

# Expert-sourced component texts (domain-provided). Combined below in both
# orders so order effects can be measured without inventing new wording.
_EXPERT_HEALTHCARE = (
    "Healthcare is primarily concerned with the prevention, diagnosis and "
    "treatment of illness."
)
_EXPERT_SOCIAL_CARE = (
    "Social care provides practical support to people who need assistance "
    "with everyday living because of age, disability, chronic illness or "
    "other long-term needs. Typical services include residential care, care "
    "homes, home care, support for carers and services for adults with "
    "disabilities."
)

_EXPERT_DEFINITION_TEXT = {
    # Healthcare first, then social care.
    "expert_hc_sc": f"{_EXPERT_HEALTHCARE} {_EXPERT_SOCIAL_CARE}",
    # Social care first, then healthcare.
    "expert_sc_hc": f"{_EXPERT_SOCIAL_CARE} {_EXPERT_HEALTHCARE}",
}

HSC_DEFINITIONS: dict[str, Topic] = {
    "current": HEALTH_SOCIAL_CARE,
    # Name-only baseline: topic name with no expanded construct wording.
    "name_only": replace(
        HEALTH_SOCIAL_CARE, description="", definition_id="name_only",
    ),
    **{
        key: replace(HEALTH_SOCIAL_CARE, description=text, definition_id=key)
        for key, text in {**_DEFINITION_TEXT, **_EXPERT_DEFINITION_TEXT}.items()
    },
}

# Shipping default: expert healthcare-then-social-care wording.
DEFAULT_TOPIC = HSC_DEFINITIONS["expert_hc_sc"]

# New uncached arms for the next definition run. Prior alts
# (era_neutral / narrow_clinical / broad_determinants) stay in
# HSC_DEFINITIONS so cached rows remain joinable; pass them explicitly via
# ``--definitions`` if you need to re-run or extend them.
ALT_DEFINITIONS: tuple[str, ...] = (
    "expert_hc_sc", "expert_sc_hc", "current", "name_only",
)
