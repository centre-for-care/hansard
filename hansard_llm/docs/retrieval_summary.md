# Semantic retrieval experiment — summary

Definition-as-query ranking with `Qwen3-Embedding-8B`. Proxy gold on the
pilot: speech-level majority of `mentions_topic` under expert HC→SC /
JSON / uncapped / role=none.

## Keyword baseline (pilot)

```
{
  "n": 270,
  "n_pos": 129,
  "precision": 0.6778,
  "recall": 0.9457,
  "auroc": 0.7672,
  "average_precision": 0.6669
}
```

## Ranking metrics by query × mode (pilot)

| Query | Mode | AUROC | AP | P@25 | R@25 | mean+ | mean− |
|---|---|---:|---:|---:|---:|---:|---:|
| expert_hc_only | whole | 0.872 | 0.878 | 1.000 | 0.194 | 0.304 | 0.220 |
| name_only | maxchunk | 0.874 | 0.874 | 0.960 | 0.186 | 0.417 | 0.333 |
| current | maxchunk | 0.858 | 0.855 | 0.960 | 0.186 | 0.455 | 0.380 |
| expert_hc_sc | whole | 0.863 | 0.852 | 0.960 | 0.186 | 0.335 | 0.255 |
| expert_hc_only | maxchunk | 0.843 | 0.842 | 1.000 | 0.194 | 0.352 | 0.266 |
| expert_sc_hc | whole | 0.848 | 0.831 | 0.960 | 0.186 | 0.345 | 0.270 |
| name_only | whole | 0.851 | 0.821 | 0.840 | 0.163 | 0.391 | 0.312 |
| current | whole | 0.845 | 0.817 | 0.880 | 0.171 | 0.440 | 0.368 |
| expert_hc_sc | maxchunk | 0.824 | 0.801 | 0.920 | 0.178 | 0.369 | 0.288 |
| era_neutral | whole | 0.812 | 0.792 | 0.920 | 0.178 | 0.384 | 0.312 |
| expert_sc_hc | maxchunk | 0.808 | 0.784 | 0.920 | 0.178 | 0.378 | 0.301 |
| expert_sc_only | whole | 0.765 | 0.735 | 0.760 | 0.147 | 0.310 | 0.255 |
| era_neutral | maxchunk | 0.763 | 0.734 | 0.840 | 0.163 | 0.418 | 0.350 |
| expert_sc_only | maxchunk | 0.731 | 0.712 | 0.760 | 0.147 | 0.346 | 0.287 |

## Threshold sweep — `expert_hc_sc` (pilot)

| Mode | Threshold | Retained | Recall | Precision |
|---|---:|---:|---:|---:|
| maxchunk | 0.20 | 0.956 | 1.000 | 0.500 |
| maxchunk | 0.25 | 0.852 | 0.977 | 0.548 |
| maxchunk | 0.30 | 0.644 | 0.876 | 0.649 |
| maxchunk | 0.35 | 0.359 | 0.605 | 0.804 |
| maxchunk | 0.40 | 0.148 | 0.264 | 0.850 |
| maxchunk | 0.45 | 0.067 | 0.124 | 0.889 |
| maxchunk | 0.50 | 0.011 | 0.023 | 1.000 |
| maxchunk | 0.55 | 0.007 | 0.015 | 1.000 |
| maxchunk | 0.60 | 0.004 | 0.008 | 1.000 |
| maxchunk | 0.65 | 0.000 | 0.000 | nan |
| maxchunk | 0.70 | 0.000 | 0.000 | nan |
| maxchunk | 0.75 | 0.000 | 0.000 | nan |
| maxchunk | 0.80 | 0.000 | 0.000 | nan |
| whole | 0.20 | 0.926 | 1.000 | 0.516 |
| whole | 0.25 | 0.767 | 0.961 | 0.599 |
| whole | 0.30 | 0.422 | 0.729 | 0.825 |
| whole | 0.35 | 0.181 | 0.341 | 0.898 |
| whole | 0.40 | 0.059 | 0.124 | 1.000 |
| whole | 0.45 | 0.022 | 0.046 | 1.000 |
| whole | 0.50 | 0.004 | 0.008 | 1.000 |
| whole | 0.55 | 0.000 | 0.000 | nan |
| whole | 0.60 | 0.000 | 0.000 | nan |
| whole | 0.65 | 0.000 | 0.000 | nan |
| whole | 0.70 | 0.000 | 0.000 | nan |
| whole | 0.75 | 0.000 | 0.000 | nan |
| whole | 0.80 | 0.000 | 0.000 | nan |

## Filter pool — score by keyword seed

| Query | Mode | Seed | N | Mean | P50 | P90 |
|---|---|---|---:|---:|---:|---:|
| current | maxchunk | False | 2819 | 0.374 | 0.371 | 0.437 |
| current | maxchunk | True | 181 | 0.449 | 0.452 | 0.525 |
| current | whole | False | 2819 | 0.357 | 0.353 | 0.416 |
| current | whole | True | 181 | 0.428 | 0.427 | 0.493 |
| expert_hc_sc | maxchunk | False | 2819 | 0.277 | 0.272 | 0.353 |
| expert_hc_sc | maxchunk | True | 181 | 0.378 | 0.372 | 0.475 |
| expert_hc_sc | whole | False | 2819 | 0.247 | 0.241 | 0.306 |
| expert_hc_sc | whole | True | 181 | 0.330 | 0.327 | 0.415 |
| name_only | maxchunk | False | 2819 | 0.323 | 0.319 | 0.388 |
| name_only | maxchunk | True | 181 | 0.412 | 0.407 | 0.493 |
| name_only | whole | False | 2819 | 0.298 | 0.292 | 0.356 |
| name_only | whole | True | 181 | 0.378 | 0.377 | 0.445 |

## LLM spot-check on filter pool (expert HC→SC / JSON)

| Band | N labeled | Mean retrieval score | LLM positive rate |
|---|---:|---:|---:|
| bottom | 50 | 0.154 | 0.0 |
| random_mid | 50 | 0.248 | 0.12 |
| top | 50 | 0.433 | 0.94 |

## Provisional takeaway

Best ranking on the labeled pilot: **expert_hc_only** / **whole** (AP=0.878, AUROC=0.872). Keyword baseline AP=0.6669 (precision=0.6778, recall=0.9457).
For the shipping query **expert_hc_sc / whole**, a cosine threshold of **0.25** reaches ≥90% recall of LLM-positives on the pilot while retaining **76.7%** of speeches (precision ≈ 59.9%). Pilot is keyword-stratified, so retained fractions will be lower on a natural corpus.
On the era-stratified filter pool spot-check (150 speeches, expert HC→SC / JSON), LLM-positive rates were **0.94** (top-50 by score), **0.12** (random mid), **0.0** (bottom-50) — strong enrichment at the top of the ranking outside the keyword-stratified pilot.
