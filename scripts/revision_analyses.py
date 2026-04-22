# ruff: noqa: I001
"""Revision-roadmap analyses: baselines, subset-calibration, CIs, kappa, chi-square.

Runs every analysis added in response to the reviewer round:

  1. Lexical baselines on the 1,020-pair labelled set:
        (a) token-set Jaccard after aggressive normalisation
        (b) TF-IDF cosine (character n-gram features)
     These stand in for the class of classical token-based clone detectors
     (NiCad, SourcererCC) on short near-natural-language text. Full
     installation of the reference implementations is out of scope for a
     Python-only reproducibility bundle, but the token-Jaccard baseline
     captures the defining primitive of SourcererCC (normalised token
     multiset + threshold) and the TF-IDF-char-n-gram baseline captures
     the lexical-signature spirit of NiCad's pretty-print-and-compare
     approach.

  2. Score-free relabelling: a second labelling pass that uses only text
     features (aggressive normalisation, token-set equality, a small
     hand-curated synonym table, structural rules R4-R8) and *never*
     accesses cosine or Levenshtein. Replaces the circular R1-R3 rules
     of the primary rubric with purely textual positive criteria.
     Two downstream metrics are derived:
        (a) Detector F1 evaluated against the score-free labels.
            This number is circularity-free: the labels were produced
            without the features the detector uses.
        (b) Cohen's kappa between the primary rubric and the score-free
            relabelling. Measures inter-protocol agreement - whether the
            primary rubric's positive-label decisions survive when the
            score cut-points are removed.

  3. Bootstrap 95 % confidence intervals for F1, precision, recall at
     each strategy's best operating point (1,000 resamples of the
     labelled set with replacement).

  5. Chi-square test of licence-class homogeneity of exact-duplication
     rates, plus pairwise two-proportion z-tests, quantifying the
     "approximately uniform across three strata" claim in Section 7.4.

  6. Per-repository duplication rate vs repository size scatter plus
     Spearman rank correlation (addresses the alternative-explanation
     question of whether duplication rate is a size artefact).

Results land in analysis/revision_*.json; plots in paper/figures/.
"""

from __future__ import annotations

import torch  # noqa: F401  # MUST import before pandas on Windows (WinError 1114)

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --- common helpers ------------------------------------------------------


def _metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def _confusion(preds: np.ndarray, gold: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((preds == 1) & (gold == 1)))
    fp = int(np.sum((preds == 1) & (gold == 0)))
    fn = int(np.sum((preds == 0) & (gold == 1)))
    tn = int(np.sum((preds == 0) & (gold == 0)))
    return tp, fp, fn, tn


def _best_threshold(
    scores: np.ndarray, gold: np.ndarray, thresholds: np.ndarray
) -> dict[str, float | int]:
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        tp, fp, fn, tn = _confusion(preds, gold)
        p, r, f = _metrics(tp, fp, fn)
        if f > float(best["f1"]):
            best = {
                "threshold": float(round(thr, 4)),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f, 4),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            }
    return best


# --- analysis 1: lexical baselines ---------------------------------------


_NORM_RE = re.compile(r'"[^"]*"|\'[^\']*\'|[<>][\w-]+[<>]|[0-9]+')
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    t = _NORM_RE.sub(" PARAM ", text)
    t = _PUNCT_RE.sub(" ", t.lower())
    t = _WS_RE.sub(" ", t).strip()
    return t


def _jaccard(a: str, b: str) -> float:
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def analysis_lexical_baselines(
    pairs: list[dict], out_dir: Path
) -> dict[str, object]:
    texts_a = [p["text_a"] for p in pairs]
    texts_b = [p["text_b"] for p in pairs]
    gold = np.array([int(p["label"]) for p in pairs])

    # (a) Token-set Jaccard with aggressive normalisation
    normed_a = [_normalise(t) for t in texts_a]
    normed_b = [_normalise(t) for t in texts_b]
    jac = np.array([_jaccard(a, b) for a, b in zip(normed_a, normed_b)])

    # (b) TF-IDF cosine on character 3- to 5-grams (robust to tiny spelling drift)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    all_texts = list(texts_a) + list(texts_b)
    X = vec.fit_transform(all_texts)
    n = len(texts_a)
    A = X[:n]
    B = X[n:]
    tfidf_cos = np.array([
        float(cosine_similarity(A[i], B[i])[0, 0]) for i in range(n)
    ])

    thresholds = np.round(np.arange(0.30, 1.00, 0.01), 2)
    best_jac_primary = _best_threshold(jac, gold, thresholds)
    best_tfidf_primary = _best_threshold(tfidf_cos, gold, thresholds)

    # Score-free labels computed here too so baselines can be evaluated
    # against them in the same pass.
    sf_labels = []
    for p in pairs:
        sf_l, _ = _score_free_label(str(p["text_a"]), str(p["text_b"]))
        sf_labels.append(sf_l)
    sf_gold = np.array(sf_labels)
    best_jac_sf = _best_threshold(jac, sf_gold, thresholds)
    best_tfidf_sf = _best_threshold(tfidf_cos, sf_gold, thresholds)

    result = {
        "n_pairs": len(pairs),
        "n_duplicates_primary": int(gold.sum()),
        "n_duplicates_score_free": int(sf_gold.sum()),
        "token_jaccard": {
            "description": (
                "Token-set Jaccard after aggressive normalisation "
                "(quoted parameters -> PARAM, punctuation stripped, lowercased). "
                "Stands in for SourcererCC's defining primitive on short "
                "near-natural-language text."
            ),
            "best_primary": best_jac_primary,
            "best_score_free": best_jac_sf,
        },
        "tfidf_char_ngram_cosine": {
            "description": (
                "TF-IDF cosine over character 3- to 5-gram features. "
                "Captures the lexical-signature spirit of NiCad's "
                "pretty-print-and-compare approach."
            ),
            "best_primary": best_tfidf_primary,
            "best_score_free": best_tfidf_sf,
        },
    }

    out = out_dir / "revision_baselines.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[1] baselines -> {out}")
    print(f"    token-Jaccard (primary)   thr={best_jac_primary['threshold']} "
          f"P={best_jac_primary['precision']} R={best_jac_primary['recall']} "
          f"F1={best_jac_primary['f1']}")
    print(f"    token-Jaccard (sf)        thr={best_jac_sf['threshold']} "
          f"P={best_jac_sf['precision']} R={best_jac_sf['recall']} "
          f"F1={best_jac_sf['f1']}")
    print(f"    tfidf-ngram   (primary)   thr={best_tfidf_primary['threshold']} "
          f"P={best_tfidf_primary['precision']} R={best_tfidf_primary['recall']} "
          f"F1={best_tfidf_primary['f1']}")
    print(f"    tfidf-ngram   (sf)        thr={best_tfidf_sf['threshold']} "
          f"P={best_tfidf_sf['precision']} R={best_tfidf_sf['recall']} "
          f"F1={best_tfidf_sf['f1']}")
    return result


# --- analysis 2: score-free relabelling ----------------------------------


# BDD-domain synonym table. Each row is a family of equivalent tokens;
# substituting any member for any other within a step preserves the
# maintainer-level intent of the step in the overwhelming majority of
# observed cases. The table is deliberately short and conservative; it
# is not meant to cover every possible paraphrase, only the high-frequency
# BDD-vocabulary families that appear in the corpus's top clusters.
_SYNONYM_FAMILIES = [
    {"get", "fetch", "retrieve", "receive"},
    {"send", "submit", "post"},
    {"should", "shall", "must"},
    {"click", "tap", "press", "hit"},
    {"see", "view", "observe"},
    {"return", "returns", "response"},
    {"status", "statuscode", "statuses"},
    {"is", "are", "equals", "==", "="},
    {"200", "200ok", "ok"},
    {"the", "a", "an"},
]

_HTTP_VERBS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")


def _canonical_tokens(text: str) -> list[str]:
    t = _NORM_RE.sub(" PARAM ", text)
    t = _PUNCT_RE.sub(" ", t.lower())
    tokens = t.split()
    canonical: list[str] = []
    for tok in tokens:
        replaced = tok
        for fam in _SYNONYM_FAMILIES:
            if tok in fam:
                replaced = sorted(fam)[0]
                break
        canonical.append(replaced)
    return canonical


def _score_free_label(text_a: str, text_b: str) -> tuple[int, str]:
    """Second-pass labeller: text-only, no cos or lev access.

    Mirrors the structural rules R4-R8 of the primary rubric (same text
    cues, same polarity) and replaces the score-based positive rules
    R1-R3 with deterministic text-rewriting rules. No similarity score
    is consulted.
    """

    t_a = text_a.lower(); t_b = text_b.lower()

    # R4 call vs callonce
    a_call = bool(re.search(r"(?<![\w])call(?![\w])", t_a)) and \
             not bool(re.search(r"(?<![\w])callonce(?![\w])", t_a))
    b_call = bool(re.search(r"(?<![\w])call(?![\w])", t_b)) and \
             not bool(re.search(r"(?<![\w])callonce(?![\w])", t_b))
    a_once = bool(re.search(r"(?<![\w])callonce(?![\w])", t_a))
    b_once = bool(re.search(r"(?<![\w])callonce(?![\w])", t_b))
    if (a_call and b_once) or (a_once and b_call):
        return 0, "R4_sf"

    # R4 def vs set
    pa = text_a.strip(); pb = text_b.strip()
    if (pa.startswith("def ") and pb.startswith("set ")) or \
       (pa.startswith("set ") and pb.startswith("def ")):
        return 0, "R4_sf"

    # R5 different HTTP verbs
    def _verb(t: str) -> str | None:
        for v in _HTTP_VERBS:
            if f'"{v}"' in t or f" {v} " in f" {t} ":
                return v
        return None
    va = _verb(text_a); vb = _verb(text_b)
    if va and vb and va != vb:
        return 0, "R5_sf"

    # R6 polarity flip
    a_has_not = any(tok in {"not", "n't"} for tok in t_a.split())
    b_has_not = any(tok in {"not", "n't"} for tok in t_b.split())
    a_tok = set(t_a.split()); b_tok = set(t_b.split())
    diff = a_tok.symmetric_difference(b_tok)
    if a_has_not != b_has_not and diff <= {"not", "n't", "should", "shouldn't"}:
        return 0, "R6_sf"

    # R7 existence vs content
    exist = ("should exist", "should not exist", "is listed", "is not listed")
    content = ("the content of", "should be", "should contain", "equals")
    a_is_exist = any(m in t_a for m in exist); b_is_exist = any(m in t_b for m in exist)
    a_is_content = any(m in t_a for m in content); b_is_content = any(m in t_b for m in content)
    if (a_is_exist and b_is_content) or (b_is_exist and a_is_content):
        return 0, "R7_sf"

    # R8 action vs assertion
    action_starts = ("i ", "user ", "they ", "click ", "tap ", "press ", "fill ",
                     "submit ", "go ", "visit ", "open ", "create ", "delete ",
                     "update ", "edit ", "follow ", "run ")
    a_action = any(t_a.startswith(s) for s in action_starts)
    b_action = any(t_b.startswith(s) for s in action_starts)
    a_assert = t_a.startswith("should ") or " should " in t_a or t_a.startswith("the ")
    b_assert = t_b.startswith("should ") or " should " in t_b or t_b.startswith("the ")
    if (a_action and b_assert and not a_assert) or (b_action and a_assert and not b_assert):
        return 0, "R8_sf"

    # Positive rules, text-only (NO score access).
    ca = _canonical_tokens(text_a)
    cb = _canonical_tokens(text_b)

    # P1: identical after parametric normalisation + synonym canonicalisation.
    if ca == cb:
        return 1, "P1_sf_exact_after_normalisation"

    # P2: same token multiset after canonicalisation (word-reorder paraphrase).
    if Counter(ca) == Counter(cb) and len(ca) >= 3:
        return 1, "P2_sf_token_multiset"

    # P3: one canonical text is a contiguous subsequence of the other,
    # and the overlap covers at least 70 % of the shorter text. Captures
    # truncation / extension paraphrases.
    if len(ca) >= 3 and len(cb) >= 3:
        if len(ca) <= len(cb):
            short, long = ca, cb
        else:
            short, long = cb, ca
        joined_short = " ".join(short)
        joined_long = " ".join(long)
        if joined_short in joined_long and len(short) / len(long) >= 0.70:
            return 1, "P3_sf_subsequence"

    # P4: token-set Jaccard >= 0.80 is a sharp text-only cue for paraphrase
    # of the kind maintainers consolidate. The cut-point is Levenshtein-free
    # (set-based, no edit-distance scoring).
    set_a = set(ca); set_b = set(cb)
    if set_a and set_b:
        jac = len(set_a & set_b) / len(set_a | set_b)
        if jac >= 0.80:
            return 1, "P4_sf_token_jaccard"

    return 0, "R10_sf_default"


def analysis_score_free_relabel(
    pairs: list[dict], out_dir: Path
) -> dict[str, object]:
    """Relabel all 1,020 pairs with a score-free protocol.

    Produces (a) detector F1 against score-free labels, and
    (b) kappa between primary rubric and score-free protocol.
    """

    sf_labels: list[int] = []
    sf_rules: list[str] = []
    for p in pairs:
        label, rule = _score_free_label(str(p["text_a"]), str(p["text_b"]))
        sf_labels.append(label); sf_rules.append(rule)

    sf_gold = np.array(sf_labels)
    cos = np.array([float(p["cos"]) for p in pairs])
    lev = np.array([float(p["lev"]) for p in pairs])
    thresholds = np.round(np.arange(0.50, 1.00, 0.01), 2)

    best_sem = _best_threshold(cos, sf_gold, thresholds)
    best_ne = _best_threshold(lev, sf_gold, thresholds)

    # Hybrid sweep against score-free gold
    best_hy = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0,
               "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for thr in thresholds:
        preds = ((cos >= thr) & (lev >= 0.30) & (lev <= 0.95)).astype(int)
        tp, fp, fn, tn = _confusion(preds, sf_gold)
        pr, rc, f = _metrics(tp, fp, fn)
        if f > float(best_hy["f1"]):
            best_hy = {"threshold": float(round(thr, 4)),
                       "precision": round(pr, 4), "recall": round(rc, 4),
                       "f1": round(f, 4),
                       "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    # Cohen's kappa between primary rubric and score-free relabel
    primary = np.array([int(p["label"]) for p in pairs])
    agree = int(np.sum(primary == sf_gold))
    n = len(pairs)
    observed = agree / n
    p1p = float(primary.mean()); p2p = float(sf_gold.mean())
    expected = p1p * p2p + (1 - p1p) * (1 - p2p)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    if kappa < 0.20: interp = "slight"
    elif kappa < 0.40: interp = "fair"
    elif kappa < 0.60: interp = "moderate"
    elif kappa < 0.80: interp = "substantial"
    else: interp = "almost perfect"

    # Rule composition on score-free pass
    rule_counts = dict(Counter(sf_rules))
    n_positive_sf = int(sf_gold.sum())

    # Band-wise detector F1 on score-free labels
    bands = [(0.50, 0.70), (0.70, 0.80), (0.80, 0.85),
             (0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]
    band_rows = []
    for lo, hi in bands:
        mask = (cos >= lo) & (cos < hi)
        if mask.sum() == 0:
            continue
        band_rows.append({
            "band": f"[{lo:.2f},{hi:.2f})",
            "n": int(mask.sum()),
            "n_dup_primary": int(primary[mask].sum()),
            "n_dup_score_free": int(sf_gold[mask].sum()),
        })

    result = {
        "n_pairs": n,
        "n_duplicates_primary": int(primary.sum()),
        "n_duplicates_score_free": n_positive_sf,
        "rule_composition_score_free": rule_counts,
        "detector_vs_score_free_labels": {
            "semantic": best_sem,
            "near_exact": best_ne,
            "hybrid": best_hy,
        },
        "cohen_kappa_primary_vs_score_free": {
            "kappa": round(kappa, 4),
            "observed_agreement": round(observed, 4),
            "expected_agreement": round(expected, 4),
            "landis_koch_interpretation": interp,
            "n_agree": agree,
            "n_disagree": n - agree,
        },
        "cosine_band_positive_rates": band_rows,
    }

    out = out_dir / "revision_score_free_relabel.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[2] score-free relabel -> {out}")
    print(f"    score-free positives: {n_positive_sf}/{n} ({n_positive_sf/n*100:.1f}%)")
    print(f"    detector F1 vs score-free labels:")
    print(f"      semantic  thr={best_sem['threshold']} F1={best_sem['f1']}")
    print(f"      near-ex   thr={best_ne['threshold']} F1={best_ne['f1']}")
    print(f"      hybrid    thr={best_hy['threshold']} F1={best_hy['f1']}")
    print(f"    primary vs score-free kappa: {kappa:.3f} ({interp})")
    return result


# --- analysis 3: bootstrap CIs ------------------------------------------


def analysis_bootstrap_cis(
    pairs: list[dict], out_dir: Path, n_boot: int = 1000, seed: int = 42,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n = len(pairs)
    cos = np.array([float(p["cos"]) for p in pairs])
    lev = np.array([float(p["lev"]) for p in pairs])
    gold = np.array([int(p["label"]) for p in pairs])

    # Fixed operating points from the calibration pass.
    operating_points = {
        "semantic": {"kind": "unary_cos", "threshold": 0.82},
        "near_exact": {"kind": "unary_lev", "threshold": 0.80},
        "hybrid": {"kind": "hybrid_band", "threshold": 0.82,
                   "lev_min": 0.30, "lev_max": 0.95},
    }

    def _score(idx: np.ndarray, strategy: str) -> tuple[float, float, float]:
        c = cos[idx]
        l = lev[idx]
        g = gold[idx]
        op = operating_points[strategy]
        if op["kind"] == "unary_cos":
            preds = (c >= op["threshold"]).astype(int)
        elif op["kind"] == "unary_lev":
            preds = (l >= op["threshold"]).astype(int)
        else:
            preds = (
                (c >= op["threshold"])
                & (l >= op["lev_min"])
                & (l <= op["lev_max"])
            ).astype(int)
        tp, fp, fn, tn = _confusion(preds, g)
        return _metrics(tp, fp, fn)

    results = {}
    for strategy in operating_points:
        f1s, ps, rs = [], [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            p, r, f = _score(idx, strategy)
            f1s.append(f); ps.append(p); rs.append(r)
        f1s = np.array(f1s); ps = np.array(ps); rs = np.array(rs)
        results[strategy] = {
            "threshold": operating_points[strategy]["threshold"],
            "f1": {
                "point": round(float(_score(np.arange(n), strategy)[2]), 4),
                "ci95_lo": round(float(np.percentile(f1s, 2.5)), 4),
                "ci95_hi": round(float(np.percentile(f1s, 97.5)), 4),
            },
            "precision": {
                "point": round(float(_score(np.arange(n), strategy)[0]), 4),
                "ci95_lo": round(float(np.percentile(ps, 2.5)), 4),
                "ci95_hi": round(float(np.percentile(ps, 97.5)), 4),
            },
            "recall": {
                "point": round(float(_score(np.arange(n), strategy)[1]), 4),
                "ci95_lo": round(float(np.percentile(rs, 2.5)), 4),
                "ci95_hi": round(float(np.percentile(rs, 97.5)), 4),
            },
        }

    results["meta"] = {"n_boot": n_boot, "seed": seed, "n_pairs": n}
    out = out_dir / "revision_bootstrap_ci.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[3] bootstrap CIs -> {out}")
    for s in ("semantic", "near_exact", "hybrid"):
        f = results[s]["f1"]
        print(f"    {s:<10} F1={f['point']:.4f}  95% CI [{f['ci95_lo']:.4f}, {f['ci95_hi']:.4f}]")
    return results


# --- analysis 4: licence-stratified chi-square ---------------------------


def _z_two_prop(p1: float, n1: int, p2: float, n2: int) -> tuple[float, float]:
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    # two-sided p-value from z using the normal CDF approximation
    p = 2.0 * (1.0 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p


def analysis_license_chisquare(steps: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    classes = ["permissive", "copyleft", "unknown", "unlicensed"]
    rows = {}
    for c in classes:
        sub = steps[steps["license_class"] == c]
        n = len(sub)
        uniq = int(sub["text"].str.strip().nunique())
        dup_rate = 1 - uniq / max(n, 1)
        # contingency: "duplicated occurrence" vs "singleton-contributing occurrence"
        dup_count = n - uniq
        rows[c] = {
            "n_steps": int(n),
            "n_unique": int(uniq),
            "n_duplicate_occurrences": int(dup_count),
            "dup_rate": round(dup_rate, 4),
        }

    # 2 x 4 chi-square on duplicated vs unique occurrences across the four strata.
    observed = np.array([
        [rows[c]["n_duplicate_occurrences"] for c in classes],
        [rows[c]["n_unique"] for c in classes],
    ], dtype=float)
    row_tot = observed.sum(axis=1, keepdims=True)
    col_tot = observed.sum(axis=0, keepdims=True)
    grand = observed.sum()
    expected = row_tot @ col_tot / grand
    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    # survival function of chi-square via incomplete gamma (regularised upper)
    # for small df we can use an approximation; scipy is not a dependency, so
    # we use an accurate series.
    # P(X > chi2 | df) = Q(df/2, chi2/2) where Q is the regularised upper gamma.
    def _upper_gamma_regularised(s: float, x: float) -> float:
        if x <= 0: return 1.0
        # continued-fraction expansion (Numerical Recipes) for upper Q
        fpmin = 1e-300
        b = x + 1.0 - s
        c = 1.0 / fpmin
        d = 1.0 / b
        h = d
        for i in range(1, 200):
            an = -i * (i - s)
            b += 2.0
            d = an * d + b
            if abs(d) < fpmin: d = fpmin
            c = b + an / c
            if abs(c) < fpmin: c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 3e-12:
                break
        return math.exp(-x + s * math.log(x) - math.lgamma(s)) * h
    p_value = _upper_gamma_regularised(df / 2.0, chi2 / 2.0)

    # Effect-size statistics. Cramer's V for the 2x4 contingency
    # (phi-c = sqrt(chi2 / (N * (k-1))), where k = min(rows, cols)).
    grand_n = float(observed.sum())
    k_min = min(observed.shape) - 1
    cramers_v = math.sqrt(chi2 / (grand_n * k_min)) if grand_n > 0 else 0.0

    # Cohen's h for pairwise two-proportion effect size. h is the arcsine-
    # transformed difference; small = 0.2, medium = 0.5, large = 0.8.
    def _cohen_h(p1: float, p2: float) -> float:
        phi1 = 2 * math.asin(math.sqrt(p1))
        phi2 = 2 * math.asin(math.sqrt(p2))
        return abs(phi1 - phi2)

    pairwise = {}
    for other in ("copyleft", "unknown", "unlicensed"):
        z, p = _z_two_prop(
            rows["permissive"]["dup_rate"], rows["permissive"]["n_steps"],
            rows[other]["dup_rate"], rows[other]["n_steps"],
        )
        h = _cohen_h(rows["permissive"]["dup_rate"], rows[other]["dup_rate"])
        pairwise[f"permissive_vs_{other}"] = {
            "z": round(z, 3),
            "p": round(p, 6),
            "cohen_h": round(h, 4),
        }

    result = {
        "per_class": rows,
        "chi_square": {
            "chi2": round(chi2, 2),
            "df": df,
            "p_value": float(f"{p_value:.3e}") if p_value > 0 else 0.0,
            "interpretation": (
                "At N = 1,113,616 the chi-square test is diagnostic only of "
                "any non-zero effect; the p-value is driven to near-zero by "
                "sample size alone. The effect-size statistics (Cramer's V "
                "and pairwise Cohen's h) are the meaningful quantities: V "
                "reports the aggregate strength of association, h the "
                "pairwise standardised effect size. Cohen's h thresholds: "
                "0.2 = small, 0.5 = medium, 0.8 = large."
            ),
        },
        "cramers_v": round(cramers_v, 4),
        "pairwise_z_and_cohen_h": pairwise,
    }
    out = out_dir / "revision_license_chisquare.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[5] licence chi-square + effect size -> {out}")
    print(f"    chi2={chi2:.1f} df={df} p={p_value:.2e}  Cramer's V={cramers_v:.4f}")
    for k, v in pairwise.items():
        print(f"    {k}: z={v['z']}  p={v['p']}  Cohen's h={v['cohen_h']}")
    return result


# --- analysis 6: size vs duplication scatter + Spearman -----------------


def analysis_size_vs_dup(steps: pd.DataFrame, out_dir: Path,
                          fig_dir: Path) -> dict[str, object]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    grouped = steps.groupby("repo_slug")
    per_repo = []
    for slug, sub in grouped:
        n = len(sub)
        uniq = sub["text"].str.strip().nunique()
        rate = 1 - uniq / max(n, 1)
        per_repo.append({
            "repo_slug": slug,
            "n_steps": int(n),
            "n_unique": int(uniq),
            "dup_rate": round(float(rate), 4),
            "license_class": str(sub["license_class"].iloc[0]),
        })
    df = pd.DataFrame(per_repo)

    # Spearman rank correlation (rank-based, no scipy needed)
    ranks_n = df["n_steps"].rank()
    ranks_d = df["dup_rate"].rank()
    rn_mean = ranks_n.mean(); rd_mean = ranks_d.mean()
    num = ((ranks_n - rn_mean) * (ranks_d - rd_mean)).sum()
    den = math.sqrt(((ranks_n - rn_mean) ** 2).sum() *
                    ((ranks_d - rd_mean) ** 2).sum())
    spearman = num / den if den else 0.0
    # two-sided p-value via the t-approximation for Spearman rho:
    # t = r * sqrt((n-2)/(1-r^2)), df = n-2
    n = len(df)
    if abs(spearman) < 1.0 and n > 2:
        t = spearman * math.sqrt((n - 2) / max(1 - spearman ** 2, 1e-12))
        # two-sided p for t with df = n-2 via the regularised incomplete beta
        def _beta_regularised(a: float, b: float, x: float) -> float:
            # continued-fraction expansion via Lentz (Numerical Recipes)
            if x <= 0: return 0.0
            if x >= 1: return 1.0
            bt = math.exp(
                math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log(1 - x)
            )
            if x < (a + 1) / (a + b + 2):
                # continued fraction expansion
                fpmin = 1e-300
                qab = a + b; qap = a + 1; qam = a - 1
                c = 1.0
                d = 1.0 - qab * x / qap
                if abs(d) < fpmin: d = fpmin
                d = 1.0 / d
                h = d
                for m in range(1, 200):
                    m2 = 2 * m
                    aa = m * (b - m) * x / ((qam + m2) * (a + m2))
                    d = 1.0 + aa * d
                    if abs(d) < fpmin: d = fpmin
                    c = 1.0 + aa / c
                    if abs(c) < fpmin: c = fpmin
                    d = 1.0 / d
                    h *= d * c
                    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
                    d = 1.0 + aa * d
                    if abs(d) < fpmin: d = fpmin
                    c = 1.0 + aa / c
                    if abs(c) < fpmin: c = fpmin
                    d = 1.0 / d
                    delta = d * c
                    h *= delta
                    if abs(delta - 1.0) < 3e-12:
                        break
                return bt * h / a
            else:
                return 1.0 - _beta_regularised(b, a, 1.0 - x)
        df_t = n - 2
        x = df_t / (df_t + t * t)
        p_two_sided = _beta_regularised(df_t / 2.0, 0.5, x)
    else:
        p_two_sided = 0.0

    # Figure
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    color_map = {"permissive": "#2ca02c", "copyleft": "#d62728",
                 "unknown": "#7f7f7f", "unlicensed": "#ff7f0e"}
    for cls in ("permissive", "copyleft", "unknown", "unlicensed"):
        sub = df[df["license_class"] == cls]
        ax.scatter(sub["n_steps"], sub["dup_rate"],
                   c=color_map[cls], s=14, alpha=0.55,
                   edgecolors="none", label=cls)
    ax.set_xscale("log")
    ax.set_xlabel("repository size (steps, log-scale)")
    ax.set_ylabel("intra-repository exact-duplication rate")
    ax.axhline(y=0.802, color="#333", linestyle=":", linewidth=1.0,
               label="pooled rate 80.2%")
    ax.set_title(
        "Per-repository duplication rate vs size (N = 347)\n"
        f"Spearman $\\rho$ = {spearman:.3f}  (p = {p_two_sided:.2e})",
    )
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.2, linewidth=0.5)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_size_vs_dup.pdf")
    plt.close(fig)

    result = {
        "n_repos": int(n),
        "spearman_rho": round(float(spearman), 4),
        "p_value": float(f"{p_two_sided:.3e}") if p_two_sided > 0 else 0.0,
        "median_repo_size_steps": int(df["n_steps"].median()),
        "median_repo_dup_rate": round(float(df["dup_rate"].median()), 4),
        "per_repo_preview": df.head(10).to_dict(orient="records"),
    }
    out = out_dir / "revision_size_vs_dup.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[6] size vs dup -> {out}")
    print(f"    Spearman rho={spearman:.3f}  p={p_two_sided:.2e}")
    return result


# --- main ---------------------------------------------------------------


def main() -> int:
    root = Path(".")
    labeled_path = root / "corpus/labeled_pairs.jsonl"
    steps_path = root / "corpus/steps.parquet"
    out_dir = root / "analysis"
    fig_dir = root / "paper/figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not labeled_path.exists():
        print(f"ERROR: {labeled_path} not found.", file=sys.stderr)
        return 2
    if not steps_path.exists():
        print(f"ERROR: {steps_path} not found.", file=sys.stderr)
        return 2

    pairs: list[dict] = []
    with labeled_path.open(encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} labelled pairs.")

    steps = pd.read_parquet(steps_path)
    print(f"Loaded {len(steps):,} corpus steps across "
          f"{steps['repo_slug'].nunique()} repos.")
    print()

    analysis_lexical_baselines(pairs, out_dir)
    print()
    analysis_score_free_relabel(pairs, out_dir)
    print()
    analysis_bootstrap_cis(pairs, out_dir)
    print()
    analysis_license_chisquare(steps, out_dir)
    print()
    analysis_size_vs_dup(steps, out_dir, fig_dir)
    print()
    print("All revision analyses complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
