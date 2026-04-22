# Labeling rubric for `labeled_pairs.jsonl`

This document specifies the decision rule applied to every pair in
`corpus/labeled_pairs.jsonl`. Each row carries `"label": 1` for duplicate
and `"label": 0` for not-duplicate. The `"labeler"` field records
provenance; in the present version the labeller is an LLM applying this
rubric, with the author verifying every label by reading the file.

## 1. Operational definition of "duplicate"

Two Gherkin step texts are **duplicates** if and only if a competent
maintainer of a BDD suite would consolidate them into a single step
definition, parameterising over the positions in which the two texts
differ. All other pairs are **not duplicates**.

This is an *intent-equivalence* definition, not a *behavioural-
equivalence* definition: it concerns the maintenance intent a human
would apply when viewing the two step phrasings side by side, not the
glue-code that each phrasing happens to bind to at execution time. Two
step texts that bind to divergent glue implementations in practice are
still duplicates under this rubric if the maintainer's intent would be
to consolidate the phrasings and reconcile the glue.

## 2. Rules

Applied in order. The first matching rule wins.

### R1. Parametric variants of the same template → DUPLICATE.

If the two texts share an identical structural template and differ only
in quoted string literals, numeric literals, bracketed identifiers, or
other named argument positions, they are duplicates.

Examples:
- `the current account is "test1"` vs `the current account is "test2"` → **DUP**
- `the mark price should be "15900" for the market "ETH/FEB22"` vs `the mark price should be "1590" for the market "ETH/DEC19"` → **DUP**
- `I run \`rspec a_spec.rb\`` vs `I run \`rspec b_spec.rb\`` → **DUP**
- `def fundId2 = callonce uuid6` vs `def fundId3 = callonce uuid9` → **DUP** (same Karate idiom, different variable)

### R2. Synonym-level paraphrase of the same intent → DUPLICATE.

If the two texts express the same assertion or action in different
English surface forms (synonyms, voice changes, minor grammatical
reordering), they are duplicates.

Examples:
- `I see an input field to enter my password` vs `I see an input field to type my password` → **DUP**
- `no scroll indicators should be shown` vs `no scroll indicators should be displayed` → **DUP**

### R3. Structural swap of argument positions → DUPLICATE.

If the two texts differ only in the order of named arguments inside
the same template and the template is symmetric in those positions
(i.e., the assertion is invariant under the swap), they are duplicates.

Example:
- `branch "X" now has type "Y"` vs `branch "Y" now has type "X"` → **DUP**

### R4. Different framework keywords with different semantics → NOT DUPLICATE.

If the two texts differ in a framework-level keyword whose semantics
are intentionally distinct, they are not duplicates. This overrides R1.

Examples:
- `def v = call createFund { ... }` vs `def v = callonce createFund { ... }` → **NOT DUP** (Karate `call` re-executes; `callonce` caches)
- `def x = uuid()` vs `def x = callonce uuid` → **NOT DUP** (different invocation mechanism)
- `set poLine.cost = X` vs `def poLine.cost = X` → **NOT DUP** (different variable-scoping in Karate)

### R5. Different HTTP verbs → NOT DUPLICATE.

If the two texts use different HTTP verbs (GET vs POST vs PUT vs
PATCH vs DELETE), they are not duplicates even when the rest of the
template is identical.

Examples:
- `I send a GET request to "/accounts/x"` vs `I send a POST request to "/accounts/x"` → **NOT DUP**
- `I perform a HTTP "PATCH" request to "/queues"` vs `I perform a HTTP "GET" request to "/queues"` → **NOT DUP**

But if both use the same verb and only the URL differs:
- `I send a GET request to "/accounts/components"` vs `I send a GET request to "/accounts/engines"` → **DUP** (parametric, R1 applies)

### R6. Opposite-polarity assertions → NOT DUPLICATE.

If the two texts differ in a polarity marker (`should` vs `should not`,
`has` vs `has not`, `is` vs `is not`, present vs absent), they are not
duplicates.

Examples:
- `I should see notification about create new user` vs `I should not see notification about create new user` → **NOT DUP**
- `check general log has "X"` vs `check general log has not "X"` → **NOT DUP**

### R7. Presence vs content, existence vs modification → NOT DUPLICATE.

If the two texts differ in the nature of the operation or assertion
(checking that X exists vs checking X's content, creating X vs
deleting X, reading vs writing), they are not duplicates.

Examples:
- `file "a.txt" should exist` vs `the content of file "a.txt" should be "X"` → **NOT DUP**
- `I go to the groups page` vs `I should not see link to the groups page` → **NOT DUP** (navigation vs assertion)
- `user deletes file X` vs `file X should exist` → **NOT DUP**

### R8. Same prefix but different operation → NOT DUPLICATE.

If the two texts share a surface prefix but the core verb or operation
differs in a way that is semantically load-bearing, they are not
duplicates.

Examples:
- `I follow "X"` vs `I fill in "X" with "Y"` → **NOT DUP** (link-click vs form-fill)
- `I see the select component` vs `they select all items in the current page` → **NOT DUP**

### R9. Different scenario-outline placeholder text with same template → DUPLICATE.

Scenario Outlines expose `<placeholder>` tokens that are substituted
from Examples tables. Two outline-step texts differing only in the
placeholder identifier are duplicates.

Example:
- `the user logs in as "<role>"` vs `the user logs in as "<user-type>"` → **DUP** (purely cosmetic difference in the outline variable name)

### R10. Default → NOT DUPLICATE.

If no rule above applies, treat the pair as not-duplicate. This
is the conservative default: false negatives are less costly than
false positives when the downstream use is a maintainer's "suggest
consolidation" tool.

## 3. Known limitations of this rubric

1. **Intent-equivalence is judged by the labeller, not measured.** We
   have no access to the glue code behind each step. Two step texts
   that look consolidatable may in practice bind to intentionally
   distinct implementations; those are false positives under this
   rubric (though they are still arguably worth flagging to a human
   reviewer).
2. **The rubric does not operationalise "competent maintainer".** Two
   maintainers may disagree on borderline parametric cases; we default
   to R1 (consolidation-oriented) when the template structure is clear.
3. **Framework-specific knowledge is needed for R4.** The labeller
   applies R4 conservatively only to clearly-documented keyword
   distinctions (Karate `call`/`callonce`, `def`/`set`). Obscure
   framework idioms not listed above fall through to R10 (not-dup).
4. **This is not a behavioural-equivalence definition.** A paper
   claiming behavioural equivalence would require execution traces
   (as in Binamungu et al. 2018); that is outside this rubric's scope.
5. **Partial circularity with the detector under evaluation.** Rules
   R1, R2, and R3 use cosine and Levenshtein-ratio cut-points to
   decide parametric variants, paraphrases, and structural swaps, and
   those cut-points live in the same similarity space that the
   detector under evaluation operates in. Consequently, a threshold
   calibration on the resulting labels partly measures the rubric's
   own cut-points rather than the detector's standalone discriminative
   power. The non-circular rules (R4--R9) cover approximately 80 pairs
   out of 1{,}020; the remaining 940 pairs are labelled by R1--R3 or
   R10 (default), which are score-correlated. The paper's threats
   section acknowledges this; the only fully-defensible remedy is
   independent human labels, proposed as future work. Until then,
   readers should treat the reported F1 numbers as an upper bound on
   the detector's agreement with this specific rubric, not as a pure
   estimate of the detector's accuracy against human ground truth.

## 4. Validation procedure

The `labeler` field in each `labeled_pairs.jsonl` row is
`"llm"`. The author validates by reading the full file and flipping
any label that the rubric disagrees with, or by amending this rubric
when a class of pair is systematically mislabelled.

This process is documented so that an independent reviewer can audit
both the rubric and its application.
