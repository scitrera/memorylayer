# Design: Graphiti-derived adoptions — supersession, fuzzy resolution, tier batching

**Status:** Evaluated and partially shipped. Verdicts + measurements under "Findings";
see "Recommended sequence" for what has landed and what has not.
**Scope:** OSS core (`services/memory`, `services/contradiction`, `services/entity_registry`,
`services/semantic_tiering`), with automatic pickup by the enterprise AGE materializer.
**Source:** review of [getzep/graphiti](https://github.com/getzep/graphiti) @ 0.29.3, **Apache-2.0**
(code may be borrowed directly, unlike the AGPL Honcho review which was concepts-only).

## Motivation

Two concerns drove this review:

1. **Knowledge-graph quality.** Our own moat eval found the graph a wash on LoCoMo QA
   (moat-ON 77.8 vs baseline 77.5), and naive `similar_to` association injection did not
   help. We need to know whether that means "graphs don't help us" or "we are missing the
   mechanism by which a graph helps".
2. **LLM calls at scale.** Whether the per-memory inference cost is sustainable.

Graphiti is the closest well-engineered comparison: a temporal knowledge graph over
episodic input, with an explicitly cost-tuned ingestion path.

## Measured baseline — where our inference budget actually goes

Per `remember()`, counted from call sites:

| Stage | Calls | Site |
|---|---|---|
| Fact decomposition | 1 | `extraction/default.py` |
| **per decomposed fact:** tier generation | **2** (sequential) | `semantic_tiering/default.py:158,167` |
| **per fact:** auto-associate classify | 1 (batched) | `association/default.py:260` |
| **per fact:** optional type classify | 0–1 | `memory/default.py` auto_enrich |
| Contradiction check | **0** (regex) | `contradiction/default.py` |

**A 5-fact memory costs ~16 LLM calls, and ~2/3 of that is tier generation.**

Graphiti, per episode: ~4–6 calls total — extract nodes, one *batched* dedup call over
only the nodes deterministic matching could not resolve, extract edges, per-edge resolve,
and entity summaries batched **30 per call** behind an optional `should_summarize_node`
filter. Every maintenance call runs on `ModelSize.small`.

The asymmetry is not graph work. It is that **we summarize every memory** while Graphiti
summarizes **entities**, batched and skippable.

---

## Candidate 1 — Supersession at recall (quality)

### Problem

`ContradictionService.check_new_memory()` runs on every store, detects negations, and
writes `ContradictionRecord`s carrying `newer_memory_id`. Grepping the recall path for
`superseded|newer_memory|invalid_at|valid_at|expired` returns **nothing**. The only
consumer is `knowledgebase/default.py:903`, for article generation.

**We positively identify a fact as stale and then retrieve it at full score.**

This is the explicit non-goal of `DESIGN_contradiction_association_unification.md`
("Changing recall's contradiction-aware ranking (separate concern)"), so it is unowned.

### Graphiti's approach

Bi-temporal edges. `resolve_edge_contradictions` (`edge_operations.py:538`) is **pure
timestamp arithmetic, zero LLM**: when a new edge's `valid_at` postdates an existing
contradicted edge, the old edge gets `invalid_at` + `expired_at` set, and expired edges
drop out of search.

### Proposed adaptation

Add `superseded_by` / `invalid_at` to the memory read path — not a new detector. Recall
gains a policy knob:

- `exclude` — filter superseded memories from results
- `demote` — multiplicative score penalty (default; safer, reversible)
- `off` — current behavior

Supersession is already computed (`_determine_newer_memory`); this only makes recall
consult it.

### Why this is the interesting one

It is a *different lever* from what our moat eval tested. That eval asked whether adding
association edges improves ranking. Supersession does not add candidates — it **removes
wrong ones**. A graph can fail the first test and pass the second.

### Risk

LoCoMo rewards recalling *any* mention; aggressive exclusion could drop a still-correct
older memory. Hence `demote` as the default and an eval arm per mode.

---

## Candidate 2 — Entropy-gated fuzzy entity resolution (quality, zero inference)

### Problem

`entity_registry/default.py` resolves exact-normalized name → alias, and the code comment
defers everything harder to "the enterprise GLiNER2/fuzzy/LLM tier". OSS therefore never
merges `Bob Smith` / `Robert Smith`, splitting an entity across mentions.

### Graphiti's approach

`dedup_helpers.py`: MinHash (32 permutations) + LSH banding over 3-gram shingles,
Jaccard ≥ 0.9 — **gated by Shannon entropy**. Names below `_NAME_ENTROPY_THRESHOLD = 1.5`,
or shorter than 6 chars with <2 tokens, are explicitly *not* trusted to fuzzy-match and
defer to the LLM instead.

The entropy gate is the load-bearing idea: it is what makes a cheap heuristic *safe*,
by refusing to apply it where it is unreliable.

### Proposed adaptation

Insert a fuzzy tier into OSS resolution: exact → alias → **fuzzy(MinHash + entropy gate)**
→ (enterprise LLM/GLiNER2 tier). ~200 lines, dependency-free, Apache-2.0, **no inference
cost at all**. Feeds the existing alias machinery rather than replacing it.

### Risk

False merges are worse than misses — conflating two real people corrupts every downstream
mention. Eval must report false-merge rate, not just merge count.

---

## Candidate 3 — Tier generation batching (cost)

### Problem

2 sequential LLM calls per fact, unconditional. The dominant term in our budget.

### Graphiti's approach

Entity summaries batched at `MAX_NODES = 30` per call, behind an optional
`should_summarize_node` filter, on the small model.

### Proposed adaptation, cheapest first

1. **Skip filter.** A memory shorter than its own target abstract needs no tiers.
   Pure win, no quality risk.
2. **Batch across decomposed facts.** The facts from one `remember()` are already in
   hand; one call can tier all of them. Turns N×2 into ~2.
3. **Fuse overview+abstract into one call** returning both fields.

Also verify the `tier_generation` LLM profile targets a small model (not found in
`config.py` during review).

### Risk

(3) is the one to be careful with: the sequential chain is deliberate — "shorter input =
better short summaries" — so fusing may degrade abstract quality. Needs a quality diff,
not just a call count.

---

## Candidate 4 — Fused duplicate + contradiction detection (quality)

### Problem

Contradiction detection is `_has_negation_pattern` (regex negation word-pairs) and
`_extract_entity_values` (regex triples compared with `val_a != val_b`). Cheap — zero
inference — but likely low recall on natural phrasing.

### Graphiti's approach

`resolve_edge` returns `duplicate_facts` **and** `contradicted_facts` from **one**
small-model call, with an exact-normalized-fact short-circuit before it.

### Relationship to the existing design doc

`DESIGN_contradiction_association_unification.md` already owns this problem and proposes
Phase 3: an NLI cross-encoder detector. **These are competing implementations of the same
phase**, and should be evaluated as alternatives:

| | NLI cross-encoder (existing plan) | Fused small-model call (Graphiti) |
|---|---|---|
| Cost | 1 cross-encoder pass | 1 small LLM call |
| Latency | lower | higher |
| Output | entailment/contradiction only | duplicates + contradictions together |
| Ontology reuse | shares the ontology provider endpoint | separate prompt |

This doc does not re-litigate that; it supplies the **quality floor measurement** that
tells us how much headroom an upgrade has.

---

## Eval plan

All four are measurable against the cached LoCoMo corpus
(`benchmarks/.cache/raw/locomo/locomo10.json`, 2.8 MB) without a live server:

| Candidate | Method | Corpus | Needs LLM? |
|---|---|---|---|
| 1 Supersession | detector firing rate on cases that require supersession | **LongMemEval `knowledge-update`** (78 items) — *not* LoCoMo, see Findings | no |
| 2 Fuzzy dedup | merge count + **false-merge rate** over real extracted entity names | LoCoMo | no |
| 3 Tier batching | static call-count model over real memory-length distribution | LoCoMo + LongMemEval | no (quality diff does) |
| 4 Fused detection | recall/precision of the current regex detector | capability set + 40k real LoCoMo pairs | no |

All four were decided on evidence without spending inference budget. A QA arm is only
warranted after C4 lands, and it must run on LongMemEval `knowledge-update`.

---

## Findings

Experiments were run from `.slop/graphiti-eval/` (gitignored, so the scripts are local to
whoever reran them — the method for each is described inline below and against the cached
corpora under `benchmarks/.cache/`). All offline; no inference spent to reach any verdict.

### Verdict table

| # | Candidate | Verdict | Basis |
|---|---|---|---|
| 3 | Tier skip filter | **ADOPT — shipped** | 44–57% fewer LLM calls per `remember()` |
| 4 | Fused detector + floor 0.5 | **ADOPT — measured** | **0.0% → 55.1%** on LongMemEval knowledge-update; both changes required |
| 1 | Supersession at recall | **BUILT, ships OFF** | no significant end-to-end QA gain: off 74.4%, demote 69.2%, exclude 76.9% |
| 2 | Fuzzy entity dedup | **REJECT for OSS** | 1 merge per 1040 entities; laxer thresholds are unsafe |

The two shipped items (word-boundary fix, tier skip filter) are in `749091d` / `779e3ee`;
profile-routing diagnostics in `82a532b`.

### C3 — tier skip filter: the cost win is real and larger than batching

Measured over the LoCoMo corpus as ingested (n=5882, median memory **145 chars**):

- **100%** of memories are already shorter than the overview target ("2–3 sentences", ~500 chars)
- **76.2%** are shorter than the abstract target ("a single short sentence", ~200 chars)

We are paying an LLM to write a 2–3 sentence overview of text that is *already* 1–2
sentences. Calls per `remember()` with N decomposed facts:

| variant | N=1 | N=5 | N=10 |
|---|---|---|---|
| current | 4.0 | 16.0 | 31.0 |
| **+ skip filter** | **2.2** | **7.2** | **13.4** |
| + batch across facts (no skip) | 4.0 | 8.0 | 13.0 |
| + fuse overview/abstract only | 3.0 | 11.0 | 21.0 |

**44% / 55% / 57% reduction** at N=1/5/10. Note skip and batch are *not* additive here —
once the skip filter fires there is nothing left to batch. They serve different workloads:

- **Skip filter** wins on short/decomposed-fact (conversational) content.
- **Batching** wins on long-content ingestion, where nothing is skippable — LongMemEval
  sessions have a median of **10,228 chars**, so 0.9% are skippable and batching is what
  takes N=10 from 31 → 13.

Ship the skip filter first (no quality risk — it only declines to summarize text already
shorter than the summary), then batching for the document/session path. **Fusing
overview+abstract is the weakest of the three** and carries the quality risk flagged
above; deprioritize it.

### C4 — the detector is below its useful floor, and has a real bug

`_has_negation_pattern` on 12 realistic supersessions and 8 similar-but-not-contradictory
pairs:

- **RECALL 2/12 (16.7%).** The only hits are the two that literally contain `is not` /
  `should not`. Relocation, job change, preference change, count change, date change,
  status change, possession change, and value update are all missed — as is the explicit
  flip `"I don't drink coffee"` → `"I drink coffee every morning now"`.
- **3/8 false positives.**

Root cause is a genuine bug: matching is `term in text`, a **substring** test, not
word-boundary. So `"can"` matches inside `"cannot"` and `"is"` inside `"island"`, and
because `"is"` is a substring of `"is not"`, **any two texts that both contain `"is not"`
flag as contradicting each other**. On 40,000 real LoCoMo utterance pairs it fires **514
times (1.29%)**, and the sampled firings are all spurious (`"Can't wait to create
something"` vs `"Can't wait to make more memories"`).

A word-boundary + polarity-differs fix removes 4/5 of the false positives, keeps both
true positives, and drops in-the-wild firing 514 → 391. **That is a small, shippable fix
worth making immediately**, independent of any redesign — but 16.7% recall is a ceiling
regex cannot raise. The architectural upgrade (NLI cross-encoder per the unification doc,
or Graphiti's fused call) is justified.

### C1 — supersession is right, but blocked, and LoCoMo cannot judge it

Two findings, one methodological and one blocking.

**LoCoMo is the wrong benchmark.** Its "temporal" category (cat 2, 16.2%) asks *"when did
X happen"* — historical dates. It *rewards* retrieving stale facts with their timestamps
and would penalize demoting them. The right benchmark is **LongMemEval's
`knowledge-update` type — 78 items (15.6%)**, which asks for the *current* value after a
change (e.g. a 5K personal best stated once, then improved; only the later value is
correct). Any future supersession arm must run there, not on LoCoMo.

**The signal isn't there yet.** Running the current detector over the evidence turns of
all 78 knowledge-update items: it fires on **6 of 78 (7.7%)**, flagging 23 of 4921 pairs
(0.47%). Wiring recall to a signal that is absent in 92% of the cases that need it — and
polluted with false positives — cannot help and can actively demote correct memories.

**This inverts the original ordering.** Supersession-at-recall was the most attractive
idea on review; measurement shows it is downstream of C4. Sequence: fix detector → verify
recall on knowledge-update → *then* wire supersession, `demote` mode first.

### C4/C1 follow-up — the candidate neighbourhood is the real ceiling

Before choosing between an NLI cross-encoder and a fused LLM call, we measured the bound
both are subject to. `check_new_memory` only ever compares a new memory against the
**top-20 neighbours at `min_relevance >= 0.7`**. A pair outside that window is never shown
to any detector, so detector quality is irrelevant below it.

Simulating the real retrieval over all 78 LongMemEval knowledge-update items, embedding
every haystack user turn with all-MiniLM-L6-v2 (the embed-server default) and asking
whether the superseded fact is co-retrieved with its replacement:

| relevance floor | items where the pair is reachable |
|---|---|
| **0.7 (current)** | **51.3%** |
| 0.6 | 71.8% |
| 0.5 | 93.6% |
| 0.4 | 100.0% |

Median best old↔new similarity is **0.707** — the threshold sits directly on top of the
distribution it is supposed to admit, and 38 of 78 items have their best pair below it.

**A perfect detector caps at 51% today.** Building the NLI endpoint first would have
bought a better detector for pairs it is never shown. The floor is a single constant and
lifting it to 0.5 raises the ceiling to 93.6%.

The two are coupled and must move together: a lower floor shows the detector more pairs,
which with today's regex (measured precision problems above) means proportionally more
false contradictions, and with a per-pair LLM detector means proportionally more calls.
Sequence is therefore: raise the ceiling and improve precision in the same change, then
measure. Not one without the other.

**Blocking infrastructure note.** `MEMORYLAYER_NLI_URL` / the `cross_encoder` ontology
provider are written against a `POST {nli_url}/v1/nli` contract that **is not implemented
in either the OSS or enterprise embed-server**, and no NLI-finetuned model is present
locally (`deberta-v3-base` is the un-finetuned base, no entailment head). "Evaluate the
NLI option" is therefore build-an-endpoint-and-ship-a-model, not a measurement. That cost
should be weighed against the fused-LLM-call alternative, which needs no new service.

### C4 bake-off — measured: floor and detector are both necessary, detector dominates

Ran all three arms over the same byte-identical candidate sets (78 items, 397 probe
events, every LongMemEval knowledge-update item; fused call = Graphiti's `resolve_edge`
shape returning duplicates **and** contradictions in one call, on
`accounts/fireworks/models/qwen3p7-plus`):

| arm | items detected | note |
|---|---|---|
| regex @ floor 0.7 (**today**) | **0.0%** (0/78) | |
| regex @ floor 0.5 | 3.8% (3/78) | floor alone barely helps a regex |
| **fused LLM @ floor 0.5** | **55.1%** (43/78) | 75 of 123 flags land on the target pair |
| *ceiling @ 0.5 (pair reachable)* | *93.6%* | |

**Correction to an earlier figure in this doc.** The 7.7% quoted for the current detector
was measured loosely — regex firing on *any* pair of turns within the two answer sessions,
whether or not retrieval would ever surface that pair, and whether or not the flagged
partner was the actual superseded fact. Measured strictly (candidate actually retrieved
**and** it is the true superseded fact) the current detector scores **0.0%**. The bar was
lower than reported; the conclusion is unchanged and strengthened.

Reading the decomposition:

- **The floor alone is not the fix.** 0% → 3.8% with regex. Raising it is necessary but
  buys almost nothing on its own.
- **The detector is the dominant term.** At the same floor, 3.8% → 55.1%.
- **But the floor was still required.** 55.1% is *above* the 51.3% ceiling that floor 0.7
  imposes, so the LLM could not have reached this at the old floor no matter how good it
  is. They genuinely have to ship together, as predicted.
- **Headroom remains** — but see the validity finding immediately below before treating
  the gap to 93.6% as a target. It is mostly not real.

#### ⚠ Validity: the per-pair label is a proxy, and the absolute numbers are lower bounds

Follow-up error analysis (`expA3`–`expA5`) invalidates the *absolute* level of every
number in this section, though not the comparisons between arms.

`is_target` marks **any user turn from the earlier answer session**. A LongMemEval session
is a whole conversation, and most of its turns have nothing to do with the fact under
test. So a pair gets labelled "the supersession we should have caught" when it is actually
two unrelated turns, and the detector is scored as *missing* something it correctly
rejected.

Quantified over the 34 misses: for **12** of them the labelled target turn is topically
unrelated to the question (embedding relevance 0.13–0.28 — e.g. *"How often do I see
Dr. Johnson?"* at 0.129). Raw recall 53.4% becomes **63.9%** once those are removed from
the denominator.

Reading the remaining 22 "plausible" misses by hand shows the filter is still far too
lenient — they are overwhelmingly not supersessions either:

| question | NEW turn | labelled OLD turn |
|---|---|---|
| What day is my cocktail class? | "class on Friday…" | appetizer recommendations |
| How many engineers do I lead? | group hike for 6 people | team-building package |
| French press coffee ratio? | how to clean the press | trying the new beans |

**Conclusion: true detector recall is materially higher than any number reported here, and
the 93.6% "ceiling" is not a reachable target — it is the reachability of a proxy label.**

What survives:

- **The comparisons.** Every arm was scored on the identical metric and candidate sets, so
  regex 0.0% vs fused ~49–53% is a real ordering with a margin far larger than the noise,
  as is qwen vs deepseek and floor 0.7 vs 0.5. The shipped decisions stand.
- **The floor argument**, which is structural: a pair outside the retrieval window is
  never shown to any detector, independent of labelling.

What does not survive: the absolute recall levels, the "~40% of shown pairs missed"
framing, and **any plan to tune the prompt against this metric**. Optimising toward these
labels would push the detector to flag unrelated pairs — i.e. to trade real precision for
proxy recall. That work is explicitly not worth doing until the metric is fixed.

**The fix is not better labels, it is a different metric.** The detector exists to improve
answers, so measure that: an end-to-end QA arm on LongMemEval `knowledge-update`, with
supersession applied at recall, needs no per-pair ground truth and scores the thing we
actually care about.

#### Model comparison — the cheap model holds up

Re-ran the identical sweep against the cheaper text-only profile (routed via
`MEMORYLAYER_LLM_ASSIGN_ONTOLOGY=cheap`, which is where a fused detector would live):

| model | detected | flags | precision proxy | rel. cost |
|---|---|---|---|---|
| `qwen3p7-plus` (multimodal default) | **55.1%** (43/78) | 123 | 61.0% | 1× |
| `deepseek-v4-flash-0731` (cheap) | **48.7–50.0%** (38–39/78) | 95 | 65.3% | **~1/5.7×** |

The cheap model gives up roughly 5–6 points of recall for a **5.7× cost reduction**, and
is *more* conservative rather than sloppier: fewer flags (95 vs 123) at a slightly higher
precision proxy. That is the right failure direction for a detector whose false positives
would demote correct memories.

Against the reachable set (73 of 78 items — 5 are unreachable at any floor), that is
58.9% vs 52.1%. **Recommend the cheap model for this path**; the recall gap does not
justify 5.7× on a per-stored-memory hot path, and the headroom to the ceiling is larger
than the gap between the two models anyway.

Range rather than a point estimate because 6 calls persisted in truncating; only 1 of
them belongs to an item that is otherwise undetected, which bounds the true value to
48.7–50.0%.

The 61% "precision proxy" is a proxy only: a flag on a non-target candidate is not
necessarily wrong, since the neighbourhood contains other genuinely superseded facts that
LongMemEval does not label. It bounds nothing on its own.

**Cost caveat, and it is significant.** The configured model is a *reasoning* model: it
spent ~1,200–5,000 completion tokens per classification, and at 2,000 max_tokens 34 calls
truncated mid-reasoning. Truncation parses as "no contradictions found", i.e. a harness
failure that is indistinguishable from a clean negative — the first sweep silently scored
0% for exactly this reason. Any production use must assert on `finish_reason`. For a
per-stored-memory hot path, a **non-reasoning** small model should be evaluated before
adopting this shape; the reasoning budget is likely the dominant cost here, not the call
itself.

### C2 — fuzzy resolution does not transfer to us

Ported Graphiti's algorithm verbatim and ran it over entity names extracted from LoCoMo,
after our own `normalize_entity_name` had already collapsed exact matches (1040 distinct
entities; 481 after excluding a sentence-initial-capitalization confound).

At Graphiti's own threshold (Jaccard ≥ 0.9): **1 merge** — `german shepherd` /
`german shepherds`, a plural a stemmer would catch. Relaxing the threshold does not
surface real variants, it surfaces **false merges**: `changing`/`hanging`,
`reaching`/`teaching`, `catching`/`watching`, `researching`/`searching`.

The reason it fails here is structural, not a tuning problem. 3-gram Jaccard at 0.9 only
matches near-identical strings; `Bob Smith`/`Robert Smith` scores nowhere near it. In
Graphiti this tier is **not a recall improvement for entity linking** — it is a **cost
optimization that short-circuits near-identical restatements before they reach the LLM
dedup call**. OSS has no LLM dedup tier for it to short-circuit, so it buys nothing.

Correct reframing: this is only interesting to the **enterprise** GLiNER2/fuzzy/LLM tier,
where it could cut LLM dedup calls. It is not an OSS quality lever. Do not adopt as
proposed. (The entropy gate remains a good idea to remember for any future cheap-heuristic
tier: apply the shortcut only where it is reliable, defer the rest.)

## Recommended sequence

1. **Word-boundary + polarity fix** to `_has_negation_pattern`. Small, self-contained,
   removes 4/5 measured false positives and 24% of in-the-wild spurious firings. Ship now.
2. **Tier skip filter.** Largest cost win (44–57%), no quality risk. Ship now.
3. **Tier batching** for the long-content/document path, where skip does not apply.
4. **Raise the candidate ceiling.** ✅ **Shipped** (`25442c2`) as configuration:
   `MEMORYLAYER_CONTRADICTION_MIN_RELEVANCE` / `_CANDIDATE_LIMIT`, with the seam
   `fetch_candidates` shared by both providers. The deterministic provider keeps 0.7 (no
   behaviour change); the `llm` provider defaults to 0.5, so the floor moves only with a
   detector that can absorb it.
5. **Detector upgrade.** ✅ **Shipped** (`25442c2`) as an opt-in provider:
   `MEMORYLAYER_CONTRADICTION_PROVIDER=llm`, routed through a new `contradiction`
   activity so it can sit on a cheap text-only profile
   (`MEMORYLAYER_LLM_ASSIGN_CONTRADICTION=cheap`). Chose the fused call over NLI because
   it needs no new service, where NLI requires building `POST /v1/nli` and shipping a
   model that does not exist today. Do NOT move `extraction` to a text-only profile —
   it is the multimodal document-page path.
6. **Supersession at recall** — ✅ **built** (`b5c9238`), ⚠ **measured, and it does not
   pay off**. Ships `off`. See "The end-to-end verdict" below.

Candidate 2 is dropped for OSS and handed to the enterprise dedup tier as a possible cost
optimization.

## Related

- `DESIGN_contradiction_association_unification.md` — owns Candidate 4's architecture;
  this doc supplies its quality floor. Candidate 1 is that doc's stated non-goal.
- Typed-edge eval harness: `benchmarks/assoc_typed_edge_eval.py`.
- Moat eval verdict: graph/entity moat was a wash on LoCoMo QA — motivating the
  "different lever" argument in Candidate 1.

## Non-goals

- Community detection / label propagation (Graphiti's `update_communities`): large
  recurring cost, no eval reason to want it yet.
- Per-edge LLM resolution: Graphiti's weakest scaling point; our batched classify is
  already better.
- Combined node+edge extraction: bulk-path-only and opt-in even upstream.


---

## The end-to-end verdict

The per-pair metric could not answer whether any of this helps, so the feature was scored
the way it will actually be used: ingest each LongMemEval `knowledge-update` item into its
own store, run the fused detector over every memory in chronological order, retrieve top-10
for the question under each policy, generate an answer from that context, and have an LLM
judge it against the gold answer. 78/78 items scored across all three arms; 56 had at least
one memory the detector marked superseded.

| arm | QA accuracy | vs `off` | McNemar p |
|---|---|---|---|
| `off` | **74.4%** (58/78) | — | — |
| `demote` | **69.2%** (54/78) | fixed 4, broke 8 | 0.39 |
| `exclude` | **76.9%** (60/78) | fixed 6, broke 4 | 0.75 |

**No arm is a statistically significant improvement.** The ordering `exclude` > `off` >
`demote` is suggestive, but at n=78 with 10 and 12 discordant pairs it is indistinguishable
from noise. `exclude` vs `demote` is the largest gap (8 vs 2 discordant, p=0.11) and still
does not clear 0.05.

**`demote` measured worst, which inverts the design assumption.** It was chosen as the
cautious default on the reasoning that a false positive costs only rank, where `exclude`
costs the answer. The mechanism turns out to run the other way: demoting leaves the stale
memory *in the retrieved context*, merely lower, so the answering model still sees both the
old and the new value — you pay the ranking distortion without removing the thing that
confuses it. `exclude` at least takes the stale fact out of the window. The config comment
has been corrected accordingly; if this is ever enabled, prefer `exclude`.

### What this settles

Supersession-at-recall was the most attractive idea in the original review — the one
mechanism by which a graph plausibly earns its keep after our moat eval found association
edges a wash. Built and measured on the benchmark designed to test it, **it does not
demonstrably improve answers.** It ships `off`, and the honest summary of the whole
Graphiti thread is:

- the **cost** work paid off and is live (tier skip filter, 44–57% fewer calls)
- the **correctness** work paid off and is live (the negation detector's substring bug)
- the **quality/graph** work does not, so far, show a user-visible win — consistent with,
  and now more specific than, the earlier moat verdict

### Caveats before treating this as final

- **Underpowered.** n=78 is the entire `knowledge-update` set; a real effect below ~10pp
  is not detectable here. A larger corpus (the full LongMemEval haystack rather than the
  oracle subset) would tighten it.
- **Retrieval slice, not full `recall()`.** The arm exercises
  `search_memories -> apply_supersession -> top-10`, deliberately isolating the change from
  hybrid fusion, reranking and boosts. Interaction effects with those are untested.
- **Detector under-detection.** 13 detection calls truncated at 5000 tokens and were
  excluded rather than scored as "no contradiction", so the `demote`/`exclude` arms saw
  slightly fewer supersessions than a production run would.
- **Judge is the same model family as the answerer**, which is standard for this benchmark
  but not independent.