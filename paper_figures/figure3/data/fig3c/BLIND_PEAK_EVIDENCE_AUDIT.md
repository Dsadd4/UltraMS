# Figure 3c candidate-blind peak-evidence audit

Date: 2026-08-22

## Question

Can the m/z 316.020386 UltraMS peak embedding help select the benchmark
structure without using the target or fixed Pleurostimin structures to retain
or discard retrieved annotations?

## Blindness contract

The protocols in `stage10_blind_candidate_scoring/protocol.json` and
`stage11_blind_all_peak_scoring/protocol.json` were frozen before their
respective executions. Evidence retention uses only the retrieved reference
peak, its own reference-parent structure, MAGMa assignment validity, the 0.55
cosine gate and the frozen 10 ppm mass window. The following Figure 3c fields
are forbidden during evidence retention: target identity, fixed-isomer
identity, pairwise relation, compatibility with either candidate, compatibility
with any of the 60 candidates, and the old candidate-conditioned
`strict_retained_substructure` field.

Candidate SMILES are introduced only after the blind evidence table is frozen.
All 60 connectivity-distinct candidates are then scored by an identical rule;
benchmark labels are merged only after the unlabeled score CSV has been written.

## m/z 316 result

Thirty-two parent-diverse neighbors passed the frozen retrieval gate. Eight
annotations passed the source-intrinsic structural-validity gate and collapsed
to four fragment-connectivity families. Under the preregistered primary rule
(sum of each compatible topology family's maximum cosine), the benchmark target
scored 0.732306 and occupied the exact-tie rank interval 16--20 of 60. Fixed
Pleurostimin scored 0 and occupied rank interval 26--60. All four prespecified
sensitivity rules also scored the target above Pleurostimin. Under the most
literal highest-valid-topology rule, the target belonged to an 11-candidate
rank-1 tie, while Pleurostimin remained unsupported.

Therefore m/z 316 is a reproducible pairwise exclusion signal for
Pleurostimin, but it is not a unique blind molecular identification signal.

## Candidate-blind top-topology selection

The registered `global_top_family_sum` sensitivity rule selects, for each
query peak, the source-valid topology with the highest peak-embedding cosine
before either candidate identity or candidate compatibility is read. Forty-nine
of the 54 query peaks had a source-valid top topology. After freezing those 49
selections, pairwise compatibility with the benchmark target and fixed
Pleurostimin was revealed: 43 topologies matched neither displayed candidate,
five were shared, one was target-only and none was Pleurostimin-only.

The sole discriminative top-ranked topology occurred at m/z 316.020386. Its
`YRRAGUMVDQQZIY` C15O8 connectivity was recovered from three spectra belonging
to three distinct reference parents (maximum cosine 0.732306), was compatible
with the benchmark target and was incompatible with Pleurostimin. Thus the
direction of this local decision is determined by the blind peak search rather
than by choosing a target-compatible topology after retrieval.

## All-54-peak result

Across all 54 measured peaks, 398 source-valid neighbor annotations collapsed
to 300 fragment-topology families on 49 peaks. The preregistered peak-balanced
primary score placed the target at rank interval 24--25 and Pleurostimin at
rank 32. The target scored above Pleurostimin under the primary rule and all
four sensitivity rules, but it was not global rank 1 under any registered rule.

The m/z 316 contribution was 0.732306 for the target and 0 for Pleurostimin. It
accounted for 74.31% of the full blind target-minus-Pleurostimin primary margin.
Removing this peak moved the target from rank interval 24--25 to 29--30 and
reduced the pairwise margin from 0.985507 to 0.253201.

## Claim decision

Supported:

> In a candidate-blind peak-evidence analysis, the m/z 316.020386 signal
> provided a reproducible local-topology constraint that supported the benchmark
> target and excluded fixed Pleurostimin, accounting for 74.3% of their blind
> pairwise score margin.

Not supported:

- m/z 316 alone uniquely identified the benchmark target among 60 candidates;
- the blind all-peak MAGMa-transfer score reproduced the UltraMS full-spectrum
  rank-1 result;
- the current candidate-conditioned 3/3 display is an independent structural
  annotation;
- computational fragments constitute authentic-standard, MS3 or NMR
  confirmation.

## Reproducible outputs

- `source_data/peak_embedding_cross_parent_20260823/stage10_blind_candidate_scoring/`
- `source_data/peak_embedding_cross_parent_20260823/stage11_blind_all_peak_scoring/`
- `source_data/peak_embedding_cross_parent_20260823/stage12_blind_topology_selection/`
- `../../scripts/fig3c/blind_score_peak_evidence.py`
- `../../scripts/fig3c/blind_score_all_peak_evidence.py`
- `../../scripts/fig3c/summarize_blind_topology_selection.py`
