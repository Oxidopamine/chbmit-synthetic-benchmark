# Positioning & Citations (v5.3 §3, §7)

How this benchmark relates to prior work, and the citations to verify before submission.
The novelty rests on the **harm characterization** and the **generator × detector map**, not
on the gate. State the deltas honestly.

## Trust-gated augmentation (TGA) — the method we ADAPT

The fail-closed trust gate is **not our invention**. It was introduced as trust-gated
augmentation (TGA):

> Choi, D.; Yip, C.; Choi, A.; Park, J. (2026). "Trust-gated synthetic EEG augmentation
> reduces performance drops when generalizing to new patients." *npj Digital Medicine*
> **9(1), art. 634**, DOI `10.1038/s41746-026-02778-0`, 25 May 2026, PMID 42185473.

This is the **version of record**, confirmed at Crossref and Europe PMC
(`the verification record` §2). The bioRxiv preprint
`10.64898/2026.01.26.701638` is real but superseded; its `published: NA` field is stale and
is **not** evidence of preprint status. Cite the npj article, not the preprint.

What TGA established (state plainly, do not minimize):
- The core idea: score synthetic windows with a real-data teacher for label consistency and
  confidence, admit only those above a confidence quantile `q`, and inject synthetic data only
  if validation performance beats real-only by a margin, else revert (fail-closed).
- That ungated synthetic augmentation can silently harm subject-disjoint EEG generalization,
  with harm concentrated in the tail of paired performance deltas.
- A covariance-manifold audit showing synthetic windows can lie far off the real manifold.
- Evidence on chronic-pain resting-state EEG and motor-imagery BCI, using AUROC and harm rate.

What is genuinely new here (the explicit delta — for related work):
- **Domain:** seizure detection — a rare-event, extreme-imbalance, high-stakes setting, unlike
  the balanced discriminative tasks (chronic pain, motor imagery) TGA evaluated.
- **Metric:** harm and admission are defined at the **event level**. TGA's fail-closed rule
  uses a validation AUROC margin; ours uses validation **event-F1** improvement under an
  explicit **FP/24h** safety constraint — a clinically meaningful criterion AUROC can't express.
- **Object:** we characterize the seizure-specific harm itself (false-alarm inflation,
  event-sensitivity loss on held-out patients), novel independent of any gating.
- **Breadth:** generator × detector interaction under one leakage-safe harness.

Safe positioning sentence:
> Recent subject-shift work shows ungated synthetic EEG augmentation can silently harm
> cross-subject generalization and proposes a fail-closed trust gate to control it. We ask
> whether the same harm appears in patient-independent seizure detection at the clinically
> relevant event level, and whether an event-level reformulation of fail-closed gating
> controls it without discarding genuine benefit.

## GP-EEG (strong seizure generator) — motivation, not competitor

GP-EEG is a strong seizure-specific synthetic EEG generator (Gaussian-process modeling plus a
domain-adaptation VAE-style stage), evaluated on epileptic EEG including CHB-MIT and Siena. It
**motivates** our downstream question: now that strong seizure generators exist, are their
outputs safe to train on under patient-independent event-level evaluation? We do not claim
generator superiority; GP-EEG is an optional patient-specific provider arm if code/samples are
available. **[VERIFY citation before use.]**

## Geometry-preserving / Riemannian VAE — supporting context only

**[VERIFY THIS CITATION EXISTS AND MATCHES BEFORE USE.]** If it is BCI/motor-imagery based,
cite it only as evidence that synthetic-EEG utility is representation- and
classifier-dependent. Do **not** present it as seizure-detection evidence.

## Pre-submission verification checklist (v5.3 §7)

- [ ] Confirm the TGA preprint's authors, title, venue, status; cite precisely; note preprint
      status if still unreviewed at submission time.
- [ ] Search for concurrent **seizure-specific** trust-gated / harm-aware augmentation work
      (the TGA idea is public — a seizure follow-up from another group is possible). Position
      accordingly.
- [ ] Verify the geometry-preserving / Riemannian VAE citation exists and is characterized
      correctly (do not cite BCI results as seizure evidence).
- [ ] Ensure "adapted" / "reformulated" language for the gate appears consistently (no
      "proposed" / "novel").
- [ ] Ensure the abstract and intro lead with the **harm** finding, gate introduced second.
