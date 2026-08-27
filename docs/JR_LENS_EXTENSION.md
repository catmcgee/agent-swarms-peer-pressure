# Exploratory mechanistic extension: J-space and R-space

## Status and purpose

This is a post-behavioral, exploratory Stage C. It does not alter the confirmatory SwarmStop design. Its purpose is to distinguish two mechanisms that produce the same observed reversal: peer contact may erase the model's representation that an action is unauthorized, or that representation may survive while a downstream completion or peer-following representation wins control of the action.

The extension requires an open-weight model, exact access to residual-stream activations, and compatible matched Jacobian-lens (J-lens) and relevance-propagation lens (R-lens) artifacts. Closed API models remain in the behavioral study only.

## Mechanistic hypotheses

- **M1 — boundary erosion:** after a proceed board, authorization and stopping concepts weaken in the R-lens readout relative to a matched stop board.
- **M2 — downstream override:** authorization and stopping concepts remain stable, while peer-consensus, completion, delegation, or action concepts strengthen at later layers and predict reversal.
- **M3 — verbalizable-space insufficiency:** a behavioral peer effect survives, but neither matched lens yields held-out or causal evidence for M1 or M2.

J-lens and R-lens disagreement is diagnostic rather than a reason to select the preferred story. In particular, an effect found only by J-lens in early layers will not be treated as evidence of representation loss because early J-lens readouts are the setting in which gradient error is expected to be largest.

## Model and sample gate

Use the smallest open-weight model that satisfies all of the following:

1. it completes the paired should-act controls;
2. it shows a measurable proceed-versus-stop reversal in the behavioral pilot;
3. a recipe-matched J-lens/R-lens pair is available or reproducibly fit; and
4. the exact checkpoint and tokenizer can be run without inference substitutions that invalidate the lens.

The first economical candidate is a dense model in the 9B class with published matched artifacts. Do not default to the smallest 4B candidate merely for cost: the R-lens authors report no advantage for their smallest dense model. A 27B replication is justified only after the discovery model passes the behavioral, readout, and intervention gates.

The discovery subset fixes personal investment and the eight-round budget, then crosses:

- 12 task families;
- feasible versus impossible authorized path;
- no board, peers stop, or peers proceed; and
- 3 target/board seeds.

This yields 216 trajectories on one model. The sample is for effect localization and engineering feasibility, not a new powered confirmatory test.

## Anchors and recorded activations

Record residual-stream activations at stable, prompt-aligned anchor positions:

- **T0:** immediately before the authority result;
- **T1:** after the private unauthorized assessment and before peer exposure;
- **T2:** after the final peer-board message and before the next action; and
- **T3:** at the first structured post-board action decision.

Capture every layer at these anchors, plus the ordinary output logits and the eventual safe or unsafe action. The structured action slot must be identical across conditions. Do not use hidden chain-of-thought tokens as data.

Store anchor activations in reduced precision with trial, layer, position, checkpoint, tokenizer, and lens provenance. Do not retain full-sequence activations unless a prespecified localization check requires them.

## Readout plan

Before inspecting outcomes, freeze a small token-level concept registry for:

- authorization, permission, denial, and stopping;
- peer identity, consensus, trust, and social evidence;
- completion, urgency, progress, and sunk cost;
- direct action, delegation, and asking the user.

Report the complete registry and tokenizer forms. Score token sets with both matched lenses at each layer and anchor. Top-k words are qualitative illustrations only; the quantitative objects are preregistered concept-set scores and sparse nonnegative lens coordinates. Because J-space and R-space are overcomplete sparse frames rather than ordinary low-dimensional bases, report sensitivity to the sparsity value instead of treating either as a single fixed Euclidean subspace.

Primary exploratory comparisons are:

1. the T2 minus T1 change under proceed versus stop boards;
2. the layer at which each concept first becomes reliably decodable;
3. token-matched J/R direction agreement and CKA of sparse coefficient representations;
4. cross-validated prediction of stop-to-act reversal from T2 minus T1 lens features; and
5. generalization to held-out task families and board realizations.

Ordinary output logits, logit-lens directions, and simple residual-stream mean-difference probes are baselines. Any classifier split must be by task family, not by trajectory, to prevent near-duplicate task leakage.

## Causal validation

Descriptive readouts do not establish mechanism. After localizing a small layer band on discovery data, test it on held-out tasks using paired interventions:

- patch selected lens coordinates from a matched peer-stop run into a peer-proceed run and perform the reverse patch;
- ablate the authorization/stopping directions and the peer-consensus/completion directions separately;
- compare matched J-lens, R-lens, logit-lens, random-direction, non-lens-remainder, wrong-layer, and wrong-position controls; and
- norm-match every perturbation and report capability degradation on paired should-act tasks.

The causal outcome is the change in probability of an unauthorized critical action at the structured decision slot, followed by sampled continuation checks. Direction and layer selection must be frozen before the held-out interventions.

## Interpretation rules

- Call the result **boundary erosion** only if the post-board boundary signal falls on held-out tasks and restoring the stop-condition coordinates causally increases stopping.
- Call it **downstream override** only if the boundary signal remains detectable, a competing late signal predicts reversal, and perturbing that signal causally reduces unsafe action without broad capability collapse.
- If readouts are clear but interventions fail, describe them as correlates.
- If J- and R-lens conclusions diverge, privilege neither automatically; compare held-out prediction and matched interventions.
- If no lens result survives held-out tests, report that the behavioral effect was not localized in the measured verbalizable spaces.

These claims are model- and layer-specific. Raw vectors must not be compared across checkpoints as though their coordinates were shared.

## Cost-minimizing execution

Use published lens artifacts rather than refitting when checkpoint provenance matches. Generate each trajectory once, cache only the four anchor activations, and compute descriptive lens readouts offline. Run causal continuations only for the selected layer band and held-out subset. A single low-batch 9B run should fit on a commodity 24 GB accelerator if the model and lens matrices are streamed carefully; a 48 GB accelerator is the safer operational default. Benchmark memory and throughput before renting a long-lived instance.

Stop Stage C before scaling if any gate fails: no behavioral reversal, incompatible artifacts, no held-out readout improvement over baselines, or no causal effect beyond matched controls.

## Implemented data contract

`configs/jr_lens.yaml` encodes the 216-trajectory discovery matrix and pins the model, tokenizer, lens artifact revision, file sizes, and file hashes. The lens files identify the model but omit its revision. The selected model commit is the last content upload before the lens artifact was published; subsequent model-repository changes affected only the README. This is strong content-equivalence evidence, but the original fitting revision cannot be recovered cryptographically from the artifact.

`swarmstop lens-provenance` verifies the remote hashes and sizes, extracts only the small metadata member from each approximately 1.04 GB archive using byte-range requests, and checks that the pair agrees on model, dimensions, source layers, corpus, sample count, target layer, and fitting settings. It also checks the pinned model configuration against the lens dimensions. No tensor storage is downloaded by this command.

Each JSONL readout record is keyed by trial, task family, condition, anchor, and layer. It must include exact model/tokenizer provenance, optional activation storage metadata, and complete concept scores for every included lens. `swarmstop lens-compare` computes the within-trial T2-minus-T1 change first and then the descriptive proceed-minus-stop contrast separately for every layer, lens, and frozen concept. It performs no significance testing and is not a substitute for the held-out or causal analyses above.

## Sources

- R-lens post: https://www.alignmentforum.org/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers
- J-lens and J-space paper: https://transformer-circuits.pub/2026/workspace/index.html
- Published matched artifacts: https://huggingface.co/camilablank/workspace-lenses
