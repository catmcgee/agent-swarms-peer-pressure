# Failed structured-recognition canary

This immutable result bundle is the second remote attempt on 2026-08-28. The run used Qwen3.5-9B
at revision `ef3d031a90d340a92d71f83ec17d054e100ce713` with a 1,024-token output cap.

The paired authorized controls passed: 34 of 36 completed their required critical action
(94.4%). The worker then ran the pre-registered 24-row impossible-task stop/proceed canary and
aborted before the other 192 discovery rows because only 2 of 24 rows contained a structured
unauthorized assessment. One post-recognition response contained truncated tool markup.

A subsequent transcript audit found:

- stop/proceed recognition responses were byte-identical within each task because the board is
  shown later;
- 22 rows answered the authority result in prose instead of calling the assessment tool;
- the recognition parser had zero malformed responses and missed no assessment markup;
- the single malformed response occurred after peer exposure and attempted an equivalent
  report-writing path that the original task wrapper had not marked critical;
- some authorized controls called task tools during recognition even though those tools were not
  advertised in that phase.

Accordingly, the recorded 8.3% is structured-assessment compliance, not a retrospective semantic
recognition estimate. The run remains a failed protocol pilot. Its should-abstain rows are not
reused in discovery, and the raw traces are preserved without post-hoc score replacement.
