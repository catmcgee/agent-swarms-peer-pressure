# Third-party material

SwarmStop does not vendor benchmark source or data. The fetch script downloads pinned copies into an ignored directory.

## AgentAbstain

- Source: https://github.com/AntiQuality/agentabstain
- Pinned source commit: `f581249704b26804e28a39e37396f1be00b71a4d`
- Dataset: https://huggingface.co/datasets/antiquality/agentabstain
- Code license: MIT
- Dataset license: CC BY 4.0
- Citation: Liu et al., *AgentAbstain: Do LLM Agents Know When Not to Act?*, arXiv:2607.10059, 2026.

## AuthorityBench

- Source: https://github.com/yazcaleb/can-is-not-may
- Pinned source commit: `da8a0ce8c779da067ccf6caa5dd311c1ff443960`
- Code license: MIT
- Paper source license: CC BY 4.0
- Citation: Celebi, *Can Is Not May: Authority Models for Governable AI Agents*, 2026.

## Qwen 3.5 and workspace lenses

- Model: https://huggingface.co/Qwen/Qwen3.5-9B
- Pinned model and tokenizer content commit: `ef3d031a90d340a92d71f83ec17d054e100ce713`
- Model license: Apache 2.0
- Lens artifacts: https://huggingface.co/camilablank/workspace-lenses
- Pinned lens repository commit: `d740106d1e0f95456dc8718fba2895e9c8ffd6ef`
- Lens artifact license: MIT
- J-lens SHA-256: `e9396f37eec4c031462b24559b0360027ac45d5944628214301b8e54883b58f3`
- R-lens SHA-256: `76c7372c52b453958e064e4d366fa868cb3733460728236cf2e9fa6bdb632d52`
- Reference implementation: https://github.com/anthropics/jacobian-lens
- Pinned reference implementation commit: `581d398613e5602a5af361e1c34d3a92ea82ba8e`
- Reference implementation license: Apache 2.0

The lens archives identify the model but do not record its repository revision. The pinned model commit is the last content upload before the lens release; the only subsequent repository change before this project pinned it was to `README.md`. This supports content equivalence but is not a cryptographic record of the authors' local model checkout.

The adapter modules are clean interfaces written for SwarmStop. If upstream code is later copied or modified, preserve the relevant copyright and license notice next to that material.
