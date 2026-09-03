# NOTICE — third-party components & licensing

This repository's **code** is MIT-licensed (see [`LICENSE`](LICENSE)). But this
project *runs* a model and weights that are **not** yours to use freely. Read this
before you expose the gateway to anyone.

## ⚠️ The most important line

**MiniMax-H3 is NOT open source in the OSI sense.** It ships under the **MiniMax H3
Community License** (a custom license). Serving it to other people is **commercial
use**, and the license currently limits that to specific regions (per the model card,
**US / EU / UK / KR**) and may require a signed application form. **Clear this
compliance step BEFORE opening the endpoint to users** — it is outside the code
but unavoidable. See [`docs/PLAN.md`](docs/PLAN.md) §4.3. The optional image path uses
**FLUX.2-dev**, which has its own separate license (below) — clear that too if you enable images.

## Components this project pulls in

| Component | What it is | License | Where it comes from |
|---|---|---|---|
| **MiniMaxAI/MiniMax-H3** | The 33B image-text-to-video model weights (~196GB root modular layout served; the repo also ships redundant FL2VA/Ref2VA copies we skip) | **MiniMax H3 Community License** (custom, non-OSI; commercial + regional restrictions) | Hugging Face `MiniMaxAI/MiniMax-H3` |
| **lightx2v/Minimax-h3-Turbo** | Step-distillation Turbo LoRA (8-step / 4-step) | Apache-2.0 | Hugging Face `lightx2v/Minimax-h3-Turbo` |
| **FLUX.2-dev** (optional, image path) | The text-to-image model weights | **FLUX.2-dev Non-Commercial License** (custom, non-OSI — check its terms before any commercial use) | Hugging Face `black-forest-labs/FLUX.2-dev` |
| **diffusers** | Inference framework (modular pipeline + LoRA loader) | Apache-2.0 | `huggingface/diffusers` |
| **PyTorch / transformers / peft / boto3 / FastAPI** | Runtime dependencies | BSD-3 / Apache-2.0 / MIT (each own license) | PyPI |

The weights and the Turbo LoRA are **not** included in this repository (see
`.gitignore`) — they are downloaded from Hugging Face at deploy time by
[`serving/download_model.sh`](serving/download_model.sh). Your use of them is
governed by their respective licenses above, not by this repo's MIT license.

## Your responsibilities as a deployer

1. **Confirm you are permitted to serve MiniMax-H3** in your region / for your use
   case under the MiniMax H3 Community License before exposing the API.
2. Keep the result bucket private and the GPU box zero-inbound (the IaC does this by
   default — don't loosen it).
3. Comply with the licenses of every component in the table above.
