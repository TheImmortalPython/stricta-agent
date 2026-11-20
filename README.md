<p align="right">
  <img src="assets/TIP_Logo_Square.jpg" alt="TheImmortalPython Labs Logo" width="30" />
</p>
<br>

<p align="center">
  <img src="assets/Stricta_Banner.jpg" alt="Stricta Banner" width="100%">
</p>
<br>
<br>


<div align="center">

<p><h1>Stricta: Deterministic Agent Orchestration Runtime</h1></p>

A deterministic, phase-gated agent orchestration engine built on Paninian Karaka task modeling, structural logit masking, shadow verification, single-attempt rollback recovery, and dual-backend ledger persistence (JSONL and SQLite).

---

<p><em>Developed by Joshua St. Germain | TheImmortalPython Labs</em></p>

---

<p>
<img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python Version">
<br>
<br>
<a href="https://github.com/TheImmortalPython/immortal-inference/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-AGPLv3-blue.svg" alt="License: AGPLv3"></a>
<a href="https://github.com/TheImmortalPython/immortal-inference/blob/main/LICENSE-COMMERCIAL.md"><img src="https://img.shields.io/badge/license-Commercial-green.svg" alt="License: Commercial"></a>
</p>
  
Dual-licensed under **AGPLv3** and **Commercial Terms**.
  
</div>

---

## Overview

`Stricta` structures multi-step agent execution around explicit dependency graphs and deterministic lifecycle phases (`INIT` → `TARGET` → `BOUNDARY` → `COMPLETE`) rather than unconstrained conversational loops. Each task is modeled as a `KarakaTaskNode` with structured semantic roles (`karta`, `karma`, `karana`, `sampradana`, `apadana`, `adhikarana`) and explicit upstream dependencies.

During execution, `StrictaAgentRunner` pairs local Hugging Face model inference (`HuggingFacePrimaryGenerator`) with structural token constraints (`PaninianKarakaLogitsProcessorFactory`), validates proposed transitions through `ShadowCritic`, recovers from single-attempt transient failures via rollback, and records accepted steps across JSONL and SQLite persistence backends.

### Standard Agent Loops vs. Deterministic Control Planes

| Feature | Standard Conversational Agent | Deterministic Control Plane |
| :--- | :--- | :--- |
| **Execution Flow** | Free-form prompt loops | Explicit DAGs & phase gates |
| **Output Enforcement** | Prompt engineering & retries | Structural logit masking |
| **Error Recovery** | Re-prompting or full restart | Isolated single-step snapshot rollback |
| **Auditability** | Unstructured chat logs | Replayable, parity-checked state ledgers |

## What You Can Make With Stricta

Built on `Stricta`'s core orchestration, masking, and persistence primitives, you can build:

* **Deterministic Multi-Step Task & Workflow Engines:** Construct structured execution pipelines where every stage is defined by explicit Karaka semantic slots and dependency links, holding downstream nodes until upstream prerequisites reach `VERIFIED` and `COMPLETE`.
* **Constrained Local LLM Execution Pipelines:** Build local Hugging Face inference workflows that apply structural logit masking and identifier boosting during token generation rather than relying solely on free-form prompting.
* **Resilient Join & Convergence Orchestrators:** Coordinate multi-branch graphs—including deeper trees and multi-convergence topologies—that synchronize parallel or sequential branches and recover cleanly from a single transient failure at a join node without corrupting completed upstream work.
* **Auditable & Replayable Ledger-Backed Agents:** Create execution harnesses that persist accepted steps to JSONL or SQLite ledgers, query historical state transitions, and verify cross-backend parity with `assert_ledgers_match`.
* **Self-Contained Local or Browser Runtime Harnesses:** Assemble lightweight local or browser-hosted agent loops around `StrictaAgentRunner` and `ShadowCritic` where candidate actions are evaluated against confidence and state-change checks before committing state transitions.
* **Custom Domain & Role-Driven State Machines:** Model specialized task lifecycles tailored to custom `karta`, `karma`, `karana`, `sampradana`, `apadana`, and `adhikarana` assignments for controlled, repeatable automation.

## Architecture & Repository Layout

```text
├── stricta/
│   ├── __init__.py                   # Package exports
│   ├── critic.py                     # ShadowCritic proposal validation & verification checks
│   ├── drivers.py                    # Execution drivers and ledger backends (JSONL & SQLite)
│   ├── generator.py                  # HuggingFacePrimaryGenerator integration
│   ├── logits.py                     # PaninianKarakaLogitsProcessorFactory structural token masking
│   ├── roles.py                      # Phase progression and Karaka role lifecycle logic
│   ├── runner.py                     # StrictaAgentRunner DAG scheduler, gating, and rollback recovery
│   ├── schemas.py                    # KarakaTaskNode and core state data structures
│   └── tools.py                      # Tool execution wrappers and ledger parity utilities
├── assets/
│   ├── Stricta_Banner.jpg            # Repository banner image
│   └── TIP_Logo_Square.jpg           # TheImmortalPython Labs logo
├── .gitignore                        # Git ignore rules for local ledgers and build artifacts
├── CONTRIBUTING.md                   # Solo-maintainer guidelines, no-PR policy, and issue expectations
├── LICENSE                           # GNU Affero General Public License v3.0 (AGPL-3.0)
├── LICENSE-COMMERCIAL.md             # Self-service commercial IP acquisition license terms
├── pyproject.toml                    # Package metadata and build configuration
├── smoke_test_deeper_tree.py         # Verified 8-node deeper-tree & rollback test
├── smoke_test_multi_convergence.py   # Verified 8-node multi-convergence & join-rollback test
└── smoke_test_six_node.py            # Verified 6-node DAG baseline test

```

## Installation

`Stricta` is installed directly from source (no live PyPI package is published yet):

```bash
git clone https://github.com/TheImmortalPython/stricta-agent.git
cd stricta-agent
pip install -e .

```

## Verified Smoke Tests

The repository includes standalone smoke tests that validate dependency gating, phase progression, single-attempt transient rollback recovery, and JSONL/SQLite ledger parity without modifying the core `stricta/` modules:

* **6-Node Baseline (`smoke_test_six_node.py`):** Validates multi-branch dependency gating and dual-backend ledger parity across a 6-node graph.
* **8-Node Deeper Tree (`smoke_test_deeper_tree.py`):** Validates deeper sequential dependency chains (`NODE-A` through `NODE-H`) converging at `NODE-H`, including single-attempt rollback recovery on `NODE-G`.
* **8-Node Multi-Convergence (`smoke_test_multi_convergence.py`):** Validates multiple convergence points (`NODE-G` and `NODE-H`) and single-attempt rollback recovery at a join node (`NODE-G`) while preserving identical JSONL and SQLite ledger entries (`matched: true`).

Run any of the standalone smoke tests from the repository root:

```
python smoke_test_six_node.py
```
```
python smoke_test_deeper_tree.py
```
```
python smoke_test_multi_convergence.py
```

## Solo Maintainer & Contributions

This project is 100% solo-authored and maintained by Joshua St. Germain (TheImmortalPython Labs). Pull requests are not accepted. Please review `CONTRIBUTING.md` before opening an issue.

## Licensing

`Stricta` is dual-licensed:

1. **Open Source (`LICENSE`):** Licensed under the GNU Affero General Public License v3.0 (`AGPL-3.0`).
2. **Commercial (`LICENSE-COMMERCIAL.md`):** Commercial licenses are available strictly on an as-is basis as self-service intellectual property acquisitions, excluding ongoing consulting, engineering support desks, SLAs, and embedded hardware deployment rights.

---

> **Founder & Principal AI Architect:** Joshua St. Germain  
> **Enterprise Governance & IP Incubator:** TheImmortalPython Labs ([theimmortalpythonlabs@proton.me](mailto:theimmortalpythonlabs@proton.me))  
> **Repositories:** [`immortal-inference`](https://github.com/TheImmortalPython/immortal-inference) | `stricta-agent` | [`the-nsai-guide`](https://github.com/TheImmortalPython/the-nsai-guide)

---

*© TheImmortalPython Labs. All rights reserved.*
