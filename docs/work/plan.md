# NullGraph — Program map

**What this file is:** the one place that says *where the work is going* and *how each piece of
work serves that*. Read it before starting any step. It changes only when the user decides a
change; each change is dated in "Scope changes" at the bottom. Detailed designs live in the
`*-design.md` files linked below; this file never repeats their contents.

**Not this file:** the engine's versioned module roadmap is [`../roadmap.md`](../roadmap.md).
Today's position is [`now.md`](./now.md).

---

## 1. The goal

NullGraph is a WebGPU engine. The current program (Cycle C) adds **batched agents**: thousands of
agents simulated, drawn and steered entirely on the user's GPU, first by hand-written rules, then
by small neural networks. It is the showpiece of the agreed AI plan: *AI outputs that light,
shadow and populate a world, entirely on the user's GPU* (no server cost per user; private data
stays on the device).

Every step below must make that more real for a user of the library. A step that does not is
drift; raise it with the user instead of doing it.

## 2. How the work is named

From largest to smallest:

| Level | Example | Means |
|---|---|---|
| **Cycle** | A, B, C | A program of several weeks with its own goal. A: verifiable, agent-operable engine. B: picking and text labels. C: batched agents. |
| **Milestone** | C0, C1, C2, Phase 2 | A stopping point inside Cycle C. C0: measure first (spikes). C1: agents with rules, no ML. C2: neural brains. Phase 2: evolution, eyes, obstacles (after C2). |
| **Step** | step 6 | A numbered deliverable in the parent plan [`../agents-design.md`](../agents-design.md) §13. Steps 1–10 run across C1 and C2. |
| **Phase** | phase 6.1 | One commit-sized piece of a step, defined in that step's design doc (`6.1` = step 6, phase 1). |
| **Spike** | S3 | A measurement script that decides a design before code (C0). |
| **Test IDs** | U10, A4 | `U` = headless unit check, `A` = real-GPU check. Defined in the parent plan §10–11. |

Note the clash: **"Phase 2"** (capital, a milestone after C2) is not a step's phase like 6.2.

## 3. Cycle C — every step, what it is for, and where it stands

| Step | Milestone | Delivers | Serves the goal by | Status |
|---|---|---|---|---|
| S3 | C0 | Cost of the network as a WGSL kernel | Proving the built-in brain is cheap enough to ship | ✅ 2026-10-11 |
| S1, S2, S4, S5 | C0 | ONNX Runtime (ORT) cost, latency, startup, device sharing | Deciding step 9's defaults | Not run; due before step 9 |
| 1 | C1 | Agent record, layouts, CPU reference math | The data model everything else uses | ✅ `7f4cdb5` |
| 2 | C1 | GPU spatial grid, neighbour query, debug views | Agents can find neighbours on the GPU | ✅ `4ebfe99` |
| 3 | C1 | `AgentSystem`: sense → think (rules) → act on the GPU, arrows demo | A whole population runs with no CPU per agent | ✅ phases 3.0–3.5 ([design](../agents-gpu-loop-design.md)) |
| 4 | C1 | Instanced darts, picking, labels, inspector, API guide | Agents are visible, clickable and explainable | ✅ phases 4.1–4.5 |
| 5 | C1 | Scaling limits and bottleneck study ([design](../agents-demo-design.md)) | Honest limits for users | **Deferred** by the user (2026-10-12); the stress test covers it for now |
| 6 | C2 | `MlpBrain`: a small network runs as the think stage ([design](../agents-mlp-design.md)) | Neural policies run at rule-brain speed | ✅ 6.0 `e345921`, 6.1 `dbaf884`, 6.2 `b3424b1`, 6.3 (guide, cost, size) |
| 7 | C2 | Record the rule brain → train a copy → export (`tools/agents`) | **First network that has actually learned Herd & Wolves** | Not started; needs the species decision (§5) |
| 8 | C2 | `MlpBrain.fromOnnx` | Load models trained anywhere | Not started |
| 9 | C2 | `null-graph/inference` + `OrtBrain` (ORT optional peer) | Larger models; shared with the photo-booth and relight ideas | Designed, not built; spikes first |
| 10 | C2 | Demo: switch Rules / Network / ORT live with timings | The showable result | Not started |

**C1 is a product on its own:** crowds, herds and RTS units with no ML.

### Other open work (outside Cycle C)

- Agent kit follow-up: guidance fixes, a pinned-tarball trial, then release/tag (handoff "Agent kit").
- Verify, then fix, the `uvMinA` shadow-atlas suspect with a GPU test.
- CI on a pinned Linux + lavapipe environment, only if golden images are wanted.

## 4. Horizon goals — how far each one is

| Goal | Needs | Distance |
|---|---|---|
| Batched neural agents: one trained brain, many agents, 60 fps, inference measured | Steps 6 → 7 (→ 10 to show it) | **Near:** 2–3 steps |
| Event photo booth; relight your photo | Step 9's `null-graph/inference`, spikes S1–S5, a segmentation/depth model | Separate track; step 9 is the shared base |
| Train on a server, play in the browser | Step 7, Phase 2 evolution, a first run on a Linux GPU | Medium-far |
| Live ecosystem (evolution), venue evacuation, agents that see | Phase 2: genomes, obstacles and flow fields, ray sensors | Far: after C2 |
| Synthetic data factory, auto-cinematographer, self-narrating sim | Not on this path | Not started |

Full descriptions: handoff "After Cycle B — the ONNX/AI product plan" and "Far horizon".

## 5. Decisions waiting on the user

1. **Species input for step 7:** one network for prey and wolves (species as an input) or one
   network per species. Must be settled before recording training data.
2. **Bundle allowance:** `agents.js` is 17,283 B min+gzip after step 6, with 2,212 B left of C2's
   +4,000 B allowance. Step 8's ONNX loader must fit, or the allowance changes.
3. **Brain swap stall:** `setBrain` compiles a pipeline synchronously (about 1.1 s under
   Node/Dawn on NVIDIA, 76 ms in Chrome on AMD). Before step 10's live Rules/Network switch:
   accept the one-time hitch, or approve asynchronous stage creation (a core change).
   **Deferred by the user to step 9 (2026-10-12).**

## 6. Constraints that hold for every step

- Design doc before code; build in phases; tests watched failing before they pass.
- Detect GPU capabilities; no polyfills or legacy paths.
- The npm package ships `dist/` only; agent docs live on GitHub.
- Commit only when the user asks; never push or publish without being asked.

## Scope changes (dated, decided with the user)

- 2026-10-11: steps 3–4 design accepted; C1 bundle line raised to 16 KB.
- 2026-10-12: step 5 rescoped from "20,000-agent demo, ≤ 2 ms" to a scaling study, then deferred;
  the user built a maximum-capacity stress test instead. Step 6 started.
- 2026-10-12: step 6 completed. The brain-swap stall decision (§5 item 3) deferred to step 9.
