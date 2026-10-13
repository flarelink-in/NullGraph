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
| 5 | C1 | Scaling limits and bottleneck study ([design](../agents-demo-design.md)) | Honest limits for users | **Deterministic neighbour lookups ✅ 2026-10-13** (phases 5.5–5.8, [ordered grid](../agents-ordered-grid-design.md)): reruns are bit-exact with overflow; crowded cells yield a window per cell and tick (the user revised decision 2 after the window per querying agent cost +15% to +26%); a step is within the 10% rule (worst case +9.6%); 5.7 `de87b3f`, 5.8 the guides. **The scaling study (phases 5.0–5.4) is still deferred, so step 5 is not complete** |
| 6 | C2 | `MlpBrain`: a small network runs as the think stage ([design](../agents-mlp-design.md)) | Neural policies run at rule-brain speed | ✅ 6.0 `e345921`, 6.1 `dbaf884`, 6.2 `b3424b1`, 6.3 (guide, cost, size) |
| 7 | C2 | Part 1: per-species brains ([design](../agents-species-brains-design.md), phases 7.1–7.5: weight-table rows, then per-species think passes). Part 2: record the rule brain → train a copy per species → export (`tools/agents`), from 7.6 | **First network that has actually learned Herd & Wolves** | Part 1: 7.1 ✅, 7.2 ✅ (measured; option B chosen), 7.3 ✅ (per-species passes, `681ccf4`), 7.4 ✅ (A22, `f19dbea`), 7.5 ✅ (the guide, `de14ff4`): **part 1 complete**. Part 2: design decided 2026-10-13 (`f1244cb`, [cloning](../agents-cloning-design.md), phases 7.6–7.10); 7.6 ✅ (the scenario, the label pass, the recorder, `c2608d7`); 7.7 ✅ (the trainer; round-0 students, `0d3fb9f`); 7.8 ✅ (the copy in the running simulation: catches within 1% of the rule without wander, no drift, no round of correction, `8afc6fa`); 7.9 ✅ (the two networks committed in `tools/agents/models/herd-wolves`, U26, A26, A8, the guide's "A trained network", the tools' README, `cd76a25`); 7.10 ✅ (the `.onnx` writer, U12, A27: ONNX Runtime within 4.5e-6 of `mlpForward`, `834b360`). **Part 2 is built and committed** |
| 8 | C2 | Reading ONNX: `mlpFromOnnx` in a new entry, `null-graph/inference` (the parent's `MlpBrain.fromOnnx`; [design](../agents-onnx-reader-design.md), phases 8.0–8.3) | Load models trained anywhere | Design decided 2026-10-14 (`a8778ce`; §5, item 8). 8.0 ✅ (`7829f4f`): 16 files from PyTorch, scikit-learn and `onnx.helper`, with what each maker computed. 8.1 ✅ (`8aa243a`): the entry and `mlpFromOnnx` (2,525 B; 13 foreign files read bit for bit, 3 refused as listed; 34 of 34 mutations caught). 8.2 ✅ (`874488d`; A28: the reader's `MlpBrain` and ONNX Runtime agree within 4.5e-6 on 17 files, on three host/GPU pairs). 8.3 ✅ (the commit after `874488d`; the guide `docs/api/inference.md` with a checked example, the README row, the parent's wording). **Every phase is built** |
| 9 | C2 | `OrtBrain` and the generic ONNX Runtime adapter, in `null-graph/inference` (ORT optional peer) | Larger models; shared with the photo-booth and relight ideas | Designed, not built; spikes first. The entry itself is created by step 8. To decide here: weights stored outside the `.onnx` file (§5, item 8) |
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

1. ~~**Species input for step 7**~~ **Decided 2026-10-12: one network per species** (option B),
   routed inside one think stage that switches on species (strategy 1), with each species kept in
   a contiguous index range. Refined after a check against the library's principles: same-shape
   networks are **weight-table rows** of one `MlpBrain` (built first), and a composite
   `SpeciesBrain` mixes kinds (second). No fallback brain. **The single-stage routing and the
   composite were withdrawn after 7.2's measurements; see item 4.** Designed in
   [`../agents-species-brains-design.md`](../agents-species-brains-design.md) (step 7, part 1).
2. ~~**Bundle allowance**~~ **Decided 2026-10-12 (the user left the number to Claude):** C2's
   allowance rises from +4,000 B to **+6,000 B** over the 15,495 B C1 baseline, so the whole
   entry may reach 21,495 B min+gzip. Expected spend: per-species routing ~1 KB, `fromOnnx`
   ~2 KB, plus margin. Rule-only apps must still tree-shake all of it (checked by `size.mjs`).
3. **Brain swap stall:** `setBrain` compiles a pipeline synchronously (about 1.1 s under
   Node/Dawn on NVIDIA, 76 ms in Chrome on AMD). Before step 10's live Rules/Network switch:
   accept the one-time hitch, or approve asynchronous stage creation (a core change).
   **Deferred by the user to step 9 (2026-10-12).**
4. ~~**Per-species mechanism**~~ **Decided 2026-10-12: option B**, after 7.2 measured per-lane
   routing in one stage at 4.6–5.6× a single network in Chrome: one think pass per brain, gated
   to its species (`setBrain(brain, { species })`). Rows were kept by the user's rule after
   measuring 8 and 16 species (rows win at 16 everywhere): **passes below 8 species, rows from 8.**
   The composite brain is withdrawn. Design: [`../agents-species-brains-design.md`](../agents-species-brains-design.md) §5–§6.
5. ~~**A rows brain for only some species**~~ **Decided 2026-10-13: keep `MlpBrain.bySpecies` as
   it is.** It takes one network per declared species, even when the rows brain serves only some
   of them; the others need a padding row of the same shape. The reasons:
   - relaxing it later breaks nothing;
   - Phase 2 rethinks rows anyway;
   - nothing planned needs it.

   Relaxing it would need the brain to declare its species, so that `setBrain` can refuse the rest.
   Recorded under "Deferred" in the design.
6. ~~**Step 5's deterministic neighbour lookups**~~ **Decided 2026-10-13** in
   [`../agents-ordered-grid-design.md`](../agents-ordered-grid-design.md) §10:
   - **Scope:** the fixed-order lookups only; the scaling study later.
   - **Crowded cells:** a rotating window, offset per cell and tick, deterministic. Decided
     before any teacher data is recorded. **Revised 2026-10-13:** first decided per querying
     agent; phase 5.7 measured that at +15% to +26% on a step, so the user chose a window per cell.
   - **Always on, as a single path.** The numbers come back to the user only if the sort adds more
     than about 10% to a step at 4,096 or 65,536 agents, on any host.
   - **Custom sensors reading the neighbour lists:** a small contract later, when a demo needs it;
     not in step 5. Phase 5.8 documents today's limits and fixes the parent's claim.

7. ~~**Step 7, part 2: record → train → export**~~ **Decided 2026-10-13** in
   [`../agents-cloning-design.md`](../agents-cloning-design.md) §14:
   - **The network copies** the rule's decision without its random wander and index coin-flips.
   - **No observation for the wolf's rest timer:** a resting wolf is frozen; its rows are dropped.
   - **The teacher lives in this repository,** under `tools/agents/`.
   - **Files:** `.mlp.json` now, with the two trained Herd & Wolves files committed; `.onnx` in
     the last phase, 7.10, which stays in step 7.
   - **The bars:** the parent's A8; if they are missed, the numbers come back to the user.
   - **The scene:** the demo's defaults only.

8. ~~**Step 8: reading ONNX**~~ **Decided 2026-10-14** in
   [`../agents-onnx-reader-design.md`](../agents-onnx-reader-design.md) §14.
   - **Before phase 8.0:**
     - the reader goes in a new entry, `null-graph/inference`, created in this step;
     - so it is a function, `mlpFromOnnx(bytes)`, and the agents entry keeps its 2,172 B;
     - files written by other exporters are committed as fixtures, made in `venv/`.
   - **After phase 8.0** (the recommended option each time):
     - a `Cast` to float32 or a `Flatten` before the first layer and a `Reshape` after the last
       are passed over, which reads scikit-learn's regressors (68 B);
     - "Use OrtBrain" stays out of the refusals until step 9;
     - the reader has a size line of its own; no TensorFlow or Keras fixtures now.
   - **After phase 8.1's first build:**
     - **weights outside the file are refused, with the fix named.** PyTorch's default call
       writes them that way. **To be looked at again in step 9,** where larger models arrive;
     - an `Identity` of an initializer, which PyTorch's older exporter writes for tensors with
       equal values, is followed (65 B); the reader's size line is 2,600 B min+gzip.

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
- 2026-10-12: per-species brains (option B, strategy 1) adopted ahead of step 7; C2 bundle
  allowance raised to +6,000 B.
- 2026-10-12: per-species passes (option B) replace the single-stage strategy and the composite
  after 7.2's measurements; rows kept for 8 or more species.
- 2026-10-12: rows first, then the composite; no fallback. Core now negotiates three storage
  limits (`PROGRESSION_LIMITS`), approved by the user, outside any step.
- 2026-10-13: step 7, part 1 completed. The user asked to "build step 5 first", ahead of step 7,
  part 2, so that teacher data is reproducible even when crowded cells overflow. The user then
  confirmed the scope: step 5 builds the fixed-order lookups only, and the scaling study comes
  later (§5, item 6).
- 2026-10-13: step 5's crowded-cell window is per cell and tick, not per querying agent, after
  phase 5.7 measured the per-agent window at +15% to +26% on a step (§5, item 6).
- 2026-10-13: step 7, part 2 designed and decided (§5, item 7). Against the parent's §9.5: labels
  come from a separate pass of the teacher, and `.mlp.json` is the working format, with `.onnx`
  last. The rest timer does not become an observation.
- 2026-10-14: step 8's reader is a function, `mlpFromOnnx`, in `null-graph/inference`, an entry
  the parent planned for step 9 and step 8 now creates. The parent's `MlpBrain.fromOnnx` is
  retired, and C2's allowance on the agents entry is not spent on it. Files from other exporters
  are committed as test fixtures (§5, item 8).
- 2026-10-14: the reader's subset grows beyond the parent's six operators by nodes that change
  no value: `Cast` to float32, `Flatten` and `Reshape` at the ends of the chain only, and an
  `Identity` of an initializer. Phase 8.0 found them in scikit-learn's files and in PyTorch's
  older exporter's (§5, item 8).
- 2026-10-14: step 9 gains a question from step 8: whether to read weights that an exporter
  stored outside the `.onnx` file (§5, item 8).
