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
| S1, S2, S4, S5 | C0 | ONNX Runtime (ORT) cost, latency, startup, device sharing | Deciding step 9's defaults | ✅ 2026-10-14, as step 9's phase 9.0 (results in [its design](../agents-ort-design.md)). S5: ORT 1.29 runs only on its own device |
| 1 | C1 | Agent record, layouts, CPU reference math | The data model everything else uses | ✅ `7f4cdb5` |
| 2 | C1 | GPU spatial grid, neighbour query, debug views | Agents can find neighbours on the GPU | ✅ `4ebfe99` |
| 3 | C1 | `AgentSystem`: sense → think (rules) → act on the GPU, arrows demo | A whole population runs with no CPU per agent | ✅ phases 3.0–3.5 ([design](../agents-gpu-loop-design.md)) |
| 4 | C1 | Instanced darts, picking, labels, inspector, API guide | Agents are visible, clickable and explainable | ✅ phases 4.1–4.5 |
| 5 | C1 | Scaling limits and bottleneck study ([design](../agents-demo-design.md)) | Honest limits for users | **Deterministic neighbour lookups ✅ 2026-10-13** (phases 5.5–5.8, [ordered grid](../agents-ordered-grid-design.md)): reruns are bit-exact with overflow; crowded cells yield a window per cell and tick (the user revised decision 2 after the window per querying agent cost +15% to +26%); a step is within the 10% rule (worst case +9.6%); 5.7 `de87b3f`, 5.8 the guides. **The scaling study (phases 5.0–5.4) is still deferred, so step 5 is not complete** |
| 6 | C2 | `MlpBrain`: a small network runs as the think stage ([design](../agents-mlp-design.md)) | Neural policies run at rule-brain speed | ✅ 6.0 `e345921`, 6.1 `dbaf884`, 6.2 `b3424b1`, 6.3 (guide, cost, size) |
| 7 | C2 | Part 1: per-species brains ([design](../agents-species-brains-design.md), phases 7.1–7.5: weight-table rows, then per-species think passes). Part 2: record the rule brain → train a copy per species → export (`tools/agents`), from 7.6 | **First network that has actually learned Herd & Wolves** | Part 1: 7.1 ✅, 7.2 ✅ (measured; option B chosen), 7.3 ✅ (per-species passes, `681ccf4`), 7.4 ✅ (A22, `f19dbea`), 7.5 ✅ (the guide, `de14ff4`): **part 1 complete**. Part 2: design decided 2026-10-13 (`f1244cb`, [cloning](../agents-cloning-design.md), phases 7.6–7.10); 7.6 ✅ (the scenario, the label pass, the recorder, `c2608d7`); 7.7 ✅ (the trainer; round-0 students, `0d3fb9f`); 7.8 ✅ (the copy in the running simulation: catches within 1% of the rule without wander, no drift, no round of correction, `8afc6fa`); 7.9 ✅ (the two networks committed in `tools/agents/models/herd-wolves`, U26, A26, A8, the guide's "A trained network", the tools' README, `cd76a25`); 7.10 ✅ (the `.onnx` writer, U12, A27: ONNX Runtime within 4.5e-6 of `mlpForward`, `834b360`). **Part 2 is built and committed** |
| 8 | C2 | Reading ONNX: `mlpFromOnnx` in a new entry, `null-graph/inference` (the parent's `MlpBrain.fromOnnx`; [design](../agents-onnx-reader-design.md), phases 8.0–8.3) | Load models trained anywhere | Design decided 2026-10-14 (`a8778ce`; §5, item 8). 8.0 ✅ (`7829f4f`): 16 files from PyTorch, scikit-learn and `onnx.helper`, with what each maker computed. 8.1 ✅ (`8aa243a`): the entry and `mlpFromOnnx` (2,525 B; 13 foreign files read bit for bit, 3 refused as listed; 34 of 34 mutations caught). 8.2 ✅ (`874488d`; A28: the reader's `MlpBrain` and ONNX Runtime agree within 4.5e-6 on 17 files, on three host/GPU pairs). 8.3 ✅ (the commit after `874488d`; the guide `docs/api/inference.md` with a checked example, the README row, the parent's wording). **Every phase is built** |
| 9 | C2 | `OrtBrain` and the generic ONNX Runtime adapter, in `null-graph/inference` (ORT optional peer) | Larger models; shared with the photo-booth and relight ideas | Design decided 2026-10-14 ([design](../agents-ort-design.md), phases 9.0–9.6; §5, item 9). 9.0 ✅ (the spikes and the design, `a57cd76`). 9.1 ✅ (`OrtRuntime`, `OrtModel`, the anchor; A29 on three host/GPU pairs; `e2ac9a4`). 9.2 ✅ (weights in a second file, read both ways; A32 on three pairs; capture off for such models, §5 item 10; `a4f4035`). 9.3 ✅ (`OrtBrain`, async brains in `AgentSystem`; A5, A30, A31 on three pairs; `e1344ca`). 9.4 ✅ (A6 on three pairs; graph capture off by default, §5 item 11; `f11ba4f`). 9.5 ✅ (`createComputeStageAsync`, `prepareBrain`; A33 on three pairs; `35e5be1`). 9.6 ✅ (the guides; `4e19a0f`). **Every phase is built** |
| 10 | C2 | Demo: switch Rules / Network / ORT live with timings | The showable result | Design decided 2026-10-14 (`2b8f689`; [design](../agents-live-switch-design.md), phases 10.0–10.5, in the site's Herd & Wolves demo; §5, item 12). 10.0 ✅ (the spikes, `9320373`: ONNX Runtime in a worker through the site's Vite, the device switch, ONNX Runtime at 4K–1M agents, the overlay's timing). After it the user revised D5 (Vite serves ONNX Runtime's files) and D6 (no cap) (§5, item 13). 10.1 ✅ (the plumbing, in the site, recorded in `22b67f8`: `onnxruntime-web` 1.29.0 pinned and served by Vite, the two `.onnx` files beside the `.mlp.json`; the site builds and the page loads as before). 10.2 ✅ (in the site, recorded with `check-site.mjs` in `ab3d9e1`: the rules a checked copy of the scenario (D3), both networks compiled ahead, every switch 0.0–0.3 ms and compiled ahead, the switch line). 10.3 ✅ (in the site, recorded in the commit after `ab3d9e1`: the device switch on the first ORT pick, 0.78–1.3 s; ORT as the third brain per species, its `.onnx` checked against the `.mlp.json`, prepared and warmed up; 61 switches compiled ahead, 0.0–1.1 ms; a refused model and ORT not starting handled; ONNX Runtime's calls queued in the demo, §5 item 14). §5 item 14 decided B: `OrtRuntime` queues them (2026-10-15, `f8c4935`), and the demo's queue is gone. 10.4 ✅ (in the site, recorded in the commit after `f8c4935`: sense and think timed apart; ONNX Runtime's own GPU work per run, its submissions bracketed, 1.0–2.1 / 3.6–4.3 / 15.4–17.6 ms at 4K / 16K / 64K; its `run()` time, runs a second, the actions' lateness, its start; §8's two marks counted the GPU idling while ONNX Runtime checks for errors after each run). 10.5 ✅ (in the site: ONNX Runtime at every count under the timeout guard, its run predicted from the runs measured on this GPU, and measured first over 65,536 rows above that count; a refused species runs its network on the built-in kernel; the steps' guard keeps its measurements across the device switch, which had sent a first ORT pick at 256K back to 4K; the documentation page rewritten; the guides point to the demo). **Every phase is built: step 10 is complete, and with it C2** |

**C1 is a product on its own:** crowds, herds and RTS units with no ML.

### Cycle D — animated agents (decided 2026-10-15)

GPU agents drawn as animated low-poly models whose animation follows each agent's simulation
state, the rendering backbone Phase 2's creatures will need, and the showcase of C1–C2.

| Step | Delivers | Serves the goal by | Status |
|---|---|---|---|
| D1 | `bakeRigidAnimation` (loaders), `vertexHook` (visibility), "Draw agents as: Models" and a toon field in Herd & Wolves ([design](../agents-animated-models-design.md), phases 0–5) | Agents that look like animals, not darts, at thousands per frame | Design decided 2026-10-15 (§11 there). Phases 0–3 ✅: costs measured; rigid bake `f538ffe`; vertex hook `75824a7`; Models mode in the site (`77c8e92` there, checks `fdf3ede`). Phase 4 ✅: built by Cline (site `850ab4f`, records `e96af9d`), then reviewed with the user ("Phase 4 review" in the design): the new `anime()` preset `8b68286`, LOD bands in visibility `c56de95`, far cubes, fences and stock materials in the site `ca81605`. Next: phase 5 (skinned models) |

Phase 5 update (2026-10-15): 5.1 skinned bake and 5.2 palette snippets committed; 5.3 site rigid
palette migration validated, with bounded allocation/legacy fallback and no reliable rigid speedup.
Next is 5.4, when the user's skinned animals are available. See the skinned-models design results.

### Other open work (outside Cycle C)

- Agent kit follow-up: guidance fixes, a pinned-tarball trial, then release/tag (handoff "Agent kit").
- Verify, then fix, the `uvMinA` shadow-atlas suspect with a GPU test.
- CI on a pinned Linux + lavapipe environment, only if golden images are wanted.

## 4. Horizon goals — how far each one is

| Goal | Needs | Distance |
|---|---|---|
| Batched neural agents: one trained brain, many agents, 60 fps, inference measured | Steps 6 → 7 (→ 10 to show it) | ✅ **Reached 2026-10-15:** step 10's demo shows it, with both species on a network, on the built-in kernel or on ONNX Runtime, at 60 fps from 4K to 16K agents on the AMD iGPU |
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
   **Deferred by the user to step 9 (2026-10-12).** **Decided 2026-10-14 (item 9): prepare at
   load, compiled in the background.**
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
       writes them that way. **To be looked at again in step 9,** where larger models arrive.
       **Looked at again 2026-10-14 (item 9): both paths read them;**
     - an `Identity` of an initializer, which PyTorch's older exporter writes for tensors with
       equal values, is followed (65 B); the reader's size line is 2,600 B min+gzip.

9. ~~**Step 9: ONNX Runtime brains**~~ **Decided 2026-10-14** in
   [`../agents-ort-design.md`](../agents-ort-design.md) §14, after phase 9.0's spikes. The user:
   "ORT creates the device, graph capture on, prepare at load compiled in the background, and both
   paths read separate weight files."
   1. **Device ownership** (the parent's decision 6): ORT creates the device and the engine
      adopts it; ORT 1.29 runs on no device the app made (S5). The parent's §12.3 is withdrawn.
   2. **Graph capture on by default.** *Superseded by item 11: off by default.*
   3. **The brain-swap stall** (item 3 above): `agents.prepareBrain(brain)` compiles at load, in
      the background, over a new core method, `createComputeStageAsync` (a core addition).
   4. **Weights outside the `.onnx` file** (item 8 above): both paths read them, ORT and
      `mlpFromOnnx`. The reader's size line moves from 2,600 B to about 2,900 B, set from the
      build in phase 9.2 and reported.
10. ~~**Step 9: graph capture for a model whose weights are in a second file**~~ **Decided
    2026-10-14** (the user: "ok pick A"), raised by phase 9.2
    ([`../agents-ort-design.md`](../agents-ort-design.md) §14, item 5). Under capture, ONNX Runtime
    1.29 can fail such a model from its second run, on every host, depending on what ran before.
    **Capture is off for such models unless the app asks for it;** every other model keeps it, as
    decided in item 9. *Superseded by item 11.*
11. ~~**Step 9: graph capture for every model**~~ **Decided 2026-10-14** (the user chose option A),
    raised by phase 9.4 ([`../agents-ort-design.md`](../agents-ort-design.md) §14, item 6). A6
    found that under capture ONNX Runtime looks up a model's output, from its second run on,
    through a handle it freed after the first: on 1.29 the second run can fail when another session
    is created, run or released between the first two runs; on 1.30.0, which `^1.29.0` admits, the
    smallest case failed with nothing in between (Node/NVIDIA, 3 of 3; not on AMD). Which case
    fails depends on the host and on what ran before. Capture also leaks memory on every run.
    **Capture is off by
    default for every model; `capture: true` stays as the app's choice.** Item 9's second point and
    item 10 are superseded.
12. ~~**Step 10: the live switch in the site's Herd & Wolves demo**~~ **Decided 2026-10-14**
    ([`../agents-live-switch-design.md`](../agents-live-switch-design.md) §12):
    - **D1:** the page gets ONNX Runtime's device on the first ORT pick: the worker makes its
      engine again on it and loads the demo again (the site's worker makes its own device at
      start, which ORT 1.29 cannot use).
    - **D2:** Network keeps loading `.mlp.json`; ORT loads `.onnx`, checked against the
      `.mlp.json` bit for bit. (The recommendation was one `.onnx` for both.)
    - **D3:** the demo's rules are a copy of the scenario file's texts, checked against the
      teacher's hash in the `.mlp.json` files.
    - **D4:** `onnxruntime-web` 1.29.0, pinned in the site. **D5:** its WASM vendored in
      `public/vendor/`. **D6:** ORT capped at the largest count phase 10.0 measures as safe.
      **D5 and D6 were revised after phase 10.0: item 13.**
13. ~~**Step 10, after phase 10.0: where ONNX Runtime's WASM comes from, and the ORT cap**~~
    **Decided 2026-10-14** (raised the same day;
    [`../agents-live-switch-design.md`](../agents-live-switch-design.md), "Phase 10.0 results", §12).
    - **D5: Vite serves ONNX Runtime's own files,** the recommendation: no copy to vendor or
      refresh, and no second copy in a build.
    - **D6: no cap.** Anyone can view the site, and visitors with stronger GPUs can test larger
      counts. ORT is offered at every count the demo builds. The timeout guard, the rule for every
      demo with a size control, still covers ORT's run (the design's §9).

    What the user was asked:
    - **D5 does not work as written.** In Vite's dev server, ONNX Runtime cannot import its loader
      from `public/vendor/`. Recommended: let Vite serve ONNX Runtime's own files. That means
      `onnxruntime-web` in `optimizeDeps.exclude` and no `wasmPaths`; the build emits one hashed
      `.wasm`. The alternative: vendor the `.wasm` alone, with `wasmPaths: { wasm }`. The build
      then also holds Vite's copy: a second, unused 25.7 MB file.
    - **D6's number:** 65,536 agents. §9's rule, "within the frame budget" (the demo's 40 ms of GPU
      work per frame), gives it from Chrome on the AMD iGPU. Both species' runs take 25 ms there,
      and one run takes 51 ms at 262,144.
14. ~~**Step 10, after phase 10.3: should `OrtRuntime` queue ONNX Runtime's calls?**~~ **Decided
    2026-10-15: B** (the user: "yes choose B"; raised the same day,
    [`../agents-live-switch-design.md`](../agents-live-switch-design.md), "Phase 10.3 results",
    finding 1). `OrtRuntime` queues every call of its models: load, run and release, on one chain
    per runtime. The adapter's size line moves, and the demo drops its wrapper.

    What the user was asked: ONNX Runtime 1.29 takes one call at a time across its sessions. Two
    models loading together fail, and one loading or running while the other runs breaks its WASM
    memory, with uncaught errors. Any app with two `OrtBrain`s meets it, one per species for
    instance. Step 9's suites never have two calls in flight at once, so they never did.
    - **A. The app queues,** as the demo does now (it wraps each model's `run`); the guides say so.
      No library change.
    - **B. `OrtRuntime` queues every call of its models** (recommended): load, run and release, on
      one chain per runtime. It needs a test against the fake `ort`, a GPU test of two models at
      once on three host/GPU pairs, and a new size line for the adapter (3,000 B; 2,988 B today).
      The demo then drops its wrapper.

    Best decided before 10.4, which times ORT's runs: with the queue, one species' run can wait
    for the other's.

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
- 2026-10-14: after step 9's phase 9.0 (spike S5), the user recorded that ORT creates the device
  and the engine adopts it. The parent's engine-first direction and its §12.3 (`requiredLimits`)
  are withdrawn (§5, item 9).
- 2026-10-14: step 9 grows by two pieces the user decided (§5, item 9): brains prepared at load
  and compiled in the background (`prepareBrain`, over a new core method,
  `createComputeStageAsync`: a core addition, approved), and `mlpFromOnnx` reading weights stored
  outside the `.onnx` file, as ORT does (the reader's size line rises to about 2,900 B, set in
  phase 9.2). Step 9's phases are 9.0–9.6.
- 2026-10-14: graph capture is off by default for a model whose weights are in a second file,
  unless the app asks (§5, item 10), after phase 9.2 found that ONNX Runtime 1.29 can fail such a
  model under capture. The reader's size line is 2,900 B.
- 2026-10-14: graph capture is off by default for every model (§5, item 11), after phase 9.4's A6
  found that ONNX Runtime can fail any captured model from its second run (on 1.30.0 even with
  nothing between its first two runs) and leaks memory under capture. Replaces item 9's "capture on" and item 10's special case.
- 2026-10-14: step 10 designed and decided (§5, item 12). The live switch is built in the site's
  Herd & Wolves demo, whose worker makes its engine again on ONNX Runtime's device at the first
  ORT pick. Network keeps loading `.mlp.json`: step 8's "the demo loading `.onnx`" holds for ORT
  only.
- 2026-10-14: after step 10's phase 10.0, the user revised two of its decisions (§5, item 13).
  Vite serves ONNX Runtime's files, where a vendored copy was planned (D5). ORT has no cap on the
  agents it runs over, where a measured cap was planned (D6): visitors with stronger GPUs can test
  larger counts.
- 2026-10-15: `OrtRuntime`, step 9's adapter, queues ONNX Runtime's calls one at a time (§5,
  item 14), after step 10's phase 10.3 found that ONNX Runtime 1.29 fails two calls in flight. Its
  size line moves, set from the build. A fix outside step 10's phases, as the profiler's was
  (`a8d0107`).
- 2026-10-15: the user's direction for brains after C2, not scheduled. ONNX Runtime stays: as the
  reference that `MlpBrain` is checked and kept matched against, for networks the kernel cannot
  run, and for trying ideas, since it is more general than null-graph's scope. `MlpBrain` widens to
  bridge what null-graph's 3D cases need, so that more stays in the library's control: a tiled
  kernel, wider layers, discrete output heads, memory, per-agent weight rows, a sensor that reads
  app buffers. Recording and training stay in this repository's tools (`agents:record`,
  `agents:train`).
- 2026-10-15: C2 complete. The user opened **Cycle D, animated agents**, ahead of Phase 2: baked
  rigid (later skinned) animation in `null-graph/loaders` and a vertex hook in
  `null-graph/visibility`, both public API, then a Models draw mode with a toon field in Herd &
  Wolves; deer or cow for the prey, the fox for the wolves; one accumulated animation phase per
  agent ([design](../agents-animated-models-design.md) §11).

- **2026-10-15, D1.3:** Models mode implemented in the site with cow prey and fox wolves (user reconfirmed cow/deer instead of pig). D1.2 committed `75824a7`. D1.4 field/grass/toon/post remains next; see now.md and the animated-model design.

- **2026-10-15, D1.4:** User approved proceeding with the documented Models-only field scope and
  requested Models → Darts → debug Arrows ordering, Darts default, and public model/palette assets
  fetched on demand. Implemented locally with static culled grass, toon bands, optional anime
  post and the site documentation panel. No engine API changes or model assets added.

- **2026-10-15, D1 phase 4 review:** after seeing D1.4, the user asked for a new anime look
  ("sharp lines and softer look"), the post chain wired through `ChainPresets` as other demos do,
  far animals as white and orange cubes, broken and leaning fences at the field's edge, and stock
  materials where custom shaders were used ("the preview and darts"). Decided: `ChainPresets.anime()`
  itself changes (a library look change; the VisibilityMeshlets demo uses it too); LOD bands by size
  on screen become public API in `null-graph/visibility` (`InstanceDrawOptions.lod`); fences are built
  from boxes in stock toon wood, Models mode only; the animals, field and grass keep their own
  fragments ([design](../agents-animated-models-design.md), "Phase 4 review").

- **2026-10-15, D1 phase 5 (the user's "10.5"):** started at the user's word, designed in its own
  file ([design](../agents-skinned-models-design.md)). Decided: a second bake,
  `bakeSkinnedAnimation`, beside the rigid one; a public WGSL sampling helper, `BAKED_POSE_WGSL`, in
  loaders; the Mixamo dancers, the butterfly and in-memory fixtures as test models while no rigged
  animal exists (the demo keeps the Kenney animals); a cost spike first (5.0).
- **2026-10-15, after phase 5.0:** the spike measured a per-agent palette (skinned at 0.69–0.98× the
  rigid hook, against 2.2–2.8× for sampling per vertex). The user adopted it, with one slot per
  visible model, moved the demo's rigid Models onto it now, and chose WGSL snippets in loaders over a
  class for now. Phases 5.1–5.4 as in the [design](../agents-skinned-models-design.md).
- **2026-10-15:** phase 5.1 committed `e84e308`; 5.2 public palette snippets and visible-slot GPU
  verification implemented. Next 5.3: migrate demo rigid Models onto the visible palette and measure.
- **2026-10-15:** 5.3 committed (site `17cd4af`, validation `84a04dd`). User requested commits,
  `yalc:push`, 5.4 start and slowdown investigation. Contiguous producer writes improve AMD 16K;
  asset-intake checker starts 5.4, but actual skinned-animal integration awaits the user's GLBs.
