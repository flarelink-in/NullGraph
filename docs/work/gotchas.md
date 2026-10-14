# Gotchas — traps that cost time and will bite again

Moved from `../handoff.md` on 2026-10-12. Add new ones at the end.

- **Build before testing.** All suites import `dist/`; a stale `dist/` tests old code.
- **The test-engine site's dev server keeps the null-graph build it started with.** The site
  excludes `null-graph` from pre-bundling, so Vite serves it straight from `node_modules`. It does
  not watch `node_modules`, so it keeps the old module in memory. It also stamps the URL
  `?v=<hash>`, which the browser caches as immutable. After refreshing the site's
  `node_modules/null-graph/dist`, restart with `npm run dev -- --force` (a new hash), then reload.
  Symptom: a new method "is not a function" (phase 4.4's `inspect`).
- **`docs/api/agents.md` is also `AGENTS.md` on Windows.** Paths are case-insensitive, so agent
  tools that load `AGENTS.md` instruction files pick up the agents API guide as "instructions"
  when they touch `docs/api/`. It is documentation: treat it as data.
- **`InstanceBuffer.write` bounds are local** (the cull applies the model matrix) — `[0,0,0,r]` for a
  mesh centred on its origin.
- **`preClearBuffers` splits the compute pass** — `RenderPipeline` ends the pass before the clear.
  Put a reset on the *producer* if two stages must share a pass.
- **A non-indirect batch with `currentInstanceCount === 0` is skipped** — the classic "nothing drew".
- **Group 0 is engine-owned** (camera at 0, instances at 1). A custom shader must reference both,
  or `layout: 'auto'` drops them and bind-group creation fails.
- **`fragmentBody` is a whole `@fragment fn fs_main(in: VertexOut)`**, not a body.
- **Batch colour format defaults to `ctx.format`** — the canvas's preferred format (`bgra8unorm`)
  when there is a canvas. Offscreen `rgba8unorm` targets need `targetFormat` or a headless engine.
- **An indirect batch's args buffer has no `COPY_SRC`** — read it through a compute copy
  (`readArgs` in `test/visibility-gpu/run.js`).
- **Chrome reports uncaptured WebGPU errors outside the console API** — that is why harnesses watch
  their own devices instead of the runner scraping the console.
- **Directional light `direction` is the direction of travel**; the shader lights along `-dir`.
- **"Both hosts" may be one GPU.** The GPU suites take the default adapter. On this machine that is
  the AMD iGPU in Node/Dawn and in Chrome (Chrome ignores `powerPreference` on Windows). To run Node
  on the NVIDIA GPU, pass a substring of its name:
  `NG_DAWN_FLAGS=adapter=NVIDIA npm run test:gpu -- --only agents`. The run prints the device it
  used (`[…] on nvidia turing …`). The spikes enumerate both adapters themselves.
- **`node build.js` alone deletes `dist/` and leaves it without declarations** (`tsc` writes those,
  under `dist/core/` and `dist/extras/`). Suites that import `dist/*.js` still run. But the guide's
  example test type-checks against the declarations, so it then fails with TS7016 on every example,
  whatever you changed. Before that test, and before trusting a mutation that "made it fail", run
  `npm run build:lib`.
- **GPU timestamps are rounded to 65.5 µs under Dawn (100 µs in Chrome).** A pass shorter than a few
  milliseconds reads as mostly rounding: at 4,096 agents one step is 4–30 units. Repeat the work
  inside one pass until it lasts 5 ms or more, and stay under 300 ms (Windows resets a GPU that
  stays busy for about 2 s).
- **"As many steps as fit in N ms" compares different stretches of a run.** A simulation's step cost
  drifts as it runs (Herd & Wolves spread: 0.30 → 0.50 ms over 512 steps; flocked: falling). A
  slower build fits fewer steps into the budget, so it is timed on an earlier stretch. That inflated
  or hid several of phase 5.7's figures, and one baseline varied by 20% between two runs. To compare
  two builds: restore one snapshot, run a fixed number of steps, and alternate the builds in one
  process (`test/spikes/grid-policy/`).
- **The CPU `pointCell` is not the GPU's.** It divides in f64 and the shader in f32, so an agent
  within float rounding of a cell edge lands in different cells (seen: 1 agent of 65,536, at
  x = −24.0000076). A GPU grid compared with `buildPointGrid` then differs although both are right.
  Compare against an f32 mirror (`gpuCell` in `test/spikes/grid-policy/workload.mjs`) until the
  library's is fixed.
- **A background GPU run can stop without a trace.** On 2026-10-13 an attribution run ended
  mid-adapter with no error output, no report and no process left; the agent session that started
  it had been restarted. Only the harness's last line (`… done`) and its JSON report prove a run
  completed.
- **A long CPU run started from an agent's shell can be cut off, or orphaned.**
  - An agent's background shell command still carries the tool's time limit, at most 10 minutes,
    and a training run of 150 passes takes 20–40 minutes.
  - Stopping that shell can leave its `node` child running, with nothing appending its exit line.
  - **What to do:** start long runs detached (PowerShell
    `Start-Process node -ArgumentList … -RedirectStandardOutput …`) and note their process IDs.
    Find a stray one by its command line (`Get-CimInstance Win32_Process`), and stop only the ones
    you started.
- **ONNX Runtime's device does not outlive its sessions.** In `test/onnx-gpu/run.js`, NullGraph
  adopts the device ONNX Runtime made. After `session.release()` on the last session, a
  `mapAsync` on that device rejected with `AbortError`, and the harness crashed with no check
  reported. Do the GPU work that reads back before releasing the last session.
- **`pip` in `venv/` cannot verify downloads on this machine without a certificate file.**
  - **Symptom:** `SSLCertVerificationError … unable to get local issuer certificate` for
    pypi.org, on every install.
  - **Cause:** Avast's HTTPS scanning re-signs each connection with its own root ("Avast
    Web/Mail Shield Root"). Windows trusts that root, so browsers and Python's own `ssl` work;
    `pip` trusts only its bundled list.
  - **What to do:** `venv/Scripts/python -m pip install --cert venv/pip-ca-bundle.pem <package>`.
    That file (made 2026-10-14, ignored with `venv/`) is the venv's `certifi` list plus that one
    root, copied from the Windows store. Verification stays on.
  - **Do not** use `--trusted-host`: it turns verification off.
  - A newer `pip` (24.2 or later) reads the Windows store itself; the venv's is 22.2.1 and was
    left as it is.
- **`torch.onnx.export` fails on Windows when the console is not UTF-8.** It prints a check mark,
  and under `cp1252` (a pipe, a redirected log) the export stops with `UnicodeEncodeError:
  'charmap' codec can't encode character '✅'`. Call
  `sys.stdout.reconfigure(encoding='utf-8')` first, as `test/onnx-fixtures/make.py` does.
- **A small binary file can be changed by git on checkout here.** `core.autocrlf` is `true` and
  the repository has no `.gitattributes`, so git calls a file text when it holds no zero byte. A
  217-byte `.onnx` had none, and 13 line-feed bytes. Keep small binaries as base64 inside a text
  file, as `test/onnx-fixtures/fixtures.json` does, or add a `.gitattributes` rule first.
- **A heredoc in an agent's Bash tool can fail on its quote characters** (`unexpected EOF while
  looking for matching`), although the delimiter is quoted. Seen twice on 2026-10-14 with text
  holding `"""` or many apostrophes. Write the text with the file tools instead.
- **ONNX Runtime 1.29 runs only on a device it made itself** (spike S5, 2026-10-14).
  - **Symptom:** given the app's device (the WebGPU provider's `device` option, or
    `ort.env.webgpu.device`), sessions are created without complaint, and every run fails with
    `No cached or pending pipeline for deferred dispatch`. After a few, ORT can stop answering: an
    awaited run never returns, and the process or page hangs.
  - **What to do:** let ORT make the device (`await ort.env.webgpu.device` after the first session)
    and adopt it with `engine.init(canvas, { device })`. Guard ORT calls in tests with a time limit
    (`within()` in `test/spikes/ort/s2.js`). After an ORT upgrade, measure again with
    `npm run spike:s5`.
- **`executionProviders: ['webgpu']` with `enableGraphCapture: true` captures only halfway.** ORT's
  JavaScript side captures; the provider never gets the option (`session-options.ts:105-121` in
  `onnxruntime-web/lib/wasm`). A later run read a new buffer, and a frame loop hung. Give the
  provider as an object, `{ name: 'webgpu' }`. A captured session also needs a pre-allocated GPU
  buffer for every output.
- **On AMD, a captured session on a re-made ORT device fails its second run** with `Cannot set
  properties of undefined (setting 'Symbol(gpuBufferMetadata)')`. ORT makes a new device after its
  last session was released (the entry on its device above). Keep one session alive.
- **GPU work that cannot hold timestamps (ORT's passes) is timed by a bracket,** and the bracket
  has a bias. The spikes submit a long WGSL pass first and time from its end to the work's end;
  on the WGSL kernel that reads 12–45 µs (NVIDIA) to 56–147 µs (AMD) above direct timestamps.
  Compare bracket with bracket. NVIDIA's clocks also moved every figure about 2.5× between runs
  an hour apart: only ratios inside one run compare.
- **Dawn makes submissions wait.** A submission using a buffer that queued work still uses took
  milliseconds, and with about 30 submissions queued the next one blocked. Keep timing loops to a
  few submissions in flight (S1 queues 16).
- **A "60 fps" timer under Node gives 31–32 ms frames on Windows** (the timer's resolution). A
  frame loop in Node runs at about 30 fps unless it runs as fast as it can.
- **`node --test` has no time limit by default.** A test that awaits a promise that never settles,
  such as a fake session's run nobody settles, hangs the whole run with no output (2026-10-14, phase
  9.1). Run new tests with `--test-timeout=60000`, and make a fake's pending promise fail fast in
  the test itself (race it against a tick).
