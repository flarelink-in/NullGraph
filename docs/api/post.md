# Post-Processing API Usage

The `null-graph/post` module chains screen-space effects into your render sequence with an explicit,
pre-allocated design: no per-frame garbage, one `writeBuffer` per frame for all params, and a layout
a Web Worker can drive. `PostChain` orchestrates a `TransientPool` (reusable render targets), a
`ParamArena` (packed effect uniforms), and a `ShaderComposer` (fuses effect bodies into passes).

```ts
import { PostChain, Bloom, SSAO, Tonemap, CustomEffect } from 'null-graph/post';
```

> Deeper design notes live in [`post-processing-design.md`](../post-processing-design.md),
> [`transient-pool.md`](../transient-pool.md), and [`post-presets.md`](../post-presets.md).

---

## 1. Create the chain

```ts
const post = new PostChain(engine, {
    format: 'rgba16float',   // HDR working format for all intermediates (default)
    scale: 1.0,              // internal resolution multiplier vs. canvas (default 1.0)
    gbuffer: { depth: 'depth24plus' }, // declare depth if an effect (e.g. SSAO) needs it
    outputToScreen: true,    // last pass targets the swapchain + does linear→display (default)
});
```

## 2. Declare the effect chain

Order matters and is enforced at compile time — the canonical order is
**`[SSAO?, Bloom?, Tonemap]`**, with Tonemap always last (everything upstream is linear HDR).

```ts
post.setChain([
    new SSAO({ radius: 0.6, intensity: 0.9 }),
    new Bloom({ threshold: 1.15, knee: 0.5, intensity: 0.7, mips: 6 }),
    new Tonemap({ curve: 'aces', exposure: 1.0 }),
]);

post.compile(); // builds pools, arenas, and pass pipelines — do this once
```

### Effect options

| Effect | Constructor-only (recompile) | Runtime (free) |
|---|---|---|
| `Tonemap` | `curve` (`'aces'` \| `'reinhard'` \| `'agx'`) | `exposure`, `whitePoint` |
| `Bloom` | `mips` | `threshold`, `knee`, `intensity` |
| `SSAO` | — | `radius`, `bias`, `intensity`, `samples` |

Runtime params can be changed every frame with zero recompilation (they are written into the param
arena); constructor-only params rebuild pipelines, so set them up front. See
[`post-presets.md`](../post-presets.md) for ready-made preset bundles.

## 3. Feed the scene into the chain

`PostChain` provides the color/depth attachments your scene pass should render into, so the chain
owns the intermediate targets:

```ts
const scenePass = engine.createPass({
    name: 'Scene',
    isMainScreenPass: false,
    ...post.sceneAttachments({ clearValue: { r: 0, g: 0, b: 0, a: 1 } }),
});
```

## 4. Render sequence

Concatenate your scene pass with the chain's post passes and render them in order:

```ts
function frame() {
    post.upload(); // one writeBuffer for all effect params

    engine.updateCamera(camera);
    engine.render([scenePass, ...post.getPassSequence()]);

    requestAnimationFrame(frame);
}
```

Handle canvas resizes by forwarding them so the transient targets are reallocated:

```ts
window.addEventListener('resize', () => post.resize(canvas.width, canvas.height));
```

## Custom effects

Drop in your own WGSL with `CustomEffect` — it participates in the same arena/pool machinery:

```ts
import { CustomEffect } from 'null-graph/post';

const vignette = new CustomEffect({
    name: 'Vignette',
    // schema of runtime params + a WGSL `effect(uv)` body
    // (see CustomEffectConfig)
});

post.setChain([vignette, new Tonemap({ curve: 'aces' })]);
```

## Key `PostChain` methods

| Method | Description |
|---|---|
| `setChain(effects)` | Define the ordered effect list. |
| `compile()` | Build pools, param arena, and pass pipelines. Run once after `setChain`. |
| `sceneAttachments(opts?)` | Color/depth attachments for your scene pass. |
| `getPassSequence()` | The `RenderPassNode[]` to append to `engine.render(...)`. |
| `upload()` | Flush all effect params to VRAM (one `writeBuffer`). Call per frame. |
| `setRawParams(data)` | Overwrite the whole param arena from a `Float32Array` (worker-friendly). |
| `describeLayout()` / `describe()` | Inspect the packed param layout and pass graph. |
| `memoryReport()` | Report transient-pool target usage. |
| `resize(w, h)` | Reallocate intermediates for a new resolution. |
| `outputView` | The final `GPUTextureView` when `outputToScreen: false`. |
| `destroy()` | Free all pooled targets and pipelines. |
