# Post-Processing Presets

> **Status: implemented** in `src/extras/post/presets.ts` and exported from
> `null-graph/post` (`TonemapPresets`, `BloomPresets`, `SSAOPresets`, `ChainPresets`,
> `forSceneScale`). Written when the library was three effects; §2 and §6 have been
> updated for the full effect set.

---

## 1. The constraint that shapes the API

Your effects split params two ways, and presets must respect the split:

| | Tonemap | Bloom | SSAO |
|---|---|---|---|
| **Constructor-only** (recompile) | `curve` | `mips` | — |
| **Runtime** (free, arena write) | `exposure`, `whitePoint` | `threshold`, `knee`, `intensity` | `radius`, `bias`, `intensity`, `samples` |

So a preset is **constructor options**, not a runtime setter bundle. `applyPreset()` at
runtime can only move the runtime row.

This gives two tiers, and they should be named differently so nobody is surprised by a
recompile mid-frame:

- **`Presets.*`** → option objects passed to constructors. Full range, requires rebuild.
- **`Presets.*.runtime`** → the runtime-only subset, safe to apply per-frame.

```ts
// build time — full preset
const bloom = new Bloom(BloomPresets.cinematic);

// runtime — no recompile, only the runtime subset moves
Object.assign(bloom, BloomPresets.dreamy.runtime);
```

---

## 2. Ordering is part of the preset

The original rule was `[SSAO?, Bloom?, Tonemap]` with Tonemap last, always. With the full
library that generalizes to a canonical stage order — Tonemap is no longer last, but it is
still the boundary that splits the chain in two:

```
SSAO -> AutoExposure -> Bokeh -> Bloom / Halation / LensFlare -> ColorGrade
  -> Tonemap ->
CelShade -> Kuwahara -> Outline -> FXAA -> ChromaticAberration -> Vignette -> FilmGrain
```

- **Before Tonemap: everything that reasons about light.** SSAO before Bloom (occluded
  creases must not glow), AutoExposure before Bloom (so the threshold sees normalized
  luminance), scatter effects (Bloom/Halation/LensFlare) and ColorGrade on linear HDR.
- **After Tonemap: everything that reasons about the image.** CelShade bands display-referred
  values (banding HDR crushes everything into the bottom band), FXAA's luma estimate assumes
  display range, and grain/vignette/CA are lens- and film-artifacts of the final picture.
- **FXAA after Outline**, so ink edges get anti-aliased; **FilmGrain last**, so nothing
  smooths it away.

The old "throw if Tonemap is not last" enforcement idea is obsolete — post-Tonemap effects
are legitimate. The ordering above ships as documentation and as the order every
`ChainPresets.*` factory uses.

---

## 3. Tonemap presets

`curve` is constructor-only. `whitePoint` is meaningful for `reinhard`; ACES and AgX have their
own built-in shoulder and largely ignore it.

| Preset | curve | exposure | whitePoint | Use for |
|---|---|---|---|---|
| `neutral` | `aces` | 1.0 | 4.0 | Default. Film-like rolloff, slight highlight desaturation. |
| `filmic` | `aces` | 1.15 | 6.0 | Games, cinematics. Brighter, wider highlight latitude. |
| `natural` | `agx` | 1.0 | 4.0 | Archviz, product. Best hue retention at extreme values. |
| `soft` | `reinhard` | 1.0 | 2.5 | Low contrast, never clips. Reads flat but safe. |
| `faithful` | `none` | 1.0 | 1.0 | **Dataviz, UI, color-critical.** Clamp only, no curve. |

**`faithful` matters more than it looks.** Any curve is a nonlinear remap — feed it `#3B82F6`
and you get back something that is not `#3B82F6`. For a library named NullGraph, rendering a
chart in the exact authored color is a correctness requirement, not a preference. `faithful`
is the only preset that guarantees it.

---

## 4. Bloom presets

`mips` is constructor-only.

| Preset | threshold | knee | intensity | mips | Character |
|---|---|---|---|---|---|
| `subtle` | 1.40 | 0.30 | 0.25 | 5 | Barely there. Highlights get a hint of lift. |
| `natural` | 1.15 | 0.50 | 0.50 | 6 | Default. Reads as lens behavior, not an effect. |
| `cinematic` | 1.00 | 0.60 | 0.80 | 7 | Wide soft halo around anything over white. |
| `dreamy` | 0.60 | 0.80 | 1.00 | 7 | Everything glows. Bloom as the look. |
| `neon` | 0.90 | 0.20 | 1.40 | 7 | Hard threshold, aggressive. Emissive-heavy scenes. |

### The bloom question you will get asked most

> "I enabled bloom and nothing happens."

`threshold` is in **linear HDR units, pre-tonemap**. A threshold of 1.15 means *brighter than
white*. If the scene has no emissive materials and no light intensities above 1.0, nothing in
the frame ever exceeds the threshold and bloom is a correct no-op.

Put this in the docstring, and consider a dev-mode warning: if the prefilter output is entirely
zero for N consecutive frames, log once that the threshold may be above the scene's peak.

### mips silently clamps by resolution

The clamp keeps the smallest mip ≥ 8 px. At 1080p, `mips: 7` bottoms out at 15×8 — already at
the limit. At 720p the same preset resolves to 6 mips, and at 4K to 8.

**A preset with high `mips` therefore looks different at different resolutions**, with a wider
glow at 4K than at 720p. That is defensible as "physically consistent in screen space," but it
must be documented, because it will otherwise be reported as a bug. If you would rather have
consistency, derive `mips` from a target glow radius in fractional screen width instead of a
raw count — that is a small change and worth considering before these presets are public API.

---

## 5. SSAO presets

| Preset | radius | bias | intensity | samples | Character |
|---|---|---|---|---|---|
| `performance` | 0.50 | 0.025 | 0.80 | 8 | Mobile / integrated GPUs. |
| `balanced` | 0.60 | 0.020 | 0.90 | 16 | Default. |
| `quality` | 0.80 | 0.020 | 1.00 | 32 | Kernel maximum. |
| `contact` | 0.25 | 0.015 | 1.20 | 16 | Tight creases only. Crisp, product-render look. |
| `ambient` | 1.50 | 0.035 | 0.70 | 24 | Large-scale soft occlusion. Interiors. |

### The caveat that makes SSAO presets different from the others

**`radius` is in view-space world units.** Bloom and Tonemap presets are unit-free and portable.
SSAO presets are not: `radius: 0.6` in a scene authored in meters is a 60 cm sampling sphere,
and in a scene authored in centimeters it is 6 mm — invisible.

So these numbers are correct **only for a roughly 1-unit-per-meter scene.** Ship them with that
stated, and give people a way to adapt:

```ts
/** Scale a radius-bearing preset from meters to the scene's unit convention. */
export function forSceneScale<T extends { radius: number; bias: number }>(
  preset: T, unitsPerMeter: number,
): T {
  return { ...preset, radius: preset.radius * unitsPerMeter,
                      bias:   preset.bias   * unitsPerMeter };
}

new SSAO(forSceneScale(SSAOPresets.balanced, 100));  // centimeter scene
```

`bias` scales with `radius` because it is a depth offset in the same units. Too low and flat
surfaces get self-occlusion acne at grazing angles; too high and the AO detaches from contact
points, leaving a bright halo where geometry meets the floor.

---

## 6. Chain presets

Complete looks. Each returns freshly constructed effects — presets must never share instances,
since `PostChain` rejects the same instance twice and each holds a live arena subarray.

| Preset | Chain | Intent |
|---|---|---|
| `faithful` | Tonemap`none` | **Dataviz, UI, color-critical.** Authored colors survive. |
| `neutral` | Tonemap`aces` | Correct HDR→display and nothing else. Safe default. |
| `product` | SSAO`contact` → Bloom`subtle` → Tonemap`natural` | E-commerce, product viz. Clean, accurate. |
| `archviz` | SSAO`ambient` → Bloom`subtle` → Tonemap`natural` @1.1 | Interiors. Soft occlusion, restrained glow. |
| `cinematic` | SSAO`balanced` → Bloom`cinematic` → Tonemap`filmic` | Games, story scenes. The "AAA" look. |
| `ethereal` | Bloom`dreamy` → Tonemap`soft` | Dreamlike. No SSAO — contact shadows fight the haze. |
| `synthwave` | Bloom`neon` → Tonemap`filmic` @1.2 | Emissive-heavy, dark backgrounds, neon. |
| `performance` | SSAO`performance` → Bloom`subtle`(mips 4) → Tonemap`neutral` | Mobile / integrated. |

### Style chains added with the expanded library

| Preset | Chain | Intent |
|---|---|---|
| `anime` | Bloom`natural` → Tonemap`filmic` → CelShade(5 bands) → Outline → FXAA | Toon shading with ink silhouettes. Best with `gbuffer.depth`. |
| `comic` | Tonemap`filmic` → CelShade(3 bands) → Outline(thick) → FXAA → Vignette | Harder bands, thicker ink, punchier color. |
| `painterly` | Bloom`subtle` → Tonemap`natural` → Kuwahara(r5) → Vignette | Brushwork surfaces, crisp edges. |
| `analog` | Halation → Bloom`subtle` → Tonemap`neutral` → FXAA → CA → Vignette → FilmGrain | 35mm print look. |
| `noir` | ColorGrade(desat, high contrast) → Tonemap`filmic` → Vignette(heavy) → FilmGrain(mono) | Near-monochrome drama. |

Thirteen chain presets, five per effect. That is a large enough menu to cover real use and
small enough that each one is visibly distinct — which is the actual test. If two presets
produce frames a user cannot tell apart, one of them is noise.

Chains using SSAO require the chain to be built with `gbuffer.depth` (SSAO throws without
it); `Outline` degrades to luma-only edges, losing silhouettes against similar-luminance
backgrounds.

---

## 7. `presets.ts`

```ts
import { Bloom, type BloomOptions } from './Bloom';
import { SSAO, type SSAOOptions } from './SSAO';
import { Tonemap, type TonemapOptions } from './Tonemap';
import type { Effect } from './Effect';

/** Runtime-safe subsets: applying these never triggers a recompile. */
type BloomRuntime   = Pick<Required<BloomOptions>,   'threshold' | 'knee' | 'intensity'>;
type SSAORuntime    = Pick<Required<SSAOOptions>,    'radius' | 'bias' | 'intensity' | 'samples'>;
type TonemapRuntime = Pick<Required<TonemapOptions>, 'exposure' | 'whitePoint'>;

type Preset<Opts, Runtime> = Readonly<Opts> & { readonly runtime: Readonly<Runtime> };

function preset<Opts extends object, R extends object>(opts: Opts, runtime: R): Preset<Opts, R> {
  return Object.freeze({ ...opts, runtime: Object.freeze(runtime) }) as Preset<Opts, R>;
}

// ---------------------------------------------------------------- Tonemap

export const TonemapPresets = {
  /** Default. Film-like rolloff. */
  neutral:  preset<TonemapOptions, TonemapRuntime>(
    { curve: 'aces',     exposure: 1.0,  whitePoint: 4.0 }, { exposure: 1.0,  whitePoint: 4.0 }),
  /** Brighter, wider highlight latitude. */
  filmic:   preset<TonemapOptions, TonemapRuntime>(
    { curve: 'aces',     exposure: 1.15, whitePoint: 6.0 }, { exposure: 1.15, whitePoint: 6.0 }),
  /** Best hue retention at extremes. Archviz, product. */
  natural:  preset<TonemapOptions, TonemapRuntime>(
    { curve: 'agx',      exposure: 1.0,  whitePoint: 4.0 }, { exposure: 1.0,  whitePoint: 4.0 }),
  /** Low contrast, never clips. */
  soft:     preset<TonemapOptions, TonemapRuntime>(
    { curve: 'reinhard', exposure: 1.0,  whitePoint: 2.5 }, { exposure: 1.0,  whitePoint: 2.5 }),
  /** Clamp only. The only preset that preserves authored colors exactly. */
  faithful: preset<TonemapOptions, TonemapRuntime>(
    { curve: 'none',     exposure: 1.0,  whitePoint: 1.0 }, { exposure: 1.0,  whitePoint: 1.0 }),
} as const;

// ------------------------------------------------------------------ Bloom
// `threshold` is LINEAR HDR, pre-tonemap. Values > 1.0 mean "brighter than
// white" — a scene with no emissives or lights above 1.0 will not bloom.

export const BloomPresets = {
  subtle:    preset<BloomOptions, BloomRuntime>(
    { threshold: 1.40, knee: 0.30, intensity: 0.25, mips: 5 },
    { threshold: 1.40, knee: 0.30, intensity: 0.25 }),
  natural:   preset<BloomOptions, BloomRuntime>(
    { threshold: 1.15, knee: 0.50, intensity: 0.50, mips: 6 },
    { threshold: 1.15, knee: 0.50, intensity: 0.50 }),
  cinematic: preset<BloomOptions, BloomRuntime>(
    { threshold: 1.00, knee: 0.60, intensity: 0.80, mips: 7 },
    { threshold: 1.00, knee: 0.60, intensity: 0.80 }),
  dreamy:    preset<BloomOptions, BloomRuntime>(
    { threshold: 0.60, knee: 0.80, intensity: 1.00, mips: 7 },
    { threshold: 0.60, knee: 0.80, intensity: 1.00 }),
  neon:      preset<BloomOptions, BloomRuntime>(
    { threshold: 0.90, knee: 0.20, intensity: 1.40, mips: 7 },
    { threshold: 0.90, knee: 0.20, intensity: 1.40 }),
} as const;

// ------------------------------------------------------------------- SSAO
// `radius` and `bias` are VIEW-SPACE WORLD UNITS. These values assume a
// ~1-unit-per-meter scene. Use forSceneScale() for other conventions.

export const SSAOPresets = {
  performance: preset<SSAOOptions, SSAORuntime>(
    { radius: 0.50, bias: 0.025, intensity: 0.80, samples: 8 },
    { radius: 0.50, bias: 0.025, intensity: 0.80, samples: 8 }),
  balanced:    preset<SSAOOptions, SSAORuntime>(
    { radius: 0.60, bias: 0.020, intensity: 0.90, samples: 16 },
    { radius: 0.60, bias: 0.020, intensity: 0.90, samples: 16 }),
  quality:     preset<SSAOOptions, SSAORuntime>(
    { radius: 0.80, bias: 0.020, intensity: 1.00, samples: 32 },
    { radius: 0.80, bias: 0.020, intensity: 1.00, samples: 32 }),
  contact:     preset<SSAOOptions, SSAORuntime>(
    { radius: 0.25, bias: 0.015, intensity: 1.20, samples: 16 },
    { radius: 0.25, bias: 0.015, intensity: 1.20, samples: 16 }),
  ambient:     preset<SSAOOptions, SSAORuntime>(
    { radius: 1.50, bias: 0.035, intensity: 0.70, samples: 24 },
    { radius: 1.50, bias: 0.035, intensity: 0.70, samples: 24 }),
} as const;

/** Adapt a radius-bearing SSAO preset from meters to the scene's unit scale. */
export function forSceneScale(preset: SSAOOptions, unitsPerMeter: number): SSAOOptions {
  return {
    ...preset,
    radius: (preset.radius ?? 0.6)  * unitsPerMeter,
    bias:   (preset.bias   ?? 0.02) * unitsPerMeter,
  };
}

// ----------------------------------------------------------------- Chains
// Factories, never shared instances: PostChain rejects a duplicate instance,
// and each effect holds a live subarray of the param arena.
// Order is always [SSAO?, Bloom?, Tonemap] — see §2.

export const ChainPresets = {
  /** Dataviz / UI. Authored colors survive to the display unchanged. */
  faithful:    (): Effect[] => [new Tonemap(TonemapPresets.faithful)],

  /** Correct HDR->display and nothing else. */
  neutral:     (): Effect[] => [new Tonemap(TonemapPresets.neutral)],

  product:     (): Effect[] => [
    new SSAO(SSAOPresets.contact),
    new Bloom(BloomPresets.subtle),
    new Tonemap(TonemapPresets.natural)],

  archviz:     (): Effect[] => [
    new SSAO(SSAOPresets.ambient),
    new Bloom(BloomPresets.subtle),
    new Tonemap({ ...TonemapPresets.natural, exposure: 1.1 })],

  cinematic:   (): Effect[] => [
    new SSAO(SSAOPresets.balanced),
    new Bloom(BloomPresets.cinematic),
    new Tonemap(TonemapPresets.filmic)],

  /** No SSAO: contact shadows fight the haze. */
  ethereal:    (): Effect[] => [
    new Bloom(BloomPresets.dreamy),
    new Tonemap(TonemapPresets.soft)],

  synthwave:   (): Effect[] => [
    new Bloom(BloomPresets.neon),
    new Tonemap({ ...TonemapPresets.filmic, exposure: 1.2 })],

  performance: (): Effect[] => [
    new SSAO(SSAOPresets.performance),
    new Bloom({ ...BloomPresets.subtle, mips: 4 }),
    new Tonemap(TonemapPresets.neutral)],
} as const;

export type ChainPresetName = keyof typeof ChainPresets;
```

### Usage

```ts
import { PostChain, ChainPresets, SSAOPresets, forSceneScale } from 'null-graph/post';

post.setChain(ChainPresets.cinematic());

// centimeter-scale scene
post.setChain([
  new SSAO(forSceneScale(SSAOPresets.balanced, 100)),
  new Bloom(BloomPresets.natural),
  new Tonemap(TonemapPresets.filmic),
]);
```

---

## 8. Before these become public API

Presets are the hardest thing to change later — every one you ship is a value someone's
screenshot depends on. Three things worth settling first:

1. **Validate each preset against one reference scene at 720p, 1080p, and 4K.** The `mips`
   clamp (§4) means bloom presets are not resolution-invariant. You want to have seen that
   before a user reports it.
2. **Confirm `whitePoint` actually does something under `aces` and `agx`.** If those curves
   ignore it, the field is misleading in four of five Tonemap presets and should either be
   documented as reinhard-only or wired into the ACES shoulder.
3. **Decide whether `faithful` should bypass the HDR intermediate entirely.** Running
   `curve: 'none'` still round-trips authored colors through `rgba16float` and the linear→sRGB
   encode. That is lossless in theory, but it is worth one test with known sRGB inputs
   confirming the values come out bit-identical — because for a charting library, "close
   enough" is not.

Written to [post-presets.md](null-graph/docs/post-presets.md).
