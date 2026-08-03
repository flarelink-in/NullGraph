// src/extras/post/chainOrder.ts
import type { Effect } from './Effect';

/**
 * Canonical stage order for the stock effects, by `Effect.name`. Tonemap is
 * the pivot: everything before it operates on linear HDR (occlusion, exposure,
 * light scatter, grade); everything after works on the display-referred image
 * (stylization, edge AA, lens/film artifacts). See docs/post-presets.md §2 for
 * the reasoning behind each adjacency.
 */
export const EFFECT_ORDER: readonly string[] = [
    // ---- pre-Tonemap: linear HDR ----
    'SSAO',                 // darken ambient before anything decides "bright"
    'AutoExposure',         // normalize luminance the bloom threshold sees
    'Bokeh',                // depth of field on scene radiance
    'Bloom',                // light scatter
    'Halation',
    'LensFlare',
    'ColorGrade',           // creative grade, still linear
    // ---- the boundary ----
    'Tonemap',              // linear HDR -> display range
    // ---- post-Tonemap: display-referred ----
    'CelShade',             // band the final tones
    'Kuwahara',             // painterly flattening
    'Outline',              // ink on the flat colors
    'FXAA',                 // smooth edges (incl. ink) last among AA
    'ChromaticAberration',  // lens fringing
    'Vignette',             // lens falloff
    'FilmGrain',            // film grain — always last, so nothing smooths it
];

const RANK = new Map(EFFECT_ORDER.map((name, i) => [name, i]));

/**
 * Sort a set of effects into the canonical pipeline order. Recognized built-in
 * effects move to their stage; any effect whose `name` is not in EFFECT_ORDER
 * (e.g. a `CustomEffect`) is treated as unranked and kept, in its original
 * relative order, at the END — position those yourself if they belong earlier.
 *
 * Stable and non-mutating: returns a new array. Does not construct or clone
 * effects, so it is safe to pass the result straight to `PostChain.setChain()`.
 */
export function orderChain(effects: Effect[]): Effect[] {
    const rankOf = (e: Effect) => RANK.get(e.name) ?? EFFECT_ORDER.length;
    return effects
        .map((effect, index) => ({ effect, index, rank: rankOf(effect) }))
        .sort((a, b) => (a.rank - b.rank) || (a.index - b.index))
        .map(entry => entry.effect);
}
