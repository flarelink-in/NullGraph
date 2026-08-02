// src/extras/post/SSAO.ts
import type { DeclareContext, EffectDeclaration } from './Effect';
import type { ParamSchema } from './types';
import { PostEffectBase } from './PostEffectBase';

export interface SSAOOptions {
    radius?: number;
    bias?: number;
    intensity?: number;
    /** Runtime sample count, clamped to the compile-time kernel size (32). */
    samples?: number;
}

const KERNEL_SIZE = 32;

const SCHEMA: ParamSchema = [
    { name: 'radius', type: 'f32', default: 0.6 },
    { name: 'bias', type: 'f32', default: 0.02 },
    { name: 'intensity', type: 'f32', default: 0.9 },
    { name: 'samples', type: 'u32', default: 16 },
];

/** Deterministic PRNG so the kernel is identical across compiles (and across
 *  main thread / worker, which matters for reproducing bugs). */
function mulberry32(seed: number): () => number {
    return () => {
        seed |= 0;
        seed = (seed + 0x6D2B79F5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/** Hemisphere sample set (z >= 0), scaled to cluster near the origin. Emitted
 *  as a WGSL const array — it never changes, so it does not belong in the arena. */
function buildKernelWGSL(): string {
    const rnd = mulberry32(1337);
    const rows: string[] = [];
    for (let i = 0; i < KERNEL_SIZE; i++) {
        let x = 0, y = 0, z = 0, len = 0;
        do {
            x = rnd() * 2 - 1;
            y = rnd() * 2 - 1;
            z = rnd();
            len = Math.hypot(x, y, z);
        } while (len < 1e-4 || len > 1.0);
        const t = i / KERNEL_SIZE;
        const s = (0.1 + 0.9 * t * t) * (0.3 + 0.7 * rnd()) / len;
        rows.push(`    vec3f(${(x * s).toFixed(6)}, ${(y * s).toFixed(6)}, ${(z * s).toFixed(6)})`);
    }
    return `const SSAO_KERNEL = array<vec3f, ${KERNEL_SIZE}>(\n${rows.join(',\n')}\n);`;
}

// Note (1.0 - uv.y): undoing the composer's Y flip. Getting this wrong tilts
// the occlusion vertically, which reads as "the lighting looks slightly off".
const OCCLUDE = `${buildKernelWGSL()}

fn viewPosFromDepth(uv: vec2f, depth: f32) -> vec3f {
    let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let v = camera.invProj * ndc;
    return v.xyz / v.w;
}

fn effect(uv: vec2f) -> vec4f {
    let depth = sampleInput0(uv);
    // Derivatives must run in uniform control flow: reconstruct before any
    // branch, clamp depth so the far plane doesn't divide by ~0.
    let P = viewPosFromDepth(uv, min(depth, 0.99999));
    var N = normalize(cross(dpdx(P), dpdy(P)));
    if (dot(N, P) > 0.0) { N = -N; }

    // Interleaved gradient noise -> per-pixel kernel rotation.
    let px = uv * passInfo.resolution;
    let angle = 6.2831853 * fract(52.9829189 * fract(dot(px, vec2f(0.06711056, 0.00583715))));
    let dir = vec3f(cos(angle), sin(angle), 0.0);
    let t = normalize(dir - N * dot(dir, N) + vec3f(1e-5));
    let tbn = mat3x3f(t, cross(N, t), N);

    let n = min(params.samples, ${KERNEL_SIZE}u);
    var occlusion = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        let samplePos = P + (tbn * SSAO_KERNEL[i]) * params.radius;
        let proj = camera.proj * vec4f(samplePos, 1.0);
        if (proj.w <= 0.0) { continue; }
        let sndc = proj.xyz / proj.w;
        let suv = vec2f(sndc.x * 0.5 + 0.5, 0.5 - sndc.y * 0.5);
        if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) { continue; }
        let occluderZ = viewPosFromDepth(suv, min(sampleInput0(suv), 0.99999)).z;
        // View space is right-handed: larger (less negative) z is closer.
        if (occluderZ >= samplePos.z + params.bias) {
            occlusion += smoothstep(0.0, 1.0, params.radius / max(abs(P.z - occluderZ), 1e-4));
        }
    }
    var ao = 1.0 - occlusion / f32(max(n, 1u));
    if (depth >= 1.0) { ao = 1.0; }
    ao = clamp(ao, 0.0, 1.0);
    return vec4f(ao, ao, ao, 1.0);
}`;

// 4x4 box blur, offset half a texel so the 16 taps straddle texel centers.
const BLUR = `fn effect(uv: vec2f) -> vec4f {
    var sum = 0.0;
    for (var x = -1; x <= 2; x = x + 1) {
        for (var y = -1; y <= 2; y = y + 1) {
            let offs = (vec2f(f32(x), f32(y)) - vec2f(0.5)) * passInfo.texelSize;
            sum += sampleInput0(uv + offs).r;
        }
    }
    let ao = sum / 16.0;
    return vec4f(ao, ao, ao, 1.0);
}`;

const APPLY = `fn effect(uv: vec2f) -> vec4f {
    let c = sampleInput0(uv);
    let ao = sampleInput1(uv).r;
    return vec4f(c.rgb * mix(1.0, ao, params.intensity), c.a);
}`;

/**
 * Three passes: occlude (half res, r8unorm), blur, apply (full res).
 * Requires gbuffer.depth — the pool's samplable scene depth, NOT
 * ctx.depthTexture (which is RENDER_ATTACHMENT only). Normals are
 * reconstructed from depth derivatives in v1 (faceted on curved surfaces;
 * a normal g-buffer is v2).
 */
export class SSAO extends PostEffectBase {
    public readonly name = 'SSAO';

    constructor(opts: SSAOOptions = {}) {
        super();
        this.initParams(SCHEMA, {
            radius: opts.radius,
            bias: opts.bias,
            intensity: opts.intensity,
            samples: opts.samples,
        });
    }

    public get radius(): number { return this.getParam('radius'); }
    public set radius(v: number) { this.setParam('radius', v); }
    public get bias(): number { return this.getParam('bias'); }
    public set bias(v: number) { this.setParam('bias', v); }
    public get intensity(): number { return this.getParam('intensity'); }
    public set intensity(v: number) { this.setParam('intensity', v); }
    public get samples(): number { return this.getParam('samples'); }
    public set samples(v: number) { this.setParam('samples', Math.max(1, Math.min(KERNEL_SIZE, Math.floor(v)))); }

    public declare(ctx: DeclareContext): EffectDeclaration {
        const aoRaw = ctx.target({ scale: 0.5, format: 'r8unorm' }, 'aoRaw');
        const ao = ctx.target({ scale: 0.5, format: 'r8unorm' }, 'ao');

        ctx.pass({ name: 'SSAO occlude', shaderCode: OCCLUDE, inputs: ['depth'], output: aoRaw });
        ctx.pass({ name: 'SSAO blur', shaderCode: BLUR, inputs: [aoRaw], output: ao });
        ctx.pass({ name: 'SSAO apply', shaderCode: APPLY, inputs: ['chain', ao], output: 'chain' });
        return { params: this.schema };
    }
}
