// Celestial Torii & Split-Comet Sakura Shrine — Makoto Shinkai / Ufotable-inspired anime art scene.
// Combines a 3D raymarched toon-shaded sanctuary (Vermilion Water-Torii, Sacred Sakura Tree, stone
// lanterns, floating cliff shrines, mirror-rippling twilight sea, and split comet sky) writing exact
// hardware depth (`@builtin(frag_depth)`) with 65,536 GPU-simulated 3D sakura blossoms, floating
// paper lanterns, and intertwined Musubi spirit-dragon streams, processed through NullGraph's
// Anime PostChain (Bloom + Halation + Tonemap + FXAA + pure 3D Depth Ink Outline).
import { hash } from 'node:crypto';
import { Camera } from '../../../dist/index.js';
import { PostChain, Bloom, Halation, Tonemap, Outline, FXAA, Vignette } from '../../../dist/post.js';

const CAMERA = `struct Camera {
    viewProj: mat4x4f, view: mat4x4f, proj: mat4x4f, invViewProj: mat4x4f,
    invView: mat4x4f, invProj: mat4x4f, eye: vec3f, time: f32,
    near: f32, far: f32, fov: f32, aspect: f32, pad: vec4f,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> particles: array<vec4f>;
`;

export const EVOLVE = `
@group(0) @binding(0) var<storage, read_write> particles: array<vec4f>;
@compute @workgroup_size(128) fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i * 2u >= arrayLength(&particles)) { return; }
    var a = particles[i * 2u];
    let b = particles[i * 2u + 1u];
    let dt = 1.0 / 60.0;

    let t = a.w + dt;

    // Population 1 (b.w < 0.07): ~4,600 hero 3D-tumbling Sakura blossom petals & detached leaves in the wind.
    if (b.w < 0.07) {
        let speed = 0.065 + 0.035 * fract(b.x * 7.13);
        let s = fract(b.z + t * speed);

        // Wind stream sweeping from the Sacred Tree boughs (z = -1.65, y = 2.8)
        // forward through the Vermilion Torii gate and out over the mirror water.
        let spread = (b.x - 0.5) * 3.6;
        let curveX = 1.45 * sin(s * 5.0 - t * 0.6) + spread * cos(s * 2.6 + b.y * 3.14159);
        let curveZ = mix(-2.1, 4.8, s) + 0.35 * sin(s * 6.283 + b.x * 6.283);
        let curveY = mix(2.95, -1.25, pow(s, 0.85)) + b.y * 0.55 + 0.16 * sin(t * 2.4 + b.x * 19.0);

        let p = vec3f(curveX, max(curveY, -1.52), curveZ);
        particles[i * 2u] = vec4f(p, t);
        return;
    }

    // Population 2 (0.07 <= b.w < 0.36): ~19,000 real 3D anime leaf & blossom sprigs rustling on the 11 tree bush pads!
    if (b.w < 0.36) {
        let padIdx = u32(clamp(floor(b.z * 11.0), 0.0, 10.0));
        let centers = array<vec3f, 11>(
            vec3f( 0.00, 3.32, -1.60),
            vec3f(-1.25, 2.95, -1.35),
            vec3f( 1.25, 2.95, -1.40),
            vec3f(-2.35, 2.42, -1.25),
            vec3f( 2.35, 2.42, -1.30),
            vec3f(-0.75, 2.52, -0.95),
            vec3f( 0.80, 2.52, -0.98),
            vec3f(-1.75, 2.02, -1.05),
            vec3f( 1.75, 2.02, -1.08),
            vec3f(-2.95, 1.92, -1.45),
            vec3f( 2.95, 1.92, -1.48)
        );
        let radii = array<vec3f, 11>(
            vec3f(0.95, 0.44, 0.72),
            vec3f(0.88, 0.40, 0.68),
            vec3f(0.88, 0.40, 0.68),
            vec3f(0.86, 0.38, 0.65),
            vec3f(0.86, 0.38, 0.65),
            vec3f(0.76, 0.35, 0.58),
            vec3f(0.76, 0.35, 0.58),
            vec3f(0.78, 0.34, 0.58),
            vec3f(0.78, 0.34, 0.58),
            vec3f(0.70, 0.32, 0.54),
            vec3f(0.70, 0.32, 0.54)
        );
        let c = centers[padIdx];
        let r = radii[padIdx];

        // Distribute 3D leaf sprigs across the upper/outer shell of each bush pad
        let theta = fract(b.x * 17.31 + b.z * 5.17) * 6.2831853;
        let cosPhi = b.y * 0.75 + 0.15;
        let sinPhi = sqrt(max(1.0 - cosPhi * cosPhi, 0.0));
        let shell = 0.78 + 0.28 * fract(b.x * 43.7);
        let localDir = vec3f(sinPhi * cos(theta), cosPhi, sinPhi * sin(theta));
        let rustle = vec3f(
            0.025 * sin(t * 3.2 + b.x * 29.0 + c.x * 2.0),
            0.018 * cos(t * 2.7 + b.z * 31.0),
            0.020 * sin(t * 2.9 + b.y * 23.0)
        );
        let p = c + localDir * r * shell + rustle;
        particles[i * 2u] = vec4f(p, t);
        return;
    }

    // Population 3 (0.36 <= b.w < 0.82): ~30,100 particles forming twin intertwined Musubi Spirit-Dragon streams!
    if (b.w < 0.82) {
        let s = fract(b.z + t * 0.075); // [0, 1) along the ascending dragon path
        let isRoseGold = b.w >= 0.59;

        // Sinuous 3D calligraphic S-curve:
        // s=0.0: rises from foreground mirror water (x=1.55, y=-1.35, z=4.5)
        // s=0.30: threads the Vermilion Torii gate opening (x=0.0, y=0.38, z=1.85)
        // s=0.62: wraps around the outer-left silhouette of the tree bushes (x=-3.55, y=2.55, z=-0.45)
        // s=0.95: coils high into the starry sky toward the Split Comet (x=-1.35, y=4.65, z=-2.45)
        let zCenter = mix(4.5, -2.55, pow(s, 0.90));
        let yCenter = -1.35 + 6.15 * pow(s, 1.05) + 0.28 * sin(s * 6.2831853);
        let xCenter = 1.65 * cos(s * 5.0) - 2.05 * pow(sin(s * 3.14159265), 1.6);

        // Coherent double-helix braid tapering at both ends like an anime brush stroke
        let strandOffset = select(0.0, 3.14159265, isRoseGold);
        let helixAngle = s * 16.5 - t * 2.8 + strandOffset;
        let endTaper = smoothstep(0.0, 0.08, s) * smoothstep(1.0, 0.88, s);
        let helixRadius = (0.36 + 0.10 * sin(s * 6.2831853)) * (0.25 + 0.75 * endTaper);
        let coreSpread = pow(b.x, 1.4) * 0.058 * (0.4 + 0.6 * endTaper);
        let jitterAngle = b.y * 3.14159265;
        let ribbonOffset = vec3f(
            helixRadius * cos(helixAngle) + coreSpread * cos(jitterAngle),
            helixRadius * sin(helixAngle) * 0.68 + coreSpread * sin(jitterAngle),
            0.16 * sin(helixAngle)
        );
        let p = vec3f(xCenter, yCenter, zCenter) + ribbonOffset;
        particles[i * 2u] = vec4f(p, t);
        return;
    }

    // Population 4 (b.w >= 0.82): Floating Sky Lanterns (b.w >= 0.94) & Twilight Water Fireflies.
    let rise = fract(b.z + t * (0.015 + 0.010 * b.x));
    let ringR = 2.5 + pow(b.x, 0.75) * 7.2;
    let theta = fract(b.z * 13.37) * 6.2831853 + t * 0.08 * select(-1.0, 1.0, b.y >= 0.0);
    let sway = 0.18 * sin(t * 1.6 + b.x * 13.0);
    let maxH = select(1.4, 5.2, b.w >= 0.94);
    let yLantern = -1.54 + pow(rise, 1.12) * maxH;
    let p = vec3f(ringR * cos(theta) + sway, yLantern, ringR * sin(theta) * 0.85 - 0.5);
    particles[i * 2u] = vec4f(p, t);
}`;

const BACKDROP = CAMERA + `
struct V {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

struct FragOut {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
};

@vertex fn vs_main(@builtin(vertex_index) i: u32) -> V {
    let p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    var o: V;
    o.pos = vec4f(p[i], 0.9999, 1.0);
    o.uv = p[i];
    return o;
}

fn hash3(p: vec3f) -> f32 {
    var q = fract(p * vec3f(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 33.33);
    return fract((q.x + q.y) * q.z);
}

fn noise3(p: vec3f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash3(i + vec3f(0,0,0)), hash3(i + vec3f(1,0,0)), u.x),
            mix(hash3(i + vec3f(0,1,0)), hash3(i + vec3f(1,1,0)), u.x), u.y),
        mix(mix(hash3(i + vec3f(0,0,1)), hash3(i + vec3f(1,0,1)), u.x),
            mix(hash3(i + vec3f(0,1,1)), hash3(i + vec3f(1,1,1)), u.x), u.y),
        u.z
    );
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

fn sdBox(p: vec3f, b: vec3f) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdCapsule(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

fn sdBushPad(p: vec3f, c: vec3f, r: vec3f) -> f32 {
    let q = (p - c) / r;
    return (length(q) - 1.0) * min(r.x, min(r.y, r.z));
}

// Returns vec2f(signedDistance, materialID):
// 1.0 = Vermilion Torii wood
// 2.0 = Dark slate Torii roof / pillar base / stone lantern frame
// 3.0 = Sanctuary stone island & stepped causeway
// 4.0 = Gnarled Sacred Shrine Tree trunk & multi-tiered boughs
// 5.0 = Layered Anime Foliage Bush Pads (individual depth-inked leafy bush clusters!)
// 6.0 = Incandescent golden stone-lantern core & Torii emblem
fn mapScene(p: vec3f) -> vec2f {
    var res = vec2f(1e5, 0.0);

    // --- 1. Grand Vermilion Water-Torii at z = 1.85 ---
    let tp = p - vec3f(0.0, 0.0, 1.85);
    let symX = vec3f(abs(tp.x), tp.y, tp.z);

    let pillarA = vec3f(1.38, -1.70, 0.0);
    let pillarB = vec3f(1.28, 1.72, 0.0);
    let dPillar = sdCapsule(symX, pillarA, pillarB, 0.13);

    // Secondary outer sanctuary gate posts (Ryōbu Torii outriggers)
    let dOutPillar = sdCapsule(symX, vec3f(1.38, -1.70, 0.55), vec3f(1.38, -0.35, 0.55), 0.075);
    let dOutBeam = sdBox(vec3f(symX.x - 1.38, tp.y + 0.65, abs(tp.z) - 0.28), vec3f(0.055, 0.055, 0.32));

    let dNuki = sdBox(tp - vec3f(0.0, 1.02, 0.0), vec3f(1.82, 0.08, 0.09)) - 0.015;
    let archCurve = 0.15 * pow(clamp(tp.x / 2.25, -1.2, 1.2), 2.0);
    let dShimaki = sdBox(tp - vec3f(0.0, 1.56 + archCurve * 0.75, 0.0), vec3f(2.05, 0.075, 0.115)) - 0.015;
    let dStrut = sdBox(tp - vec3f(0.0, 1.30, 0.0), vec3f(0.085, 0.24, 0.075));
    let dToriiRed = min(min(min(dPillar, dNuki), min(dShimaki, dStrut)), min(dOutPillar, dOutBeam));
    if (dToriiRed < res.x) { res = vec2f(dToriiRed, 1.0); }

    let dKasagi = sdBox(tp - vec3f(0.0, 1.70 + archCurve, 0.0), vec3f(2.28, 0.09, 0.16)) - 0.02;
    let dNemaki = sdCapsule(symX, vec3f(1.38, -1.70, 0.0), vec3f(1.35, -0.92, 0.0), 0.16);
    let dOutCap = sdBox(vec3f(symX.x - 1.38, tp.y + 0.30, abs(tp.z) - 0.55), vec3f(0.11, 0.045, 0.11));
    let dToriiDark = min(min(dKasagi, dNemaki), dOutCap);
    if (dToriiDark < res.x) { res = vec2f(dToriiDark, 2.0); }

    let dPlaque = sdBox(tp - vec3f(0.0, 1.30, 0.085), vec3f(0.12, 0.16, 0.025));
    if (dPlaque < res.x) { res = vec2f(dPlaque, 6.0); }

    // --- 2. Sanctuary Stone Island & Stepped Causeway ---
    let ip = p - vec3f(0.0, -1.35, -1.65);
    let rIsland = length(ip.xz);
    let dMainIsland = max(rIsland - 3.35 + ip.y * 0.38, abs(ip.y) - 0.42);
    let dUpperDais = max(rIsland - 2.10, abs(ip.y - 0.44) - 0.20);
    let stepZ = clamp(round((p.z - 0.05) / 0.58), -1.0, 2.0);
    let dSteps = sdBox(p - vec3f(0.0, -1.52, 0.05 + stepZ * 0.58), vec3f(0.58, 0.11, 0.19)) - 0.025;

    let dStone = min(smin(dMainIsland, dUpperDais, 0.16), dSteps);
    if (dStone < res.x) { res = vec2f(dStone, 3.0); }

    // Carved Stone Lanterns (Tōrō) on the sanctuary perimeter & water approach
    let lp = vec3f(abs(p.x) - 1.95, p.y + 0.65, abs(p.z + 0.90) - 0.95);
    let dLanternPost = sdBox(lp - vec3f(0.0, -0.15, 0.0), vec3f(0.10, 0.38, 0.10));
    let dLanternRoof = sdBox(lp - vec3f(0.0, 0.42, 0.0), vec3f(0.27, 0.075, 0.27)) - 0.02;
    let dLanternBase = sdBox(lp - vec3f(0.0, -0.45, 0.0), vec3f(0.20, 0.07, 0.20));
    let dLanternStone = min(min(dLanternPost, dLanternRoof), dLanternBase);
    if (dLanternStone < res.x) { res = vec2f(dLanternStone, 2.0); }

    let dLanternLight = sdBox(lp - vec3f(0.0, 0.22, 0.0), vec3f(0.13, 0.13, 0.13));
    if (dLanternLight < res.x) { res = vec2f(dLanternLight, 6.0); }

    // --- 3. Ancient Sacred Shrine Tree: Gnarled Asymmetric Boughs & Billowing Leafy Bush Clusters ---
    let sp = p - vec3f(0.0, -0.85, -1.65);
    if (length(sp - vec3f(0.0, 2.3, 0.0)) < 5.0) {
        let twistX = 0.24 * sin(sp.y * 1.35 + 0.2);
        let dTrunk = sdCapsule(sp - vec3f(twistX, 0.0, 0.0), vec3f(0.0, -0.25, 0.0), vec3f(-0.08, 2.30, -0.05), 0.36 - 0.08 * clamp(sp.y, 0.0, 2.4));
        let spSym = vec3f(abs(sp.x), sp.y, sp.z);
        let dRoot = sdCapsule(spSym, vec3f(0.18, -0.15, 0.0), vec3f(1.05, -0.55, 0.48), 0.14);

        // Spreading primary & secondary wooden boughs visible between the foliage bush clusters
        let dBoughL = sdCapsule(sp, vec3f(-0.12, 1.20, 0.0), vec3f(-2.15, 2.45, 0.22), 0.15);
        let dTwigL  = sdCapsule(sp, vec3f(-1.65, 2.15, 0.15), vec3f(-2.95, 2.62, 0.10), 0.10);
        let dBoughR = sdCapsule(sp, vec3f( 0.14, 1.30, 0.0), vec3f( 1.95, 2.58, 0.18), 0.14);
        let dTwigR  = sdCapsule(sp, vec3f( 1.50, 2.25, 0.12), vec3f( 2.75, 2.48, 0.25), 0.10);
        let dBoughMidL = sdCapsule(sp, vec3f(-0.10, 1.75, -0.05), vec3f(-1.25, 3.45, 0.18), 0.12);
        let dBoughMidR = sdCapsule(sp, vec3f( 0.10, 1.80, -0.05), vec3f( 1.18, 3.35, 0.22), 0.12);
        let dCrownFork = sdCapsule(sp, vec3f(-0.05, 2.15, -0.05), vec3f( 0.15, 3.85, -0.10), 0.11);

        let dBoughs = min(
            min(smin(dBoughL, dTwigL, 0.08), smin(dBoughR, dTwigR, 0.08)),
            min(min(dBoughMidL, dBoughMidR), dCrownFork)
        );
        let dTreeBark = smin(smin(dTrunk, dRoot, 0.18), dBoughs, 0.15);
        if (dTreeBark < res.x) { res = vec2f(dTreeBark, 4.0); }

        // Multi-tiered, staggered-depth anime bush clusters (organic asymmetric arrangement so
        // hardware depth Outline inks crisp overlapping bush contours & leaves branch gaps open!)
        let bp = p + vec3f(
            (noise3(p * 3.6) - 0.5) * 0.22,
            (noise3(p * 4.2 + vec3f(3.1)) - 0.5) * 0.18,
            (noise3(p * 3.6 + vec3f(7.3)) - 0.5) * 0.22
        );

        // Top crown bush cluster
        var dBushes = sdBushPad(bp, vec3f(-0.10, 3.38, -1.62), vec3f(0.88, 0.48, 0.72));
        dBushes = smin(dBushes, sdBushPad(bp, vec3f(0.55, 3.18, -1.45), vec3f(0.68, 0.40, 0.58)), 0.08);

        // Upper-left & upper-right bush clusters (at distinct z depths for crisp internal ink lines!)
        let dBushUL = smin(
            sdBushPad(bp, vec3f(-1.28, 2.95, -1.32), vec3f(0.82, 0.44, 0.65)),
            sdBushPad(bp, vec3f(-0.68, 2.72, -1.02), vec3f(0.64, 0.36, 0.52)),
            0.07
        );
        let dBushUR = smin(
            sdBushPad(bp, vec3f( 1.22, 2.88, -1.38), vec3f(0.80, 0.43, 0.64)),
            sdBushPad(bp, vec3f( 0.72, 2.60, -0.98), vec3f(0.66, 0.36, 0.52)),
            0.07
        );
        dBushes = min(dBushes, min(dBushUL, dBushUR));

        // Mid-left & mid-right sweeping bough bushes
        let dBushML = smin(
            sdBushPad(bp, vec3f(-2.28, 2.42, -1.22), vec3f(0.82, 0.42, 0.62)),
            sdBushPad(bp, vec3f(-1.68, 2.08, -0.96), vec3f(0.68, 0.36, 0.54)),
            0.07
        );
        let dBushMR = smin(
            sdBushPad(bp, vec3f( 2.18, 2.35, -1.26), vec3f(0.78, 0.40, 0.60)),
            sdBushPad(bp, vec3f( 1.65, 2.02, -1.02), vec3f(0.68, 0.35, 0.54)),
            0.07
        );
        dBushes = min(dBushes, min(dBushML, dBushMR));

        // Outer-lower overhanging foliage pads + deep background bush tier
        dBushes = min(dBushes, sdBushPad(bp, vec3f(-2.85, 1.98, -1.48), vec3f(0.62, 0.34, 0.52)));
        dBushes = min(dBushes, sdBushPad(bp, vec3f( 2.72, 1.92, -1.42), vec3f(0.60, 0.33, 0.50)));
        dBushes = min(dBushes, sdBushPad(bp, vec3f(-0.85, 2.85, -2.18), vec3f(0.85, 0.44, 0.62)));
        dBushes = min(dBushes, sdBushPad(bp, vec3f( 0.95, 2.75, -2.12), vec3f(0.82, 0.42, 0.60)));

        // Fine pointed leaf-tuft detail on bush surfaces
        if (dBushes < 0.35) {
            let tuft = (noise3(p * 9.5) - 0.5) * 0.085 + (noise3(p * 19.0) - 0.5) * 0.038;
            dBushes += tuft;
        }
        if (dBushes < res.x) { res = vec2f(dBushes, 5.0); }
    }

    return res;
}

fn calcNormal(p: vec3f) -> vec3f {
    let e = vec2f(0.0035, -0.0035);
    return normalize(
        e.xyy * mapScene(p + e.xyy).x +
        e.yyx * mapScene(p + e.yyx).x +
        e.yxy * mapScene(p + e.yxy).x +
        e.xxx * mapScene(p + e.xxx).x
    );
}

// Makoto Shinkai Twilight Sky + Towering Cel-Shaded Cumulonimbus + The Great Split Comet
fn sampleAnimeSky(rd: vec3f) -> vec3f {
    let y = clamp(rd.y, 0.0, 1.0);

    // Calibrated for low-horizon camera (visible sky spans rd.y in [0.0, 0.32]):
    // Warm amber-gold horizon -> vivid twilight magenta -> deep royal indigo -> starry midnight sapphire!
    let zenithCol = vec3f(0.006, 0.018, 0.095);
    let upperCol = vec3f(0.024, 0.032, 0.165);
    let midCol = vec3f(0.24, 0.055, 0.20);
    let horizonCol = vec3f(0.76, 0.28, 0.10);

    var sky = mix(horizonCol, midCol, smoothstep(0.0, 0.055, y));
    sky = mix(sky, upperCol, smoothstep(0.045, 0.135, y));
    sky = mix(sky, zenithCol, smoothstep(0.115, 0.24, y));

    // Golden-hour sunset glow behind the sanctuary on the low horizon
    let sunDir = normalize(vec3f(0.22, 0.025, -0.975));
    let sunDot = max(dot(rd, sunDir), 0.0);
    sky += vec3f(0.70, 0.24, 0.07) * pow(sunDot, 7.0) * exp(-y * 12.0) * 0.75;

    // Crisp twinkling stars & Milky Way dust in the upper midnight-cobalt sky (y > 0.07)
    if (y > 0.07) {
        let starFade = smoothstep(0.07, 0.16, y);
        let starCell = floor(rd * 265.0);
        let starFrac = fract(rd * 265.0) - 0.5;
        let sr = hash3(starCell);
        if (sr > 0.988) {
            let sBright = exp(-dot(starFrac, starFrac) * 30.0) * starFade;
            let sCol = mix(vec3f(1.1, 0.88, 0.70), vec3f(0.55, 0.92, 1.45), fract(sr * 31.0));
            sky += sCol * sBright * 1.15;
        }
    }

    // --- Towering Cel-Shaded Anime Cumulonimbus Clouds along the horizon (rd.y < 0.16) ---
    if (rd.y > -0.01 && rd.y < 0.17) {
        let cloudUV = vec2f(atan2(rd.x, -rd.z) * 3.0, rd.y * 10.5);
        let n1 = noise3(vec3f(cloudUV * 1.8, 1.2));
        let n2 = noise3(vec3f(cloudUV * 3.9 + vec2f(4.1, -2.3), 2.7));
        let n3 = noise3(vec3f(cloudUV * 8.2, 5.4));
        let billow = n1 * 0.58 + n2 * 0.28 + n3 * 0.14;
        let cloudHeightMask = smoothstep(0.165, 0.02, rd.y) * smoothstep(-0.01, 0.018, rd.y);
        let cloudDensity = billow * cloudHeightMask;

        if (cloudDensity > 0.26) {
            let nUp = noise3(vec3f((cloudUV + vec2f(0.04, 0.09)) * 2.4, 1.2));
            let cloudLight = clamp((nUp - billow) * 4.5 + 0.52 + 0.35 * sunDot, 0.0, 1.0);
            let shadowCloud = vec3f(0.035, 0.03, 0.11);
            let midCloud = vec3f(0.26, 0.08, 0.18);
            let litCloud = vec3f(0.84, 0.38, 0.16);

            var cCol = mix(shadowCloud, midCloud, smoothstep(0.38, 0.48, cloudLight));
            cCol = mix(cCol, litCloud, smoothstep(0.68, 0.78, cloudLight));
            let edgeAlpha = smoothstep(0.26, 0.31, cloudDensity);
            sky = mix(sky, cCol, edgeAlpha * 0.92);
        }
    }

    // --- The Great Split Comet (Kimi no Na wa / Tiamat Comet) in the upper-left indigo sky ---
    let cometForward = normalize(vec3f(-0.18, 0.21, -0.96));
    let cometRight = normalize(vec3f(0.92, 0.28, -0.10));
    let cometUp = cross(cometForward, cometRight);
    let cu = dot(rd, cometRight);
    let cv = dot(rd, cometUp);
    let cw = dot(rd, cometForward);

    if (cw > 0.15) {
        let cwFade = smoothstep(0.15, 0.45, cw);
        let dx = cu + 0.12;
        let dy = cv - 0.01;
        let headDist = length(vec2f(dx, dy));

        // Blazing aquamarine-white nucleus + tight coma
        let nucleus = exp(-headDist * 135.0) * 3.2 + exp(-headDist * 28.0) * 0.55;
        sky += vec3f(0.45, 1.35, 2.15) * nucleus * cwFade;

        if (dx > -0.01) {
            let tailT = max(dx * 1.45, 0.0);
            // Upper curved rose-magenta & gold dust tail (smooth parabolic anime brush fan)
            let curveDust = dy - (0.018 * tailT + 0.052 * tailT * tailT);
            let widthDust = 0.0055 + 0.028 * tailT;
            let dustCore = exp(-pow(curveDust / widthDust, 2.0));
            let dustWisp = exp(-pow((curveDust - 0.006 * tailT) / (widthDust * 0.52), 2.0)) * 0.45;
            let dustMask = (dustCore + dustWisp) * exp(-tailT * 2.55) * smoothstep(-0.008, 0.022, dx);
            let dustCol = mix(vec3f(0.95, 1.15, 1.85), vec3f(1.25, 0.36, 0.72), smoothstep(0.0, 0.48, tailT));
            sky += dustCol * dustMask * 1.05 * cwFade;

            // Lower razor-sharp electric-cyan ion tail
            let curveIon = dy + 0.011 * tailT;
            let widthIon = 0.0025 + 0.0080 * tailT;
            let ionMask = exp(-pow(curveIon / widthIon, 2.0)) * exp(-tailT * 2.75) * smoothstep(-0.008, 0.018, dx);
            sky += vec3f(0.22, 1.25, 2.15) * ionMask * 1.35 * cwFade;

            // Split secondary crimson-gold fragment streaking below the main nucleus
            let dx2 = cu + 0.04;
            let dy2 = cv + 0.022;
            let fragT = max(dx2 * 1.65, 0.0);
            let fragHead = exp(-length(vec2f(dx2, dy2)) * 165.0) * 2.5;
            let curveFrag = dy2 - (0.016 * fragT + 0.028 * fragT * fragT);
            let fragTail = exp(-pow(curveFrag / (0.0042 + 0.008 * fragT), 2.0)) * exp(-fragT * 3.8) * smoothstep(-0.005, 0.018, dx2);
            sky += vec3f(1.65, 0.58, 0.85) * (fragHead + fragTail * 0.95) * cwFade;
        }
    }

    return sky;
}

// 3-Band Stepped Anime Toon Shading for 3D Sanctuary Architecture & Sacred Shrine Tree
fn shadeAnimeSurface(p: vec3f, n: vec3f, rd: vec3f, matId: f32) -> vec3f {
    let viewDir = -rd;
    let keyLightDir = normalize(vec3f(-0.38, 0.65, -0.66));
    let rimLightDir = normalize(vec3f(0.45, 0.38, -0.81));

    // Crisp 3-band anime cel-shading ramp on 3D surfaces
    let ndl = dot(n, keyLightDir) * 0.5 + 0.5;
    var toonStep = smoothstep(0.38, 0.44, ndl) * 0.48 + smoothstep(0.70, 0.76, ndl) * 0.42 + 0.12;

    // Warm point-light bounce from the sanctuary stone lanterns & Torii plaque
    let lanternPos = vec3f(sign(p.x) * 1.95, -0.43, sign(p.z + 0.90) * 0.95 - 0.90);
    let lanternDist = length(p - lanternPos);
    let lanternGlow = exp(-lanternDist * 1.05) * 0.85;

    var baseCol = vec3f(0.4);
    var shadowCol = vec3f(0.05, 0.03, 0.12);
    var rimCol = vec3f(0.30, 0.85, 1.45);

    if (matId < 1.5) {
        // Vermilion Torii wood (vivid anime cinnabar red with deep indigo-plum shadows)
        baseCol = vec3f(0.84, 0.12, 0.09);
        shadowCol = vec3f(0.16, 0.03, 0.08);
        rimCol = vec3f(1.25, 0.55, 0.20);
    } else if (matId < 2.5) {
        // Dark slate roof & stone lantern frame
        baseCol = vec3f(0.12, 0.14, 0.24);
        shadowCol = vec3f(0.03, 0.025, 0.08);
        rimCol = vec3f(0.38, 0.92, 1.45);
    } else if (matId < 3.5) {
        // Sanctuary stone island & stepped causeway
        baseCol = vec3f(0.24, 0.22, 0.36);
        shadowCol = vec3f(0.05, 0.04, 0.12);
        rimCol = vec3f(0.42, 0.82, 1.35);
    } else if (matId < 4.5) {
        // Gnarled Sacred Shrine Tree bark & spreading boughs
        let barkGrain = noise3(vec3f(p.x * 6.0, p.y * 2.5, p.z * 6.0));
        baseCol = mix(vec3f(0.20, 0.10, 0.13), vec3f(0.29, 0.15, 0.18), smoothstep(0.35, 0.65, barkGrain));
        shadowCol = vec3f(0.045, 0.022, 0.07);
        rimCol = vec3f(0.85, 0.48, 0.28);
    } else if (matId < 5.5) {
        // Layered Studio Ghibli / Shinkai Anime Foliage Bushes (rich forest-emerald & jade leaf clusters!)
        let leafMacro = noise3(p * 4.5);
        let leafMicro = noise3(p * 16.0);
        let leafBlade = abs(sin(p.x * 22.0 + cos(p.y * 19.0))) * 0.25;
        let leafCell = leafMacro * 0.52 + leafMicro * 0.33 + leafBlade * 0.15;
        let celLeaf = smoothstep(0.32, 0.38, leafCell) * 0.45 + smoothstep(0.56, 0.63, leafCell) * 0.55;

        let deepFoliage = vec3f(0.012, 0.065, 0.095);
        let darkPine = vec3f(0.032, 0.145, 0.125);
        let midJade = vec3f(0.065, 0.26, 0.155);
        let litEmerald = vec3f(0.15, 0.44, 0.19);
        let sunTipGold = vec3f(0.44, 0.62, 0.17);

        let heightSun = smoothstep(1.9, 3.4, p.y) * smoothstep(0.0, 0.60, n.y);
        baseCol = mix(midJade, litEmerald, celLeaf);
        baseCol = mix(baseCol, sunTipGold, heightSun * smoothstep(0.52, 0.72, leafCell) * 0.78);

        // Ambient occlusion in the deep crevices between bush pads and under boughs
        let bushAO = smoothstep(-0.35, 0.35, n.y);
        shadowCol = mix(deepFoliage, darkPine, celLeaf * bushAO);
        rimCol = vec3f(0.42, 0.76, 0.28);
        toonStep = clamp((toonStep * (0.65 + 0.35 * bushAO)) + (celLeaf - 0.45) * 0.26, 0.04, 1.0);
    } else {
        // Incandescent golden lantern core & Torii plaque (true HDR emissive for Bloom + Halation!)
        return vec3f(2.4, 1.35, 0.42);
    }

    var col = mix(shadowCol, baseCol, toonStep);

    // Crisp Anime Rim Light on grazing silhouette angles
    let fresnel = pow(1.0 - max(dot(n, viewDir), 0.0), 3.4);
    let rimMask = smoothstep(0.42, 0.54, fresnel * (0.55 + 0.45 * max(dot(n, rimLightDir), 0.0)));
    col += rimCol * rimMask * 0.55;

    // Warm amber bounce light from the shrine lanterns
    col += vec3f(0.95, 0.42, 0.12) * lanternGlow * max(dot(n, normalize(lanternPos - p)), 0.2);

    return col;
}

@fragment fn fs_main(v: V) -> FragOut {
    let h = camera.invViewProj * vec4f(v.uv, 1.0, 1.0);
    let ro = camera.eye;
    let rd = normalize(h.xyz / h.w - ro);

    let waterY = -1.65;
    var tWater = 1e5;
    if (rd.y < -1e-4 && ro.y > waterY) {
        tWater = (waterY - ro.y) / rd.y;
    }

    // Raymarch the 3D Sanctuary, Torii, and Sacred Shrine Tree
    var t = 0.2;
    let tMax = min(tWater, 32.0);
    var hitMat = 0.0;
    for (var step = 0; step < 72; step = step + 1) {
        if (t >= tMax) { break; }
        let p = ro + rd * t;
        let d = mapScene(p);
        if (d.x < 0.004) {
            hitMat = d.y;
            break;
        }
        t += d.x * 0.88;
    }

    var finalCol = vec3f(0.0);
    var hitPos = ro + rd * 80.0;
    var wroteSurface = false;

    if (hitMat > 0.5 && t < tMax) {
        hitPos = ro + rd * t;
        let n = calcNormal(hitPos);
        finalCol = shadeAnimeSurface(hitPos, n, rd, hitMat);
        let fog = 1.0 - exp(-max(t - 10.0, 0.0) * 0.022);
        finalCol = mix(finalCol, vec3f(0.18, 0.07, 0.22), fog * 0.40);
        wroteSurface = true;
    } else if (tWater < 48.0) {
        // Mirror-reflective twilight lake at y = -1.65
        hitPos = ro + rd * tWater;
        let wp = hitPos.xz;

        let dToriiWave = min(length(wp - vec2f(-1.38, 1.85)), length(wp - vec2f(1.38, 1.85)));
        let dIslandWave = abs(length(wp - vec2f(0.0, -1.65)) - 3.25);
        let ripple = sin(dToriiWave * 14.0 - camera.time * 4.5) * exp(-dToriiWave * 0.65)
                   + sin(dIslandWave * 10.0 - camera.time * 3.2) * exp(-dIslandWave * 0.45)
                   + 0.30 * sin(wp.x * 3.2 + camera.time * 1.8) * cos(wp.y * 3.8 - camera.time * 1.4);

        let nWater = normalize(vec3f(ripple * 0.024, 1.0, ripple * 0.024));
        let reflRd = normalize(reflect(rd, nWater));

        var reflCol = sampleAnimeSky(reflRd);
        var tr = 0.16;
        for (var rs = 0; rs < 44; rs = rs + 1) {
            if (tr > 22.0) { break; }
            let rp = hitPos + reflRd * tr;
            let rdHit = mapScene(rp);
            if (rdHit.x < 0.012) {
                let rn = calcNormal(rp);
                let surfCol = shadeAnimeSurface(rp, rn, reflRd, rdHit.y);
                reflCol = mix(reflCol, surfCol, smoothstep(22.0, 10.0, tr));
                break;
            }
            tr += rdHit.x * 0.92;
        }

        let waterDeep = vec3f(0.008, 0.020, 0.085);
        let fresnelW = 0.25 + 0.65 * pow(1.0 - max(dot(nWater, -rd), 0.0), 2.8);
        finalCol = mix(waterDeep, reflCol * vec3f(0.84, 0.82, 0.94), fresnelW);

        // Crisp cel-shaded anime foam rings around the Torii water-pillars and island shore
        let foamMask = smoothstep(0.18, 0.13, dToriiWave) + smoothstep(0.22, 0.15, dIslandWave);
        let rippleRing = smoothstep(0.66, 0.72, sin(dToriiWave * 16.0 - camera.time * 4.5) * exp(-dToriiWave * 0.7));
        finalCol = mix(finalCol, vec3f(0.65, 0.60, 0.85), clamp(foamMask * 0.55 + rippleRing * 0.38, 0.0, 0.75));

        // Smooth atmospheric horizon haze on distant water (never slice cloud noise at constant y!)
        let distFade = exp(-max(tWater - 14.0, 0.0) * 0.042);
        let sunDirH = normalize(vec3f(0.22, 0.0, -0.975));
        let sunGlowH = pow(max(dot(normalize(vec3f(rd.x, 0.0, rd.z)), sunDirH), 0.0), 6.0);
        let horizonHaze = vec3f(0.42, 0.15, 0.18) + vec3f(0.45, 0.18, 0.06) * sunGlowH;
        finalCol = mix(horizonHaze, finalCol, distFade);
        wroteSurface = false;
    } else {
        finalCol = sampleAnimeSky(rd);
    }

    var fragDepth = 0.9999;
    if (wroteSurface) {
        let clip = camera.viewProj * vec4f(hitPos, 1.0);
        fragDepth = clamp(clip.z / clip.w, 0.0001, 0.9998);
    }

    var out: FragOut;
    out.color = vec4f(finalCol + particles[0].xyz * 0.0, 1.0);
    out.depth = fragDepth;
    return out;
}`;

const PARTICLES = CAMERA + `
struct V {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec3f,
    @location(2) kind: f32,
};

@vertex fn vs_main(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> V {
    let a = particles[i * 2u];
    let b = particles[i * 2u + 1u];
    let corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    let uv = corners[v];

    var worldPos = a.xyz;
    var color = vec3f(1.0);
    var kind = 0.0;

    if (b.w < 0.07) {
        // Hero 3D-tumbling 5-lobed Sakura blossom petals & wind-blown leaves
        kind = 0.0;
        let tumbleA = a.w * 2.8 + b.x * 19.0;
        let tumbleB = a.w * 2.1 + b.z * 23.0;
        let uDir = normalize(vec3f(cos(tumbleA), sin(tumbleB) * 0.65, sin(tumbleA)));
        let vDir = normalize(vec3f(-sin(tumbleA) * 0.5, cos(tumbleB), cos(tumbleA) * 0.85));
        let normal = normalize(cross(uDir, vDir));

        let petalSize = 0.026 + 0.020 * fract(b.x * 13.7);
        worldPos += (uDir * uv.x + vDir * uv.y) * petalSize;

        let lightDir = normalize(vec3f(-0.3, 0.7, -0.6));
        let ndl = abs(dot(normal, lightDir));
        let toon = select(0.65, 1.05, ndl > 0.52);
        let isGreenLeaf = fract(b.x * 17.1) < 0.35;
        let sakuraPink = mix(vec3f(0.92, 0.28, 0.52), vec3f(1.15, 0.62, 0.82), fract(b.z * 9.3));
        let leafGreen = mix(vec3f(0.14, 0.46, 0.24), vec3f(0.45, 0.68, 0.22), fract(b.z * 9.3));
        color = select(sakuraPink, leafGreen, isGreenLeaf) * toon;
    } else if (b.w < 0.36) {
        // ~19,000 Real 3D Anime Leaf & Blossom Sprigs rustling on the 11 Tree Bush Pads!
        kind = 3.0;
        let flutter = 0.22 * sin(a.w * 3.1 + b.x * 37.0 + b.z * 19.0);
        let yaw = b.x * 6.2831853 + flutter;
        let pitch = (b.y * 0.55) + 0.25;
        let uDir = normalize(vec3f(cos(yaw), sin(pitch) * 0.45, sin(yaw) * 0.65));
        let vDir = normalize(vec3f(-sin(yaw) * 0.45, cos(pitch), cos(yaw) * 0.55));

        let sprigSize = 0.075 + 0.048 * fract(b.z * 29.3);
        worldPos += (uDir * uv.x + vDir * uv.y) * sprigSize;

        // 3-tone anime cel shading across the tree crown height & sun direction
        let sunFacing = clamp((worldPos.y - 1.75) * 0.55 - worldPos.x * 0.08 + (b.y * 0.25), 0.0, 1.0);
        let deepLeaf = vec3f(0.03, 0.13, 0.16);
        let midLeaf = vec3f(0.11, 0.39, 0.23);
        let litLeaf = vec3f(0.44, 0.65, 0.21);

        var leafCol = mix(deepLeaf, midLeaf, smoothstep(0.18, 0.45, sunFacing));
        leafCol = mix(leafCol, litLeaf, smoothstep(0.62, 0.85, sunFacing));

        // Delicate sakura blossom sprigs on ~20% of the outer/lower boughs
        let isBlossomSprig = fract(b.x * 31.7) > 0.80 && worldPos.y < 2.85;
        let blossomCol = mix(vec3f(0.78, 0.28, 0.50), vec3f(1.05, 0.64, 0.80), smoothstep(0.25, 0.75, sunFacing));
        color = select(leafCol, blossomCol, isBlossomSprig);
    } else if (b.w < 0.82) {
        // Intertwined Azure & Rose-Gold Musubi Spirit-Dragon streams!
        kind = 1.0;
        let isRoseGold = b.w >= 0.59;
        let s = fract(b.z + a.w * 0.075);
        let endFade = smoothstep(0.0, 0.08, s) * smoothstep(1.0, 0.88, s);
        let pulse = (0.82 + 0.18 * sin(a.w * 5.0 + b.z * 37.0)) * endFade;
        if (isRoseGold) {
            color = mix(vec3f(2.15, 0.08, 0.65), vec3f(2.10, 0.72, 0.04), fract(b.x * 9.1)) * pulse;
        } else {
            color = mix(vec3f(0.02, 0.65, 2.25), vec3f(0.06, 1.55, 2.05), fract(b.x * 7.3)) * pulse;
        }
    } else if (b.w < 0.94) {
        // Twilight Water Spirit Fireflies
        kind = 1.5;
        let pulse = 0.55 + 0.45 * sin(a.w * 7.5 + b.x * 53.0);
        color = mix(vec3f(0.18, 1.55, 2.15), vec3f(1.85, 1.15, 0.25), step(0.5, fract(b.z * 11.3))) * pulse;
    } else {
        // Floating Sky Lanterns (Tōrō)
        kind = 2.0;
        let flicker = 0.84 + 0.16 * sin(a.w * 6.5 + b.x * 41.0);
        color = mix(vec3f(2.4, 1.15, 0.28), vec3f(2.05, 0.62, 0.18), fract(b.z * 5.7)) * flicker;
    }

    var clip = camera.viewProj * vec4f(worldPos, 1.0);
    if (kind > 0.5 && kind < 2.5) {
        var baseSize = 0.038;
        if (kind > 1.2) { baseSize = 0.015; }
        if (kind > 1.8) { baseSize = 0.026; }
        let size = (baseSize + 0.010 * fract(b.z * 17.3)) * camera.proj[1][1];
        clip.x += uv.x * size / camera.aspect;
        clip.y += uv.y * size;
    }

    var o: V;
    o.pos = clip;
    o.uv = uv;
    o.color = color;
    o.kind = kind;
    return o;
}

@fragment fn fs_main(v: V) -> @location(0) vec4f {
    let p = v.uv;
    let r2 = dot(p, p);
    if (r2 > 1.0) { discard; }

    if (v.kind < 0.5) {
        // Authentic notched cherry-blossom (Sakura) petal / leaf silhouette in local (p.x, p.y):
        let px = p.x * 1.25;
        let py = p.y * 1.05 - 0.08;
        let notch = 0.28 * exp(-abs(px) * 6.5) * step(0.0, py);
        let petalShape = px * px * (1.15 - 0.35 * py) + pow(py + notch, 2.0);
        if (petalShape > 0.72) { discard; }

        let tipGrad = clamp(py * 0.65 + 0.5, 0.0, 1.0);
        let petalCol = mix(v.color * 0.78, v.color * 1.18, tipGrad);
        let alpha = 0.85;
        return vec4f(petalCol * alpha, alpha);
    }

    if (v.kind > 2.5) {
        // 3-Leaf Pointed Anime Foliage Sprig Silhouette (center blade + left/right angled blades!)
        let y0 = p.y + 0.35;
        // Center pointed leaf blade
        let bladeC = abs(p.x) * 2.1 + pow(y0 - 0.42, 2.0) * 1.65;
        // Left & right angled leaf blades
        let lx = abs(p.x) - 0.18;
        let ly = y0 - 0.25;
        let rotX = lx * 0.76 - ly * 0.65;
        let rotY = lx * 0.65 + ly * 0.76;
        let bladeS = abs(rotX) * 2.35 + pow(rotY - 0.22, 2.0) * 2.1;

        let dSprig = min(bladeC, bladeS);
        if (dSprig > 0.58) { discard; }

        // Crisp anime leaf tip gradient & dark stem base
        let leafGrad = clamp((p.y + 0.45) * 0.85, 0.0, 1.0);
        let vein = smoothstep(0.06, 0.0, abs(p.x)) * 0.12;
        let leafCol = mix(v.color * 0.58, v.color * 1.18 + vec3f(vein), leafGrad);
        let alpha = 0.92;
        return vec4f(leafCol * alpha, alpha);
    }

    if (v.kind < 1.2) {
        // Silky additive glowing Spirit-Dragon stream wisp
        let wisp = exp(-r2 * 3.5) * 0.056;
        return vec4f(v.color * wisp, 0.0);
    }

    if (v.kind < 1.8) {
        // Water Spirit Firefly
        let wisp = exp(-r2 * 3.8) * 0.065;
        return vec4f(v.color * wisp, 0.0);
    }

    // Floating Sky Lantern (warm rectangular paper lantern body + glowing candle core)
    let boxD = max(abs(p.x) * 1.35, abs(p.y) * 1.05);
    let lanternGlow = exp(-r2 * 3.8) * 0.11;
    if (boxD < 0.42) {
        let frameBand = step(0.34, abs(p.y) * 1.05);
        let coreCol = select(v.color, vec3f(0.12, 0.05, 0.09), frameBand > 0.5);
        return vec4f(coreCol * 0.88 + v.color * lanternGlow, 0.88);
    }
    return vec4f(v.color * lanternGlow, 0.0);
}`;

export default async function celestialTorii({ engine, width, height, scale = 1, settings = {}, film, log = () => {} }) {
    const count = settings.count ?? 65536;
    if (!Number.isInteger(count) || count < 128 || count > 1048576) {
        throw new Error('celestial-torii: count must be 128–1048576');
    }

    // NullGraph's signature Anime PostChain (see presets.ts `ChainPresets.anime`):
    // Soft atmospheric Bloom + warm film Halation + ACES Tonemap + FXAA before pure 3D Depth Outline!
    const post = new PostChain(engine, { gbuffer: { depth: 'depth24plus' }, scale });
    post.setChain([
        new Bloom({ threshold: 0.88, knee: 0.42, intensity: 0.32, mips: 5 }),
        new Halation({ threshold: 0.95, knee: 0.45, intensity: 0.35, radius: 2.0, tint: [1.0, 0.36, 0.18] }),
        new Tonemap({ curve: 'aces', exposure: 1.12, whitePoint: 4.5 }),
        new FXAA(),
        new Outline({
            thickness: 1.15,
            depthSensitivity: 7.5,
            lumaSensitivity: 0.0,
            strength: 0.88,
            color: [0.06, 0.03, 0.12],
        }),
        new Vignette({ intensity: 0.18, softness: 0.65 }),
    ]);
    post.resize(width, height);

    const pass = engine.createPass({
        name: 'Celestial Torii Anime / HDR + Depth',
        isMainScreenPass: false,
        ...post.sceneAttachments({ clearValue: { r: 0.015, g: 0.03, b: 0.12, a: 1 } }),
    });

    const background = engine.createBatch(pass, {
        shaderCode: BACKDROP,
        strideFloats: 8,
        maxInstances: 1,
        targetFormat: 'rgba16float',
        depthWriteEnabled: true,
        depthCompare: 'always',
    });
    engine.updateBatchData(background, new Float32Array(8), 1);

    const batch = engine.createBatch(pass, {
        shaderCode: PARTICLES,
        strideFloats: 8,
        maxInstances: count,
        targetFormat: 'rgba16float',
        depthWriteEnabled: false,
        depthCompare: 'less',
        blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        },
    });

    const data = new Float32Array(count * 8);
    let seed = 271828183;
    const rand = () => {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        return seed / 4294967296;
    };

    for (let i = 0; i < count; i++) {
        const rx = rand();
        const ry = rand() * 2.0 - 1.0;
        const rz = rand();
        const kind = rand();
        data.set([0, 0, 0, 0.0, rx, ry, rz, kind], i * 8);
    }
    engine.updateBatchData(batch, data, count);

    const evolve = engine.createComputeStage({
        name: 'Celestial Torii / sakura & spirit evolution',
        shaderCode: EVOLVE,
        dispatchCount: [Math.ceil(count / 128), 1, 1],
    });
    engine.setComputePassBindGroup(evolve, [{ binding: 0, resource: { buffer: batch.storageBuffer } }]);

    // Initialize positions before step zero is saved.
    engine.render([evolve]);
    await engine.device.queue.onSubmittedWorkDone();

    const camera = new Camera(48, width / height, 0.1, 120);
    const defaultView = { target: [0.0, 1.15, -0.4], distance: 10.2, azimuth: -0.22, polar: 1.44 };

    const track = film.track('sakura-spirits', batch.storageBuffer, {
        every: 6,
        stride: 32,
        fields: [0, 1, 2, 3, 4, 5, 6, 7].map(word => ({ word, store: 'f32', interp: 'linear' })),
    });

    log(`Celestial Torii Anime: ${count.toLocaleString()} GPU elements, ${(data.byteLength / 1048576).toFixed(0)} MiB persistent state (3D Sakura petals + Musubi spirit streams + hardware depth Outline + Halation).`);

    return {
        dt: 1 / 60,
        settings: { count },
        tracks: [track],
        checkpoint: [{ name: 'sakura-spirits', buffer: batch.storageBuffer }],
        info: {
            particles: count,
            model: 'Anime 3D raymarched Water-Torii & Sacred Sakura shrine with depth-inked Outline, Toon/Rim shading, Halation, 3D sakura petals, and Musubi spirit streams',
            wgsl: {
                evolve: hash('sha256', EVOLVE),
                particles: hash('sha256', PARTICLES),
                backdrop: hash('sha256', BACKDROP),
            },
        },
        async step() {
            engine.render([evolve]);
            await engine.device.queue.onSubmittedWorkDone();
        },
        view(state, seconds) {
            const v = state ?? defaultView;
            const s = Math.sin(v.polar);
            camera.updateView(
                [
                    v.target[0] + v.distance * s * Math.sin(v.azimuth),
                    v.target[1] + v.distance * Math.cos(v.polar),
                    v.target[2] + v.distance * s * Math.cos(v.azimuth),
                ],
                v.target,
            );
            camera.updateTime(seconds);
            engine.updateCamera(camera);
        },
        draw() {
            post.upload();
            return [pass, ...post.getPassSequence()];
        },
        destroy() {
            post.destroy();
            engine.clearPasses();
        },
    };
}
