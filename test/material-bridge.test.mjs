// test/material-bridge.test.mjs
//
// Guards the visibility <-> materials bridge (docs/api/visibility-shader-reference.md §4):
//   1. The VertexOut contract stays field-identical between materials and visibility — the single
//      thing that makes a stock material fragment usable in a GPU-culled draw.
//   2. Every stock material family composes into BOTH visibility draw tiers, at the right group,
//      with no binding collisions.
// Runs in plain Node — no GPU.

import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const dir = mkdtempSync(join(tmpdir(), 'ng-matbridge-'));
const entry = join(dir, 'entry.ts');
const out = join(dir, 'out.mjs');

// Absolute path with forward slashes — esbuild resolves this on every platform.
const mod = (p) => JSON.stringify(resolve(process.cwd(), p).split('\\').join('/'));

writeFileSync(entry, `
export { buildMaterialFragment } from ${mod('src/extras/materials/materialFragments')};
export { CommonChunks } from ${mod('src/extras/materials/CommonChunks')};
export { buildInstanceDrawWGSL, buildMeshletDrawWGSL } from ${mod('src/extras/visibility/visibilityShaders')};
`);

execFileSync('npx', ['esbuild', entry, '--bundle', '--platform=node', '--format=esm', '--outfile=' + out],
    { stdio: 'pipe', shell: process.platform === 'win32' });

const M = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });

let failures = 0;
const ok = (name, cond, detail = '') => {
    if (cond) console.log(`  ok    ${name}`);
    else { failures++; console.log(`  FAIL  ${name}${detail ? ' — ' + detail : ''}`); }
};

const KINDS = ['basic', 'lambert', 'emissive', 'matcap', 'toon', 'pbr'];
const lighting = {
    wgslStructs: 'struct Light { a: vec4<f32> };\n@group(2) @binding(0) var<storage, read> lights: array<Light>;',
    wgslLogic: 'fn getVisibleLightCount(p: vec3<f32>) -> u32 { return 1u; }\n'
        + 'fn getIncidentLight(i: u32, p: vec3<f32>) -> IncidentLight { return IncidentLight(vec3<f32>(1.0), vec3<f32>(0.0,1.0,0.0), 1.0); }',
};

// --- 1. VertexOut parity — the contract the whole bridge rests on ---
{
    const fields = (s) => [...s.matchAll(/@location\((\d+)\)\s*(\w+)\s*:\s*([\w<>]+)/g)]
        .map((m) => `${m[1]}:${m[2]}:${m[3]}`);
    const matVO = M.CommonChunks.VertexOut;
    const visVO = (M.buildInstanceDrawWGSL({}).match(/struct VertexOut \{[\s\S]*?\};/) || [''])[0];
    const a = fields(matVO).join(','), b = fields(visVO).join(',');
    ok('VertexOut is field-identical (materials vs visibility)', a === b && a.length > 0,
        `\n          materials:  ${a}\n          visibility: ${b}`);
}

// --- 2. Every family composes into both tiers ---
const TIERS = [
    ['instance', 1, (opts) => M.buildInstanceDrawWGSL(opts)],
    ['meshlet', 2, (opts) => M.buildMeshletDrawWGSL({ strideFloats: 8, ...opts })],
];

for (const kind of KINDS) {
    for (const [tier, group, build] of TIERS) {
        const { bindings, fragment } = M.buildMaterialFragment({ kind, groupIndex: group });
        const wgsl = build({ fragmentBody: fragment, extraBindings: bindings });

        const vertexOuts = (wgsl.match(/struct VertexOut/g) || []).length;
        const hasStages = /@vertex/.test(wgsl) && /@fragment/.test(wgsl);
        const staleField = /in\.normal\b/.test(wgsl);   // pre-alignment field name

        const seen = new Set();
        let collision = null;
        for (const m of wgsl.matchAll(/@group\((\d+)\)\s*@binding\((\d+)\)/g)) {
            const key = `${m[1]}.${m[2]}`;
            if (seen.has(key)) collision = key;
            seen.add(key);
        }
        const onGroup = [...seen].some((k) => k.startsWith(`${group}.`));

        ok(`${tier} tier + ${kind}`,
            hasStages && vertexOuts === 1 && !staleField && !collision && onGroup,
            `stages=${hasStages} VertexOut x${vertexOuts} stale=${staleField} collision=${collision} onGroup=${onGroup}`);
    }
}

// --- 3. Lit variants: material at group 1, lights at group 2, no clash ---
for (const kind of ['lambert', 'toon', 'pbr']) {
    const { bindings, fragment } = M.buildMaterialFragment({ kind, groupIndex: 1, lighting });
    const wgsl = M.buildInstanceDrawWGSL({ fragmentBody: fragment, extraBindings: bindings });
    ok(`instance tier + ${kind} + clustered lights`,
        /getVisibleLightCount/.test(wgsl) && /getIncidentLight/.test(wgsl)
        && /@group\(2\)\s*@binding\(0\)/.test(wgsl));
}

// --- 4. Fragments are real ---
{
    const { fragment } = M.buildMaterialFragment({ kind: 'pbr', groupIndex: 1 });
    ok('material fragment declares fs_main', /@fragment[\s\S]*fn fs_main/.test(fragment));
}

// --- 5. Unknown kind fails loud ---
{
    let threw = false;
    try { M.buildMaterialFragment({ kind: 'nope' }); } catch { threw = true; }
    ok('unknown material kind throws', threw);
}

console.log(failures === 0
    ? '\nAll material-bridge tests passed.'
    : `\n${failures} material-bridge test(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
