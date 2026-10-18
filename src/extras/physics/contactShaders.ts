import type { ResolvedDiscContacts } from './contactMath';
import { pointCellWGSL } from '../spatial/gridShaders';
import { GRID_WINDOW_WGSL } from '../spatial/gridWindow';

const f = (v: number) => Number.isInteger(v) ? `${v}.0` : `${v}`;
const word = (stride: number, offset: number, i = 'i') => `${i} * ${stride}u + ${offset}u`;

export function buildContactGatherWGSL(o: ResolvedDiscContacts, pass: number): string {
    const { capacity, stride, alive, fixed } = o.layout;
    const live = (i: string) => alive ? `(bitcast<u32>(records[${word(stride, alive.offset, i)}]) & ${alive.mask}u) != 0u` : 'true';
    return `
@group(0) @binding(0) var<storage, read> records: array<f32>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read> bodies: array<u32>;
@group(0) @binding(3) var<storage, read_write> deltas: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: vec4<f32>;
@group(0) @binding(5) var<storage, read> counters: array<u32>;
${GRID_WINDOW_WGSL}
${pointCellWGSL(o)}
fn position(i: u32) -> vec4<f32> {
    let b = i * ${stride}u;
    return vec4<f32>(records[b], records[b + 1u], records[b + 2u], records[b + 3u]);
}
fn inverseMass(i: u32, radius: f32) -> f32 {
    ${fixed ? `if ((bitcast<u32>(records[${word(stride, fixed.offset)}]) & ${fixed.mask}u) != 0u) { return 0.0; }` : ''}
    ${o.mass === 'area' ? `let r = max(radius, 1e-4); return ${f(o.maxRadius * o.maxRadius)} / (r * r);` : 'return 1.0;'}
}
fn tieBreak(i: u32, j: u32) -> vec2<f32> {
    let lo = min(i, j);
    let hi = max(i, j);
    var h = (lo * 0x9e3779b9u) ^ ((hi + 0x7f4a7c15u) * 0x85ebca6bu);
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    h = h ^ (h >> 16u);
    let a = f32(h >> 8u) * (6.283185307179586 / 16777216.0);
    let n = vec2<f32>(cos(a), sin(a));
    return select(-n, n, i == lo);
}
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= ${capacity}u) { return; }
    deltas[i] = vec4<f32>(0.0);
    if (!(${live('i')})) { return; }
    let me = position(i);
    let wi = inverseMass(i, me.w);
    if (wi == 0.0) { return; }
    let cell = pointCell(me.xyz);
    let cx = i32(cell % ${o.grid.cellsX}u);
    let cz = i32(cell / ${o.grid.cellsX}u);
    let window = counters[0] * ${o.iterations}u + ${pass}u;
    var sum = vec2<f32>(0.0);
    var count = 0u;
    for (var z = max(0, cz - 1); z <= min(${o.grid.cellsZ - 1}, cz + 1); z++) {
        for (var x = max(0, cx - 1); x <= min(${o.grid.cellsX - 1}, cx + 1); x++) {
            let c = u32(z) * ${o.grid.cellsX}u + u32(x);
            let first = offsets[c];
            let n = offsets[c + 1u] - first;
            var at = first;
            if (n > ${o.maxScanPerCell}u) { at += gridWindow(c, window) % n; }
            for (var left = min(n, ${o.maxScanPerCell}u); left > 0u; left--) {
                let j = bodies[at];
                at++;
                if (at == first + n) { at = first; }
                if (j == i || !(${live('j')})) { continue; }
                let other = position(j);
                let d = me.xz - other.xz;
                let reach = me.w + other.w;
                let d2 = dot(d, d);
                if (d2 >= reach * reach) { continue; }
                let wj = inverseMass(j, other.w);
                let dist = sqrt(d2);
                var normal = tieBreak(i, j);
                if (dist > 1e-6) { normal = d / dist; }
                sum += normal * ((reach - dist) * (wi / (wi + wj + params.y)));
                count++;
            }
        }
    }
    deltas[i] = vec4<f32>(sum, f32(count), 0.0);
}`;
}

export function buildContactApplyWGSL(o: ResolvedDiscContacts): string {
    const { capacity, stride, velocity } = o.layout, w = o.world;
    return `
@group(0) @binding(0) var<storage, read_write> records: array<f32>;
@group(0) @binding(1) var<storage, read> deltas: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: vec4<f32>;
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= ${capacity}u) { return; }
    let d = deltas[i];
    if (d.z <= 0.0) { return; }
    var u = d.xy * (params.x / sqrt(d.z));
    let len = length(u);
    if (len > params.z) { u *= params.z / len; }
    let b = i * ${stride}u;
    var p = vec2<f32>(records[b], records[b + 2u]) + u;
    ${o.clamp ? `let r = records[b + 3u];
    let mid = vec2<f32>(${f((w.minX + w.maxX) / 2)}, ${f((w.minZ + w.maxZ) / 2)});
    p = clamp(p, min(vec2<f32>(${f(w.minX)}, ${f(w.minZ)}) + r, mid),
        max(vec2<f32>(${f(w.maxX)}, ${f(w.maxZ)}) - r, mid));` : ''}
    records[b] = p.x;
    records[b + 2u] = p.y;
    ${velocity ? `if (params.w > 0.0 && len > 1e-9) {
        let n = u / length(u);
        let v = vec2<f32>(records[b + ${velocity.offset}u], records[b + ${velocity.offset + 2}u]);
        let damped = v - params.w * min(0.0, dot(v, n)) * n;
        records[b + ${velocity.offset}u] = damped.x;
        records[b + ${velocity.offset + 2}u] = damped.y;
    }` : ''}
}`;
}

export function buildContactCountWGSL(o: ResolvedDiscContacts): string {
    const { capacity, stride, alive } = o.layout;
    return `
@group(0) @binding(0) var<storage, read> records: array<f32>;
@group(0) @binding(1) var<storage, read_write> counters: array<atomic<u32>>;
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= ${capacity}u) { return; }
    ${alive ? `if ((bitcast<u32>(records[${word(stride, alive.offset)}]) & ${alive.mask}u) == 0u) { return; }` : ''}
    if (records[${word(stride, 3)}] > ${f(o.maxRadius)}) { atomicAdd(&counters[1], 1u); }
}`;
}

export const CONTACT_RESET_WGSL = `
@group(0) @binding(0) var<storage, read_write> counters: array<u32>;
@compute @workgroup_size(1) fn cs_main() { counters[1] = 0u; }`;
export const CONTACT_TICK_WGSL = `
@group(0) @binding(0) var<storage, read_write> counters: array<u32>;
@compute @workgroup_size(1) fn cs_main() { counters[0] += 1u; }`;