// D2.2: serializable layouts, CPU mirrors, and WGSL generated from the same field table.
const sizes = { f32: 4, f16: 2, unorm16: 2, u8: 1, u32: 4 };
const kinds = ['step', 'linear', 'angle', 'wrap', 'hermite'];
const fail = message => { throw new Error(`film: layout ${message}`); };
export const AGENTS = [
    { word: 0, store: 'f32', interp: 'hermite', velocity: 4 },
    { word: 2, store: 'f32', interp: 'hermite', velocity: 6 },
    { word: 4, store: 'f16', interp: 'linear' },
    { word: 6, store: 'f16', interp: 'linear' },
    { word: 7, store: 'f16', interp: 'angle' },
    { word: 13, store: 'u8', interp: 'step' },
];

export function resolveLayout({ stride, fields, preset, add = [], every = 6, dt = 1 / 60 }) {
    if (preset && preset !== 'agents') fail(`unknown preset "${preset}".`);
    stride ??= preset === 'agents' ? 64 : undefined;
    if (!Number.isInteger(stride) || stride < 4 || stride % 4) fail('stride must be a positive multiple of four bytes.');
    if (!Number.isInteger(every) || every < 1 || !Number.isFinite(dt) || dt <= 0) fail('every and dt must be positive (every a whole number).');
    const input = fields ?? (preset === 'agents' ? AGENTS : null);
    if (!input) return { stride, every, dt, fields: null, packedStride: stride };
    if (!Array.isArray(input) || !input.length || !Array.isArray(add)) fail('fields must be a nonempty list.');
    let offset = 0;
    const seen = new Set();
    const resolved = [...input, ...add].map(f => {
        if (!Number.isInteger(f.word) || f.word < 0 || f.word * 4 >= stride) fail(`field word ${f.word} is outside the ${stride}-byte record.`);
        if (seen.has(f.word)) fail(`duplicate word ${f.word}.`);
        seen.add(f.word);
        if (!Object.hasOwn(sizes, f.store)) fail(`unknown storage "${f.store}".`);
        const interp = f.interp ?? 'step';
        if (!kinds.includes(interp)) fail(`unknown interpolation "${interp}".`);
        if (['u32', 'u8'].includes(f.store) && interp !== 'step') fail(`${f.store} word ${f.word} must use step interpolation.`);
        if (interp === 'wrap' && (!Number.isFinite(f.maxRate) || f.maxRate <= 0 || f.maxRate * every * dt >= 0.5)) fail(`wrap word ${f.word} needs maxRate with maxRate × every × dt below half a cycle.`);
        if (interp === 'hermite' && ![...input, ...add].some(v => v.word === f.velocity && !['u8', 'u32'].includes(v.store))) fail(`hermite word ${f.word} needs a tracked velocity word.`);
        if (f.jump !== undefined && (!Number.isFinite(f.jump) || f.jump <= 0)) fail(`word ${f.word} jump must be positive.`);
        const bytes = sizes[f.store];
        offset = Math.ceil(offset / bytes) * bytes;
        const result = { ...f, interp, offset };
        offset += bytes;
        return result;
    });
    return { stride, every, dt, fields: resolved, packedStride: Math.ceil(offset / 4) * 4 };
}

const view = bytes => new DataView(bytes.buffer ?? bytes, bytes.byteOffset ?? 0, bytes.byteLength);
function load(d, at, store) {
    if (store === 'f32') return d.getFloat32(at, true);
    if (store === 'f16') return d.getFloat16(at, true);
    if (store === 'unorm16') return d.getUint16(at, true) / 65535;
    return store === 'u8' ? d.getUint8(at) : d.getUint32(at, true);
}
export function packCPU(layout, source) {
    const src = view(source), count = src.byteLength / layout.stride;
    if (!Number.isInteger(count)) fail('source size is not a whole number of records.');
    if (!layout.fields) return new Uint8Array(src.buffer, src.byteOffset, src.byteLength).slice();
    const out = new Uint8Array(count * layout.packedStride), dst = view(out);
    for (let i = 0; i < count; i++) for (const f of layout.fields) {
        const p = i * layout.packedStride + f.offset, q = i * layout.stride + f.word * 4;
        const x = ['u8', 'u32'].includes(f.store) ? src.getUint32(q, true) : src.getFloat32(q, true);
        if (f.store === 'f32') dst.setFloat32(p, x, true);
        else if (f.store === 'f16') dst.setFloat16(p, x, true);
        else if (f.store === 'unorm16') dst.setUint16(p, Math.round(Math.min(1, Math.max(0, x)) * 65535), true);
        else if (f.store === 'u8') dst.setUint8(p, x & 255);
        else dst.setUint32(p, x, true);
    }
    return out;
}
export function unpackCPU(layout, left, right, u, target, interval = layout.every * layout.dt) {
    const a = view(left), b = view(right), dst = view(target);
    u = Math.max(0, Math.min(1, u));
    if (!layout.fields) { new Uint8Array(dst.buffer, dst.byteOffset, dst.byteLength).set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength)); return target; }
    for (let i = 0; i < dst.byteLength / layout.stride; i++) for (const f of layout.fields) {
        const p = i * layout.packedStride, x = load(a, p + f.offset, f.store), y = load(b, p + f.offset, f.store);
        let z = x;
        if (u === 1) z = y;
        else if (u !== 0) {
            if (f.interp === 'step') z = u < 0.5 ? x : y;
            else if (f.interp === 'linear') z = x + u * (y - x);
            else if (f.interp === 'angle') z = x + u * ((((y - x + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) - Math.PI);
            else if (f.interp === 'wrap') z = (x + u * ((y - x + 1) % 1)) % 1;
            else {
                const v = layout.fields.find(v => v.word === f.velocity);
                const va = load(a, p + v.offset, v.store), vb = load(b, p + v.offset, v.store);
                z = (2*u**3-3*u**2+1)*x + (u**3-2*u**2+u)*interval*va + (-2*u**3+3*u**2)*y + (u**3-u**2)*interval*vb;
            }
            if (f.jump !== undefined && Math.abs(y - x) > f.jump) z = u < 0.5 ? x : y;
        }
        const q = i * layout.stride + f.word * 4;
        if (f.store === 'u8') dst.setUint32(q, (dst.getUint32(q, true) & 0xffffff00) | z, true);
        else if (f.store === 'u32') dst.setUint32(q, z, true);
        else dst.setFloat32(q, z, true);
    }
    return target;
}

export function layoutWGSL(l) {
    if (!l.fields) fail('raw tracks do not need shaders.');
    const sw = l.stride / 4, pw = l.packedStride / 4;
    const encoded = f => {
        const s = `src[i*${sw}u+${f.word}u]`, x = `bitcast<f32>(${s})`;
        return f.store === 'f32' || f.store === 'u32' ? s : f.store === 'f16' ? `(pack2x16float(vec2f(${x}, 0.0)) & 65535u)`
            : f.store === 'unorm16' ? `u32(floor(clamp(${x}, 0.0, 1.0)*65535.0+0.5))` : `(${s} & 255u)`;
    };
    const read = (f, name) => {
        const word = `${name}[i*${pw}u+${Math.floor(f.offset / 4)}u]`;
        const bits = f.offset % 4 * 8, v = bits ? `(${word} >> ${bits}u)` : word;
        return f.store === 'f32' ? `bitcast<f32>(${word})` : f.store === 'u32' ? word : f.store === 'u8' ? `(${v} & 255u)`
            : f.store === 'f16' ? `unpack2x16float(${v}).x` : `(f32(${v} & 65535u)/65535.0)`;
    };
    const stores = l.fields.map(f => {
        const x = read(f, 'a'), y = read(f, 'b'), integer = ['u8', 'u32'].includes(f.store);
        let expr = 'select(x, y, u >= 0.5)';
        if (f.interp === 'linear') expr = 'x+u*(y-x)';
        if (f.interp === 'angle') expr = 'x+u*((y-x+3.141592653589793)-floor((y-x+3.141592653589793)/6.283185307179586)*6.283185307179586-3.141592653589793)';
        if (f.interp === 'wrap') expr = 'fract(x+u*fract(y-x))';
        if (f.interp === 'hermite') {
            const v = l.fields.find(v => v.word === f.velocity);
            expr = `(2.0*u*u*u-3.0*u*u+1.0)*x+(u*u*u-2.0*u*u+u)*p.y*${read(v, 'a')}+(-2.0*u*u*u+3.0*u*u)*y+(u*u*u-u*u)*p.y*${read(v, 'b')}`;
        }
        const dst = `dst[i*${sw}u+${f.word}u]`;
        return `{ let x = ${x}; let y = ${y}; var z = ${expr};
            ${f.jump !== undefined ? `if(abs(f32(y)-f32(x)) > ${Number(f.jump).toExponential()}) { z = select(x,y,u>=0.5); }` : ''}
            if(u==0.0) { z=x; } if(u==1.0) { z=y; }
            ${dst} = ${f.store === 'u8' ? `(${dst} & 4294967040u) | z` : integer ? 'z' : 'bitcast<u32>(z)'}; }`;
    }).join('\n');
    const pack = `@group(0) @binding(0) var<storage,read> src: array<u32>;
@group(0) @binding(1) var<storage,read_write> dst: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u) {
 let i=id.x; if(i>=arrayLength(&src)/${sw}u) { return; }
 ${Array.from({ length: pw }, (_, w) => `dst[i*${pw}u+${w}u] = ` + (l.fields.filter(f => Math.floor(f.offset / 4) === w).map(f => `(${encoded(f)} << ${f.offset % 4 * 8}u)`).join(' | ') || '0u') + ';').join('\n')}
}`;
    const unpack = `@group(0) @binding(0) var<storage,read> a:array<u32>;
@group(0) @binding(1) var<storage,read> b:array<u32>;
@group(0) @binding(2) var<storage,read_write> dst:array<u32>;
@group(0) @binding(3) var<uniform> p:vec4f;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u) {
 let i=id.x; if(i>=arrayLength(&dst)/${sw}u) { return; } let u=clamp(p.x,0.0,1.0);
 ${stores}
}`;
    return { pack, unpack };
}