// A weathered wooden fence around the Models field: posts every couple of metres just outside the
// agents' world bounds, two rails between each pair. No two posts stand quite the same: each leans
// a few degrees, one in ten leans hard; some rails are gone, and some have snapped at one end and
// lie angled down to the grass. Built from two boxes, as static VisibilitySystem rows, deterministic
// for a field size (rand01). Shaded by the stock ToonMaterial (field.ts).

import { rand01 } from 'null-graph/agents';
import { Primitives, StandardLayout } from 'null-graph/geometry';

type Vec3 = [number, number, number];

const SPACING = 2.4; // m between posts
const POST = 0.16; // m: a post's width and depth
const RAIL = { height: 0.1, depth: 0.05 };
const LEVELS = [0.38, 0.8]; // the rails' heights, as shares of their posts'
const SEED = 97; // rand01's tick for the fence, apart from the grass's (83)

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const unit = (a: Vec3): Vec3 => scale(a, 1 / Math.hypot(...a));

/** Column-major model matrix from three axis columns and a translation. */
const matrix = (x: Vec3, y: Vec3, z: Vec3, t: Vec3) => [...x, 0, ...y, 0, ...z, 0, ...t, 1];

export interface FenceRow {
  model: number[];
  bounds: [number, number, number, number];
  mesh: 'post' | 'rail';
}

/** The two meshes: a post standing on its base (1 m tall, scaled per row), and a centred unit rail. */
export function fenceMeshes() {
  const post = Primitives.createCube(StandardLayout, POST, 1, POST);
  for (let v = 1; v < post.vertices.length; v += StandardLayout.strideFloats) post.vertices[v] += 0.5;
  const rail = Primitives.createCube(StandardLayout, 1, 1, 1);
  return { post, rail };
}

/** Every post and rail of the fence around a field of half-width `half`. */
export function fenceRows(half: number): FenceRow[] {
  const edge = half + 0.6;
  const corners: Vec3[] = [[-edge, 0, -edge], [edge, 0, -edge], [edge, 0, edge], [-edge, 0, edge]];
  const perSide = Math.max(2, Math.round((2 * edge) / SPACING));
  // One ring of posts, side after side from each corner; each leans by its own tilt.
  const posts = corners.flatMap((from, side) => {
    const to = corners[(side + 1) % 4];
    const along = unit(sub(to, from));
    const out: Vec3 = [along[2], 0, -along[0]]; // the side's outward normal
    return Array.from({ length: perSide }, (_, k) => {
      const i = side * perSide + k;
      const r = (s: number) => rand01(i, SEED, s);
      const step = k * (2 * edge) / perSide + (k === 0 ? 0 : (r(0) - 0.5) * 0.3);
      const base = add(add(from, scale(along, step)), scale(out, (r(1) - 0.5) * 0.15));
      const lean = (r(2) < 0.1 ? 10 + r(3) * 12 : r(3) * 4) * (Math.PI / 180);
      const heading = r(4) * Math.PI * 2;
      const up = unit([Math.sin(lean) * Math.cos(heading), Math.cos(lean), Math.sin(lean) * Math.sin(heading)]);
      return { i, base, up, along, height: 1.1 + r(5) * 0.3 };
    });
  });
  const rows: FenceRow[] = posts.map((p) => {
    const x = unit(cross(p.up, p.along));
    return { model: matrix(x, scale(p.up, p.height), cross(x, p.up), p.base), bounds: [0, 0.5, 0, 0.52], mesh: 'post' };
  });
  // Two rails from each post to the next around the ring: one in eight gone, one in twelve snapped.
  posts.forEach((p, j) => {
    const q = posts[(j + 1) % posts.length];
    LEVELS.forEach((level, n) => {
      const r = (s: number) => rand01(p.i * 2 + n, SEED + 1, s);
      const roll = r(0);
      if (roll < 0.12) return;
      let a = add(p.base, scale(p.up, p.height * level));
      let b = add(q.base, scale(q.up, q.height * level));
      if (roll < 0.2) {
        // Snapped: one end still nailed to its post, the other down in the grass, shorter.
        const keepStart = r(1) < 0.5;
        const [fixed, loose] = keepStart ? [a, b] : [b, a];
        const reach = 0.55 + r(2) * 0.3;
        const fallen: Vec3 = [
          fixed[0] + (loose[0] - fixed[0]) * reach,
          RAIL.height / 2,
          fixed[2] + (loose[2] - fixed[2]) * reach,
        ];
        [a, b] = keepStart ? [fixed, fallen] : [fallen, fixed];
      }
      const span = sub(b, a);
      const x = unit(span);
      const z = unit(cross(x, [0, 1, 0]));
      rows.push({
        model: matrix(
          scale(x, Math.hypot(...span) + 0.12),
          scale(cross(z, x), RAIL.height),
          scale(z, RAIL.depth),
          scale(add(a, b), 0.5)
        ),
        bounds: [0, 0, 0, 0.87],
        mesh: 'rail',
      });
    });
  });
  return rows;
}
