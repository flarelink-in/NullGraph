// src/extras/geometry/MeshletBuilder.ts
//
// Meshlet geometry builder — implementation of docs/geometry-meshlets-design.md (v1.0.5).
//
// Partitions each mesh's triangles into meshlets (≤ maxVertices vertices / ≤ maxTriangles
// triangles), each carrying an object-space bounding sphere and a backface normal cone, and
// packs the result into flat typed arrays ready to upload as GPU STORAGE buffers. This is a
// BUILD-TIME module: it never touches a GPUDevice, so it runs in a Web Worker or a Node bake
// script, and its per-frame cost is zero (design doc §1, §9).
//
// Meshlet vertex references are GLOBAL indices into the shared MegabufferBuilder vertex pool
// (design doc §2.1) — one mega vertex buffer backs both the instance tier and the meshlet tier.
//
// The greedy meshletizer (§4) is index-order, single-pass, and deterministic: the same mesh in
// produces byte-identical buffers out (property P5).

import { MegabufferBuilder } from "./MegaBufferBuilder";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface MeshHandle {
    /** Index into the packed meshTable. Instances store this to reference the mesh. */
    meshIndex: number;
    name: string;
}

export interface MeshLODInput {
    /** Index array for this LOD level (3 indices per triangle). */
    indices: Uint16Array | Uint32Array | number[];
    /** World-space geometric error of this level vs LOD0, in the mesh's units. LOD0 = 0. */
    error: number;
}

export interface MeshletMeshInput {
    /** Interleaved vertices at the megabuffer's stride. */
    vertices: Float32Array | number[];
    /** Single-LOD form. Exactly one of `indices` / `lods` must be given. */
    indices?: Uint16Array | Uint32Array | number[];
    /** Multi-LOD form: pre-decimated index arrays, finest first. */
    lods?: MeshLODInput[];
}

export interface MeshletBuilderConfig {
    /** Float offset of the position (xyz) inside one vertex. Default 0. */
    positionOffsetFloats?: number;
    /**
     * Float offset of the normal (xyz) inside one vertex. Accepted for API symmetry with the
     * companion draw path, but unused: meshlet face normals are derived from the position
     * buffer (a meshlet's triangles are topologically closed, so derived facet normals equal
     * the faces' geometric normals), so the cone does not read the authored normal channel.
     */
    normalOffsetFloats?: number;
    /** Max unique vertices per meshlet. 1..255 (local indices are 8-bit). Default 64. */

    maxVertices?: number;
    /** Max triangles per meshlet. 1..124 (the companion draw expansion). Default 124. */
    maxTriangles?: number;
}

export interface MeshletReport {
    meshes: number;
    lods: number;
    meshlets: number;
    triangles: number;
    /** Σ vertexCount across meshlets (boundary vertices are re-listed per meshlet). */
    vertexRefs: number;
    uniqueVertices: number;
    /** vertexRefs / uniqueVertices — want < 1.2. */
    vertexReuse: number;
    /** want > 55 with the 64-vertex cap (design doc §4.2). */
    avgVerticesPerMeshlet: number;
    /** Sum of all packed buffers, in bytes. */
    bytes: number;
}

export interface MeshletPack {
    /** The shared megabuffer vertex pool. Upload: VERTEX | STORAGE. */
    megaVertices: Float32Array;
    /** Meshlet descriptors, 64 B stride (design doc §5.1). Upload: STORAGE. */
    meshlets: ArrayBuffer;
    /** Flat global vertex indices. Upload: STORAGE. */
    meshletVertices: Uint32Array;
    /** Flat packed local triangles (3 × 8-bit local indices per u32). Upload: STORAGE. */
    meshletTriangles: Uint32Array;
    /** MeshLOD records, 4 floats each [meshletOffset, meshletCount, error, 0]. Upload: STORAGE. */
    meshLODs: Float32Array;
    /** MeshEntry records, 4 u32 each [lodOffset, lodCount, 0, 0]. Upload: STORAGE. */
    meshTable: Uint32Array;
    report: MeshletReport;
}

// ---------------------------------------------------------------------------
// Internal records
// ---------------------------------------------------------------------------

interface LODRecord {
    meshletOffset: number;
    meshletCount: number;
    error: number;
}

interface MeshRecord {
    name: string;
    lodOffset: number;
    lodCount: number;
}

export class MeshletBuilder {
    private readonly mega: MegabufferBuilder;
    private readonly positionOffset: number;
    private readonly maxVertices: number;
    private readonly maxTriangles: number;


    // Packed-so-far outputs (arrays during build, typed arrays at build()).
    private readonly meshletDescs: number[] = [];    // 16 f32/u32 per meshlet (64 B stride)
    private readonly meshletVerts: number[] = [];    // flat global vertex indices
    private readonly meshletTris: number[] = [];     // flat packed local triangles
    private readonly lods: LODRecord[] = [];
    private readonly meshes: MeshRecord[] = [];
    private readonly names: Set<string> = new Set();

    constructor(mega: MegabufferBuilder, config: MeshletBuilderConfig = {}) {
        this.mega = mega;
        this.positionOffset = config.positionOffsetFloats ?? 0;

        const maxVertices = config.maxVertices ?? 64;

        const maxTriangles = config.maxTriangles ?? 124;

        // Clamp-correct configs are legal; exceeding the hard limits is not (design doc §11).
        if (maxVertices < 1) {
            throw new Error(`MeshletBuilder: maxVertices ${maxVertices} must be at least 1.`);
        }
        if (maxVertices > 255) {
            throw new Error(`MeshletBuilder: maxVertices ${maxVertices} exceeds 255; local indices are 8-bit.`);
        }
        if (maxTriangles < 1) {
            throw new Error(`MeshletBuilder: maxTriangles ${maxTriangles} must be at least 1.`);
        }
        if (maxTriangles > 124) {
            throw new Error(`MeshletBuilder: maxTriangles ${maxTriangles} exceeds 124; the companion draw path expands 124 triangles per meshlet.`);
        }
        this.maxVertices = maxVertices;
        this.maxTriangles = maxTriangles;
    }

    /**
     * Append a mesh to the shared megabuffer AND meshletize every one of its LOD levels.
     * Returns the handle an instance stores to reference this mesh (visibility doc §4).
     */
    public addMesh(name: string, input: MeshletMeshInput): MeshHandle {
        if (this.names.has(name)) {
            throw new Error(`MeshletBuilder: mesh name "${name}" already added.`);
        }

        let lodInputs: MeshLODInput[];
        if (input.lods) {
            if (input.indices) {
                throw new Error(`MeshletBuilder: mesh "${name}" supplies both indices and lods; use exactly one.`);
            }
            if (input.lods.length === 0) {
                throw new Error(`MeshletBuilder: mesh "${name}" supplies an empty lods array.`);
            }
            lodInputs = input.lods;
        } else if (input.indices) {
            lodInputs = [{ indices: input.indices, error: 0 }];
        } else {
            throw new Error(`MeshletBuilder: mesh "${name}" supplies neither indices nor lods.`);
        }

        const meshIndex = this.meshes.length;
        const lodOffset = this.lods.length;

        // Register once with the megabuffer so the mesh exists in the unified pool; the meshlet
        // tier reads vertices out of the pool directly (design doc §2.1 — the builder references,
        // never re-stores, vertices). LOD index arrays are meshlet-tier only, so each LOD's own
        // indices — not the megabuffer's copy — are meshletized below.
        this.mega.addMesh(name, input.vertices, lodInputs[0].indices);

        const offset = this.mega.getOffset(name);
        const baseVertex = offset.baseVertex;
        const vertexCount = (input.vertices.length / this.stride()) | 0;

        for (let k = 0; k < lodInputs.length; k++) {
            const lod = lodInputs[k];
            const idx = lod.indices;
            if (idx.length % 3 !== 0) {
                throw new Error(`MeshletBuilder: mesh "${name}" index array length ${idx.length} is not a multiple of 3.`);
            }
            if (idx.length === 0) {
                throw new Error(`MeshletBuilder: mesh "${name}" LOD ${k} has no triangles.`);
            }
            for (let i = 0; i < idx.length; i++) {
                if (idx[i] >= vertexCount) {
                    throw new Error(`MeshletBuilder: mesh "${name}" index ${i} references vertex ${idx[i]}, out of range (${vertexCount} vertices).`);
                }
            }

            const meshletOffset = this.meshletCount();
            this.meshletize(input.vertices, idx, baseVertex, this.positionOffset);

            this.lods.push({
                meshletOffset,
                meshletCount: this.meshletCount() - meshletOffset,
                error: lod.error,
            });
        }

        this.meshes.push({ name, lodOffset, lodCount: lodInputs.length });
        this.names.add(name);
        return { meshIndex, name };
    }

    /** Pack everything into transferable typed arrays. The shared megabuffer stays usable. */
    public build(): MeshletPack {
        const megaBuilt = this.mega.build();

        // --- meshlet descriptors: 64 B stride, written through dual views so member 12..15
        //     (offsets/counts) land as u32 while 0..11 stay f32 (design doc §5.1).
        const meshletBytes = this.meshletDescs.length * 4;
        const meshlets = new ArrayBuffer(meshletBytes);
        const meshletsF32 = new Float32Array(meshlets);
        const meshletsU32 = new Uint32Array(meshlets);
        // descs are stored as f32 bit patterns for 0..11 and raw integers for 12..15; we tagged
        // them by position when pushing, so re-dispatch per element here.
        for (let i = 0; i < this.meshletDescs.length; i++) {
            const member = i & 15;
            if (member < 12) meshletsF32[i] = this.meshletDescs[i];
            else meshletsU32[i] = this.meshletDescs[i];
        }

        const meshletVertices = new Uint32Array(this.meshletVerts);
        const meshletTriangles = new Uint32Array(this.meshletTris);

        // --- meshLODs: 16 B stride [meshletOffset, meshletCount, error, 0] as raw u32/f32 mix.
        const meshLODsBuf = new ArrayBuffer(this.lods.length * 16);
        const meshLODsF32 = new Float32Array(meshLODsBuf);
        const meshLODsU32 = new Uint32Array(meshLODsBuf);
        for (let i = 0; i < this.lods.length; i++) {
            const lod = this.lods[i];
            meshLODsU32[i * 4 + 0] = lod.meshletOffset;
            meshLODsU32[i * 4 + 1] = lod.meshletCount;
            meshLODsF32[i * 4 + 2] = lod.error;
            meshLODsF32[i * 4 + 3] = 0;
        }

        // --- meshTable: 16 B stride [lodOffset, lodCount, 0, 0].
        const meshTable = new Uint32Array(this.meshes.length * 4);
        for (let i = 0; i < this.meshes.length; i++) {
            meshTable[i * 4 + 0] = this.meshes[i].lodOffset;
            meshTable[i * 4 + 1] = this.meshes[i].lodCount;
            meshTable[i * 4 + 2] = 0;
            meshTable[i * 4 + 3] = 0;
        }

        const meshletCount = this.meshletCount();
        const totalTriangles = meshletTriangles.length;
        const vertexRefs = meshletVertices.length;
        const uniqueVertices = megaBuilt.megaVertices.length / this.stride();

        const bytes =
            megaBuilt.megaVertices.byteLength +
            meshletBytes +
            meshletVertices.byteLength +
            meshletTriangles.byteLength +
            meshLODsBuf.byteLength +
            meshTable.byteLength;

        const report: MeshletReport = {
            meshes: this.meshes.length,
            lods: this.lods.length,
            meshlets: meshletCount,
            triangles: totalTriangles,
            vertexRefs,
            uniqueVertices,
            vertexReuse: uniqueVertices > 0 ? vertexRefs / uniqueVertices : 0,
            avgVerticesPerMeshlet: meshletCount > 0 ? vertexRefs / meshletCount : 0,
            bytes,
        };

        return {
            megaVertices: megaBuilt.megaVertices,
            meshlets,
            meshletVertices,
            meshletTriangles,
            meshLODs: meshLODsF32,
            meshTable,
            report,
        };
    }

    // ---------------------------------------------------------------------------
    // Meshletizer core (design doc §4)
    // ---------------------------------------------------------------------------

    private stride(): number {
        // The shared megabuffer owns the stride (design doc §2.1); meshlet vertex math is in
        // units of it. Exposed as a public readonly on MegabufferBuilder.
        return this.mega.vertexStrideFloats;
    }


    private meshletCount(): number {
        return this.meshletDescs.length / 16;
    }

    /**
     * Greedy, index-order, single-pass meshletizer (design doc §4.1). Deterministic and O(T).
     * The vertex-count guard counts only NEW vertices against the local map (§4.2) — the single
     * line most likely to be written wrong.
     */
    private meshletize(
        vertices: Float32Array | number[],
        indices: ArrayLike<number>,
        baseVertex: number,
        posOffset: number,
    ): void {

        let localVerts: number[] = [];            // global indices, ≤ maxVertices
        let localMap: Map<number, number> = new Map();
        let tris: number[] = [];                  // packed local triples, ≤ maxTriangles

        const localIndexOf = (g: number): number => {
            let l = localMap.get(g);
            if (l === undefined) {
                l = localVerts.length;
                localMap.set(g, l);
                localVerts.push(g);
            }
            return l;
        };

        const flush = (): void => {
            if (tris.length === 0) return;
            this.emitMeshlet(vertices, localVerts, tris, baseVertex, posOffset);

            localVerts = [];
            localMap = new Map();
            tris = [];
        };

        const triCount = indices.length / 3;
        for (let t = 0; t < triCount; t++) {
            const a = indices[t * 3 + 0];
            const b = indices[t * 3 + 1];
            const c = indices[t * 3 + 2];

            // Count only vertices NOT already local to this meshlet (design doc §4.2).
            const newVerts =
                (localMap.has(a) ? 0 : 1) +
                (localMap.has(b) ? 0 : 1) +
                (localMap.has(c) ? 0 : 1);

            if (localVerts.length + newVerts > this.maxVertices ||
                tris.length + 1 > this.maxTriangles) {
                flush();
            }

            const la = localIndexOf(a);
            const lb = localIndexOf(b);
            const lc = localIndexOf(c);
            tris.push(la | (lb << 8) | (lc << 16));
        }
        flush();
    }

    // ---------------------------------------------------------------------------
    // Descriptor emission: bounds + cone (design doc §6)
    // ---------------------------------------------------------------------------

    private emitMeshlet(
        vertices: Float32Array | number[],
        localVerts: number[],
        tris: number[],
        baseVertex: number,
        posOffset: number,
    ): void {

        const stride = this.stride();
        const vertexOffset = this.meshletVerts.length;
        const triangleOffset = this.meshletTris.length;
        const vertexCount = localVerts.length;
        const triangleCount = tris.length;

        // ---- bounding sphere: AABB-center + farthest-vertex radius (design doc §6.1).
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        for (let i = 0; i < vertexCount; i++) {
            const g = localVerts[i];
            const base = g * stride + posOffset;
            const x = vertices[base], y = vertices[base + 1], z = vertices[base + 2];
            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (y < minY) minY = y; if (y > maxY) maxY = y;
            if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
        }
        const cx = 0.5 * (minX + maxX);
        const cy = 0.5 * (minY + maxY);
        const cz = 0.5 * (minZ + maxZ);
        let radius = 0;
        for (let i = 0; i < vertexCount; i++) {
            const g = localVerts[i];
            const base = g * stride + posOffset;
            const dx = vertices[base] - cx;
            const dy = vertices[base + 1] - cy;
            const dz = vertices[base + 2] - cz;
            const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (d > radius) radius = d;
        }

        // ---- normal cone (design doc §6.2).
        //
        // Face normals are derived from the position buffer (cross of the two triangle edges,
        // area-weighted). A meshlet is topologically closed — every face it references is fully
        // contained in it — so the derived facet normals are exactly the faces' geometric
        // normals, and the cone does not depend on the authored normal channel. The cone
        // degenerates correctly for flat clusters (cutoff → 0) and for clusters spanning >90°
        // (cutoff → 2.0, the never-cull sentinel).
        let axisX = 0, axisY = 0, axisZ = 0;
        const faceNormals: number[] = new Array(triangleCount * 3);
        const faceAreas: number[] = new Array(triangleCount);


        for (let t = 0; t < triangleCount; t++) {
            const packed = tris[t];
            const l0 = packed & 0xFF;
            const l1 = (packed >> 8) & 0xFF;
            const l2 = (packed >> 16) & 0xFF;
            const g0 = localVerts[l0], g1 = localVerts[l1], g2 = localVerts[l2];

            const b0 = g0 * stride + posOffset;
            const b1 = g1 * stride + posOffset;
            const b2 = g2 * stride + posOffset;

            const p0x = vertices[b0], p0y = vertices[b0 + 1], p0z = vertices[b0 + 2];
            const e1x = vertices[b1] - p0x, e1y = vertices[b1 + 1] - p0y, e1z = vertices[b1 + 2] - p0z;
            const e2x = vertices[b2] - p0x, e2y = vertices[b2 + 1] - p0y, e2z = vertices[b2 + 2] - p0z;

            // cross(e1, e2) — magnitude is 2× the triangle area.
            let nx = e1y * e2z - e1z * e2y;
            let ny = e1z * e2x - e1x * e2z;
            let nz = e1x * e2y - e1y * e2x;
            const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
            const area = len * 0.5;
            faceAreas[t] = area;
            if (len > 1e-12) { nx /= len; ny /= len; nz /= len; }
            faceNormals[t * 3 + 0] = nx;
            faceNormals[t * 3 + 1] = ny;
            faceNormals[t * 3 + 2] = nz;

            // area-weighted accumulation (zero-area faces contribute nothing).
            axisX += nx * area; axisY += ny * area; axisZ += nz * area;
        }


        let axisLen = Math.sqrt(axisX * axisX + axisY * axisY + axisZ * axisZ);
        if (axisLen > 1e-12) { axisX /= axisLen; axisY /= axisLen; axisZ /= axisLen; }
        else { axisX = 0; axisY = 1; axisZ = 0; } // degenerate cluster: arbitrary axis, cutoff will be 2.0

        // Tightest opening across non-degenerate faces.
        let minCos = Infinity;
        for (let t = 0; t < triangleCount; t++) {
            if (faceAreas[t] <= 1e-12) continue;
            const d = faceNormals[t * 3] * axisX + faceNormals[t * 3 + 1] * axisY + faceNormals[t * 3 + 2] * axisZ;
            if (d < minCos) minCos = d;
        }
        if (minCos === Infinity) minCos = 1; // all faces degenerate → treat as flat

        // The sentinel: a cluster spanning >90° can be visible from anywhere — never cull it.
        const cutoff = (minCos <= 0) ? 2.0 : Math.sqrt(Math.max(0, 1 - minCos * minCos));

        // ---- emit descriptor (16 members = 64 B, design doc §5.1).

        // members 0..11 are f32 (boundsCenter, coneApex, coneAxis), 12..15 are u32 (offsets).
        this.meshletDescs.push(
            cx, cy, cz, radius,          // boundsCenter.xyz, .w = radius
            cx, cy, cz, 0,               // coneApex.xyz = sphere center (conservative), .w unused
            axisX, axisY, axisZ, cutoff, // coneAxis.xyz (normalized), .w = cutoff
            vertexOffset, triangleOffset, vertexCount, triangleCount,
        );

        for (let i = 0; i < vertexCount; i++) this.meshletVerts.push(baseVertex + localVerts[i]);
        for (let i = 0; i < triangleCount; i++) this.meshletTris.push(tris[i]);
    }
}
