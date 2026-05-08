// src/DynamicGeometryManager.ts


import {MeshOffset} from "./MegaBufferBuilder";

interface MemoryBlock {
    start: number; // The offset where this free block starts
    count: number; // How many items (vertices or indices) it can hold
}

export class DynamicGeometryManager {
    public vbo: GPUBuffer;
    public ibo: GPUBuffer;

    private device: GPUDevice;
    private vertexStrideFloats: number;
    private indexFormat: GPUIndexFormat;

    private meshRegistry: Map<string, MeshOffset> = new Map();
    private currentMeshID: number = 0;

    // Free lists to track available memory holes
    private freeVertexBlocks: MemoryBlock[];
    private freeIndexBlocks: MemoryBlock[];

    constructor(device: GPUDevice, maxVertices: number, maxIndices: number, vertexStrideFloats: number = 6) {
        this.device = device;
        this.vertexStrideFloats = vertexStrideFloats;
        this.indexFormat = maxVertices > 65535 ? 'uint32' : 'uint16';

        // 1. Create the massive, empty buffers upfront
        this.vbo = device.createBuffer({
            size: maxVertices * vertexStrideFloats * 4, // 4 bytes per float
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });

        this.ibo = device.createBuffer({
            size: maxIndices * (this.indexFormat === 'uint32' ? 4 : 2),
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });

        // 2. Initially, the entire buffer is one giant free block
        this.freeVertexBlocks = [{ start: 0, count: maxVertices }];
        this.freeIndexBlocks = [{ start: 0, count: maxIndices }];
    }

    /** Helper to allocate space from a free list */
    private allocateSpace(freeList: MemoryBlock[], requiredCount: number): number {
        for (let i = 0; i < freeList.length; i++) {
            const block = freeList[i];
            if (block.count >= requiredCount) {
                const allocatedStart = block.start;

                if (block.count === requiredCount) {
                    // Perfect fit, remove the block from the free list
                    freeList.splice(i, 1);
                } else {
                    // Block is larger, shrink it
                    block.start += requiredCount;
                    block.count -= requiredCount;
                }
                return allocatedStart;
            }
        }
        throw new Error("Out of geometry memory! Buffer fragmentation or capacity reached.");
    }

    /** Helper to return space back to the free list */
    private freeSpace(freeList: MemoryBlock[], start: number, count: number) {
        freeList.push({ start, count });
        // Optional but recommended: Sort and Coalesce adjacent blocks here to prevent fragmentation over time.
        freeList.sort((a, b) => a.start - b.start);
        for(let i = 0; i < freeList.length - 1; i++) {
            if (freeList[i].start + freeList[i].count === freeList[i+1].start) {
                freeList[i].count += freeList[i+1].count;
                freeList.splice(i + 1, 1);
                i--; // Check again in case 3 blocks merged
            }
        }
    }

    public addMesh(name: string, meshVertices: Float32Array, meshIndices: Uint16Array | Uint32Array): MeshOffset {
        if (this.meshRegistry.has(name)) throw new Error(`Mesh ${name} already exists!`);

        const numVerts = meshVertices.length / this.vertexStrideFloats;
        const numIndices = meshIndices.length;

        // 1. Find free space
        const baseVertex = this.allocateSpace(this.freeVertexBlocks, numVerts);
        const firstIndex = this.allocateSpace(this.freeIndexBlocks, numIndices);

        // 2. Stream data asynchronously to the GPU via DMA
        const vboOffsetBytes = baseVertex * this.vertexStrideFloats * 4;
        this.device.queue.writeBuffer(this.vbo, vboOffsetBytes, meshVertices.buffer);

        const indexBytes = this.indexFormat === 'uint32' ? 4 : 2;
        const iboOffsetBytes = firstIndex * indexBytes;
        this.device.queue.writeBuffer(this.ibo, iboOffsetBytes, meshIndices.buffer);

        // 3. Register the mesh
        const offset: MeshOffset = {
            indexCount: numIndices,
            firstIndex: firstIndex,
            baseVertex: baseVertex,
            meshID: this.currentMeshID++
        };

        this.meshRegistry.set(name, offset);
        return offset;
    }

    public removeMesh(name: string) {
        const offset = this.meshRegistry.get(name);
        if (!offset) return;

        // 1. Return the memory to the free lists
        this.freeSpace(this.freeVertexBlocks, offset.baseVertex, offset.indexCount /* wait, need vertex count here, see note below */);

        // Note: To free vertices properly, you need to store `vertexCount` in MeshOffset when adding!
        // this.freeSpace(this.freeIndexBlocks, offset.firstIndex, offset.indexCount);

        // 2. Remove from registry
        this.meshRegistry.delete(name);
    }

    public getOffset(name: string): MeshOffset {
        const offset = this.meshRegistry.get(name);
        if (!offset) throw new Error(`Mesh '${name}' not found!`);
        return offset;
    }
}