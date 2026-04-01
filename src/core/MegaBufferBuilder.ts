// src/MegabufferBuilder.ts

export interface MeshOffset {
    indexCount: number;
    firstIndex: number;
    baseVertex: number;
    meshID: number;
}

export class MegabufferBuilder {
    private vertices: number[] = [];
    private indices: number[] = [];
    private offsets: Map<string, MeshOffset> = new Map();

    private currentMeshID: number = 0;
    private vertexStrideFloats: number;

    /**
     * @param vertexStrideFloats How many floats make up one vertex?
     * (e.g., 6 for [X,Y,Z, NX,NY,NZ])
     */
    constructor(vertexStrideFloats: number = 6) {
        this.vertexStrideFloats = vertexStrideFloats;
    }

    public addMesh(name: string, meshVertices: Float32Array | number[], meshIndices: Uint16Array | number[]): MeshOffset {
        // 1. Calculate the offsets before adding the new data
        // baseVertex is the number of VERTICES currently in the buffer, not the number of floats!
        const baseVertex = this.vertices.length / this.vertexStrideFloats;
        const firstIndex = this.indices.length;

        // 2. Safely push the typed array data into our master arrays
        for (let i = 0; i < meshVertices.length; i++) this.vertices.push(meshVertices[i]);
        for (let i = 0; i < meshIndices.length; i++) this.indices.push(meshIndices[i]);

        // 3. Save the math
        const offset: MeshOffset = {
            indexCount: meshIndices.length,
            firstIndex: firstIndex,
            baseVertex: baseVertex,
            meshID: this.currentMeshID++ // Auto-increment the ID for the Compute Shader!
        };

        this.offsets.set(name, offset);
        return offset;
    }

    public build() {
        // WebGPU Safety Check: If we have massive geometry, we must use 32-bit indices!
        const totalVertices = this.vertices.length / this.vertexStrideFloats;
        const useUint32 = totalVertices > 65535;

        return {
            megaVertices: new Float32Array(this.vertices),
            megaIndices: useUint32 ? new Uint32Array(this.indices) : new Uint16Array(this.indices),
            indexFormat: (useUint32 ? 'uint32' : 'uint16') as GPUIndexFormat
        };
    }

    public getOffset(name: string): MeshOffset {
        const offset = this.offsets.get(name);
        if (!offset) throw new Error(`Mesh '${name}' not found in the Megabuffer!`);
        return offset;
    }
}