// GLTF Constants
export const GLB_MAGIC = 0x46546C67; // "glTF"

// types.ts
export interface RawGeometryData {
    vertices: Float32Array;
    indices: Uint16Array | Uint32Array;
    vertexCount: number;
    indexCount: number;
    isSkinned: boolean;
    getWebGPULayout: () => GPUVertexBufferLayout[];

    // The Full PBR Suite!
    albedoUrl?: string;
    normalUrl?: string;
    metallicRoughnessUrl?: string;
    occlusionUrl?: string;
    emissiveUrl?: string;

    /**
     * The node this mesh was found under, an index into `GLBData.nodes`. Unless skinned, its
     * vertices are baked at that node's rest transform. Set by `GLBParser.parse` and `load`, not by
     * `loadMerged`, which merges the meshes.
     */
    node?: number;
}

/** A glTF node as the file holds it: the fields NullGraph reads. */
export interface GLBNode {
    name?: string;
    children?: number[];
    mesh?: number;
    skin?: number;
    /** Column-major 4×4; a node has either this or translation, rotation and scale. */
    matrix?: number[];
    translation?: number[];
    /** A quaternion, [x, y, z, w]. */
    rotation?: number[];
    scale?: number[];
}

export interface GLBSkin {
    joints: number[]; // Array of node indices representing bones
    inverseBindMatrices: Float32Array; // Flat array of mat4s
    nodes: any[]; // The actual node hierarchy data from the glTF
}

export interface GLBAnimation {
    name: string;
    channels: { targetNode: number, path: 'translation' | 'rotation' | 'scale', samplerIndex: number }[];
    samplers: { input: Float32Array, output: Float32Array, interpolation: string }[];
    maxTime: number;
}

export interface GLBData{
    meshes:RawGeometryData[] | null,
    skin:GLBSkin | null,
    animations:GLBAnimation[] | null,
    imageUrls?: string[];
    /** Every node of the file, in its order: the parts and parents an animation moves. */
    nodes?: GLBNode[];
}