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
}