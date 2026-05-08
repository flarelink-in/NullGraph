// --- TINY MATH HELPER FOR PARSING ---
import {GLB_MAGIC, GLBAnimation, GLBData, GLBSkin, RawGeometryData} from "./types";
import { MathUtils } from "./MathUtils";

export class GLBParser {
    static async load(url: string): Promise<GLBData> {
        const response = await fetch(url);
        const arrayBuffer = await response.arrayBuffer();
        return this.parse(arrayBuffer);
    }

    static parse(arrayBuffer: ArrayBuffer): GLBData {
        const dataView = new DataView(arrayBuffer);
        const magic = dataView.getUint32(0, true);
        if (magic !== GLB_MAGIC) throw new Error("Invalid GLB.");

        const jsonChunkLength = dataView.getUint32(12, true);
        const jsonBytes = new Uint8Array(arrayBuffer, 20, jsonChunkLength);
        const gltf = JSON.parse(new TextDecoder('utf-8').decode(jsonBytes));

        const binChunkOffset = 20 + jsonChunkLength;
        const binBufferOffset = binChunkOffset + 8;
        const geometries: RawGeometryData[] = [];
        let skin:GLBSkin | null=null;
        let animations:GLBAnimation[]=[];

        let imageUrls:string[] = this.extractImageURLs(gltf, arrayBuffer, binBufferOffset);

        if (!gltf.scenes || !gltf.nodes)
            return {
                meshes: geometries,
                skin: skin,
                animations: animations
            };
        const defaultScene = gltf.scenes[gltf.scene || 0];
        const identityMatrix = MathUtils.identity();

        for (const nodeIndex of defaultScene.nodes) {
            this.processNode(gltf, nodeIndex, identityMatrix, arrayBuffer, binBufferOffset, geometries,imageUrls);
        }

        skin = this.extractSkin(gltf, arrayBuffer, binBufferOffset);
        animations = this.extractAnimations(gltf, arrayBuffer, binBufferOffset);

        return {
            meshes: geometries,
            skin: skin,
            animations: animations,
            imageUrls: imageUrls
        };
    }

    private static processNode(gltf: any, nodeIndex: number, parentMatrix: Float32Array, buffer: ArrayBuffer, binOffset: number, geometries: RawGeometryData[],imageUrls:string[]) {
        const node = gltf.nodes[nodeIndex];
        let localMatrix = MathUtils.identity();

        if (node.matrix) {
            localMatrix = new Float32Array(node.matrix);
        } else if (node.translation || node.rotation || node.scale) {
            const t = node.translation || [0, 0, 0];
            const r = node.rotation || [0, 0, 0, 1];
            const s = node.scale || [1, 1, 1];
            localMatrix = MathUtils.fromTRS(t, r, s);
        }

        const worldMatrix = MathUtils.multiply(parentMatrix, localMatrix);

        if (node.mesh !== undefined) {
            const mesh = gltf.meshes[node.mesh];
            // 1. CHECK IF NODE IS SKINNED
            const isSkinned = node.skin !== undefined;

            for (const primitive of mesh.primitives) {
                // 2. PASS isSkinned TO EXTRACTOR
                const geom = this.extractPrimitive(gltf, buffer, binOffset, primitive, isSkinned,imageUrls);

                // 3. ONLY BAKE IF IT IS A STATIC MESH
                if (!isSkinned) {
                    this.bakeTransform(geom, worldMatrix);
                }
                geometries.push(geom);
            }
        }

        if (node.children) {
            for (const childIndex of node.children) {
                this.processNode(gltf, childIndex, worldMatrix, buffer, binOffset, geometries,imageUrls);
            }
        }
    }

    private static bakeTransform(geom: RawGeometryData, matrix: Float32Array) {
        // Safe to hardcode 8 here, because we ONLY call this if !isSkinned
        const stride = 8;
        for (let i = 0; i < geom.vertexCount; i++) {
            const base = i * stride;
            const px = geom.vertices[base + 0], py = geom.vertices[base + 1], pz = geom.vertices[base + 2];
            const nx = geom.vertices[base + 3], ny = geom.vertices[base + 4], nz = geom.vertices[base + 5];

            const newPos = MathUtils.transformPos([px, py, pz], matrix);
            const newNorm = MathUtils.transformNorm([nx, ny, nz], matrix);

            geom.vertices[base + 0] = newPos[0]; geom.vertices[base + 1] = newPos[1]; geom.vertices[base + 2] = newPos[2];
            geom.vertices[base + 3] = newNorm[0]; geom.vertices[base + 4] = newNorm[1]; geom.vertices[base + 5] = newNorm[2];
        }
    }

    // --- ADDED isSkinned PARAMETER ---
    private static extractPrimitive(gltf: any, buffer: ArrayBuffer, binBufferOffset: number, primitive: any, isSkinned: boolean, imageUrls: string[]): RawGeometryData {
        const posAccessor = gltf.accessors[primitive.attributes.POSITION];
        const normAccessor = primitive.attributes.NORMAL !== undefined ? gltf.accessors[primitive.attributes.NORMAL] : null;
        const uvAccessor = primitive.attributes.TEXCOORD_0 !== undefined ? gltf.accessors[primitive.attributes.TEXCOORD_0] : null;

        // ONLY grab joint/weight accessors if it is skinned
        const jointAccessor = (isSkinned && primitive.attributes.JOINTS_0 !== undefined) ? gltf.accessors[primitive.attributes.JOINTS_0] : null;
        const weightAccessor = (isSkinned && primitive.attributes.WEIGHTS_0 !== undefined) ? gltf.accessors[primitive.attributes.WEIGHTS_0] : null;

        const vertexCount = posAccessor.count;

        let albedoUrl: string | undefined;
        let normalUrl: string | undefined;
        let metallicRoughnessUrl: string | undefined;
        let occlusionUrl: string | undefined;
        let emissiveUrl: string | undefined;

        // 4. DYNAMIC STRIDE
        const stride = isSkinned ? 16 : 8;
        const vertices = new Float32Array(vertexCount * stride);
        const dataView = new DataView(buffer);

        // 5. UPGRADED READ ATTRIBUTE (Handles Floats AND Integers safely)
        const readAttribute = (accessor: any, targetOffset: number, numComponents: number) => {
            if (!accessor) return;
            const bufferView = gltf.bufferViews[accessor.bufferView];
            const byteStride = bufferView.byteStride || (numComponents * (accessor.componentType === 5126 ? 4 : (accessor.componentType === 5123 ? 2 : 1)));
            const startByte = binBufferOffset + (bufferView.byteOffset || 0) + (accessor.byteOffset || 0);
            const compType = accessor.componentType;

            for (let i = 0; i < vertexCount; i++) {
                for (let j = 0; j < numComponents; j++) {
                    const byteOffset = startByte + (i * byteStride) + (j * (compType === 5126 ? 4 : (compType === 5123 ? 2 : 1)));

                    let val = 0;
                    if (compType === 5126) val = dataView.getFloat32(byteOffset, true); // FLOAT
                    else if (compType === 5123) val = dataView.getUint16(byteOffset, true); // UINT16 (Joints)
                    else if (compType === 5121) val = dataView.getUint8(byteOffset); // UINT8 (Joints)

                    vertices[(i * stride) + targetOffset + j] = val;
                }
            }
        };

        readAttribute(posAccessor, 0, 3);

        if (normAccessor) readAttribute(normAccessor, 3, 3);
        else for(let i=0; i<vertexCount; i++) { vertices[i*stride + 3] = 0; vertices[i*stride + 4] = 1; vertices[i*stride + 5] = 0; }

        if (uvAccessor) readAttribute(uvAccessor, 6, 2);
        else for(let i=0; i<vertexCount; i++) { vertices[i*stride + 6] = 0; vertices[i*stride + 7] = 0; }

        if (isSkinned) {
            if (jointAccessor) readAttribute(jointAccessor, 8, 4);
            else for(let i=0; i<vertexCount; i++) { vertices[i*stride + 8] = 0; }

            if (weightAccessor) readAttribute(weightAccessor, 12, 4);
            else for(let i=0; i<vertexCount; i++) { vertices[i*stride + 12] = 1; }
        }

        const indexAccessor = gltf.accessors[primitive.indices];
        const indexBufferView = gltf.bufferViews[indexAccessor.bufferView];
        const isUint32 = indexAccessor.componentType === 5125;
        const isUint8  = indexAccessor.componentType === 5121;
        const indexStride = isUint32 ? 4 : (isUint8 ? 1 : 2);
        const indices = isUint32 ? new Uint32Array(indexAccessor.count) : new Uint16Array(indexAccessor.count);
        const indexStart = binBufferOffset + (indexBufferView.byteOffset || 0) + (indexAccessor.byteOffset || 0);

        for (let i = 0; i < indexAccessor.count; i++) {
            const byteOffset = indexStart + (i * indexStride);
            if (isUint32) indices[i] = dataView.getUint32(byteOffset, true);
            else if (isUint8) indices[i] = dataView.getUint8(byteOffset);
            else indices[i] = dataView.getUint16(byteOffset, true);
        }

        if (primitive.material !== undefined && gltf.materials) {
            const material = gltf.materials[primitive.material];

            // Helper function to safely get an image URL from a texture object
            const getUrl = (textureInfo: any) => {
                if (!textureInfo || textureInfo.index === undefined) return undefined;
                const texture = gltf.textures ? gltf.textures[textureInfo.index] : undefined;
                if (!texture || texture.source === undefined) return undefined;
                return imageUrls[texture.source];
            };

            // 1. Albedo & Metallic-Roughness (Live inside pbrMetallicRoughness)
            if (material.pbrMetallicRoughness) {
                albedoUrl = getUrl(material.pbrMetallicRoughness.baseColorTexture);
                metallicRoughnessUrl = getUrl(material.pbrMetallicRoughness.metallicRoughnessTexture);
            }

            // 2. Normal, Occlusion, and Emissive (Live at the root of the material)
            normalUrl = getUrl(material.normalTexture);
            occlusionUrl = getUrl(material.occlusionTexture);
            emissiveUrl = getUrl(material.emissiveTexture);
        }

        // 6. RETURN DYNAMIC LAYOUT
        return {
            vertices, indices, indexCount: indexAccessor.count, vertexCount: vertexCount, isSkinned,
            albedoUrl, normalUrl, metallicRoughnessUrl, occlusionUrl, emissiveUrl,
            getWebGPULayout: () => {
                if (isSkinned) {
                    return [{
                        arrayStride: 64, // 16 floats
                        attributes: [
                            { format: 'float32x3', offset: 0,  shaderLocation: 0 },
                            { format: 'float32x3', offset: 12, shaderLocation: 1 },
                            { format: 'float32x2', offset: 24, shaderLocation: 2 },
                            { format: 'float32x4', offset: 32, shaderLocation: 3 }, // Joints mapped to float32x4 in WGSL!
                            { format: 'float32x4', offset: 48, shaderLocation: 4 }  // Weights
                        ]
                    }];
                } else {
                    return [{
                        arrayStride: 32, // 8 floats
                        attributes: [
                            { format: 'float32x3', offset: 0,  shaderLocation: 0 },
                            { format: 'float32x3', offset: 12, shaderLocation: 1 },
                            { format: 'float32x2', offset: 24, shaderLocation: 2 }
                        ]
                    }];
                }
            }
        };
    }

    static async loadMerged(url: string): Promise<GLBData> {
        // 1. Await the new GLBData object
        const glbData = await this.load(url);
        const geometries = glbData.meshes;

        // 2. Safety checks on the meshes array inside the object
        if (!geometries || geometries.length === 0) {
            throw new Error(`No geometry found in ${url}`);
        }

        // If it's already just one mesh, return the whole GLBData directly
        if (geometries.length === 1) {
            return glbData;
        }

        let totalVertexCount = 0;
        let totalIndexCount = 0;
        for (const geom of geometries) {
            totalVertexCount += geom.vertexCount;
            totalIndexCount += geom.indexCount;
        }

        // 3. Fix the stride check
        const isSkinned = geometries[0].isSkinned;
        const STRIDE = isSkinned ? 16 : 8;

        const mergedVertices = new Float32Array(totalVertexCount * STRIDE);
        const mergedIndices = new Uint32Array(totalIndexCount);

        let vFloatOffset = 0;
        let iOffset = 0;
        let vertexIndexOffset = 0;

        for (const geom of geometries) {
            mergedVertices.set(geom.vertices, vFloatOffset);

            for (let i = 0; i < geom.indexCount; i++) {
                mergedIndices[iOffset + i] = geom.indices[i] + vertexIndexOffset;
            }

            vFloatOffset += geom.vertices.length;
            iOffset += geom.indexCount;
            vertexIndexOffset += geom.vertexCount;
        }

        // 4. Create the single merged mesh
        const mergedMesh: RawGeometryData = {
            vertices: mergedVertices,
            indices: mergedIndices,
            vertexCount: totalVertexCount,
            indexCount: totalIndexCount,
            isSkinned,
            getWebGPULayout: geometries[0].getWebGPULayout
        };

        // 5. Return the full GLBData object with the single merged mesh
        return {
            meshes: [mergedMesh],
            skin: glbData.skin,
            animations: glbData.animations
        };
    }

    static extractSkin(gltf: any, buffer: ArrayBuffer, binBufferOffset: number): GLBSkin | null {
        if (!gltf.skins || gltf.skins.length === 0) return null;
        const skin = gltf.skins[0];
        const joints = skin.joints;
        const accessor = gltf.accessors[skin.inverseBindMatrices];
        const bufferView = gltf.bufferViews[accessor.bufferView];
        const startByte = binBufferOffset + (bufferView.byteOffset || 0) + (accessor.byteOffset || 0);
        const inverseBindMatrices = new Float32Array(buffer.slice(startByte, startByte + accessor.count * 16 * 4));
        return { joints, inverseBindMatrices, nodes: gltf.nodes };
    }

    static extractAnimations(gltf: any, buffer: ArrayBuffer, binBufferOffset: number): GLBAnimation[] {
        if (!gltf.animations) return [];

        return gltf.animations.map((anim: any) => {
            let maxTime = 0;
            const samplers = anim.samplers.map((samp: any) => {
                const inputAcc = gltf.accessors[samp.input];
                const inputView = gltf.bufferViews[inputAcc.bufferView];
                const inputStart = binBufferOffset + (inputView.byteOffset || 0) + (inputAcc.byteOffset || 0);
                const inputData = new Float32Array(buffer.slice(inputStart, inputStart + inputAcc.count * 4));

                if (inputData[inputData.length - 1] > maxTime) maxTime = inputData[inputData.length - 1];

                const outputAcc = gltf.accessors[samp.output];
                const outputView = gltf.bufferViews[outputAcc.bufferView];
                const outputStart = binBufferOffset + (outputView.byteOffset || 0) + (outputAcc.byteOffset || 0);
                const compCount = outputAcc.type === 'VEC3' ? 3 : 4;
                const outputData = new Float32Array(buffer.slice(outputStart, outputStart + outputAcc.count * compCount * 4));

                return { input: inputData, output: outputData, interpolation: samp.interpolation || 'LINEAR' };
            });

            const channels = anim.channels.map((chan: any) => ({
                targetNode: chan.target.node,
                path: chan.target.path,
                samplerIndex: chan.sampler
            }));

            return { name: anim.name || 'Animation', channels, samplers, maxTime };
        });
    }

    static extractImageURLs(gltf: any, buffer: ArrayBuffer, binBufferOffset: number): string[] {
        if (!gltf.images) return [];

        return gltf.images.map((img: any, index: number) => {
            // 1. Handle embedded Base64 strings (rare in standard GLB, but valid glTF)
            if (img.uri && img.uri.startsWith('data:')) {
                return img.uri;
            }

            // 2. Handle Binary Buffer Views
            if (img.bufferView !== undefined) {
                const bufferView = gltf.bufferViews[img.bufferView];
                const startByte = binBufferOffset + (bufferView.byteOffset || 0);
                const imageBytes = new Uint8Array(buffer, startByte, bufferView.byteLength);

                // STRICT FALLBACK: Browsers will crash if this is undefined!
                const mime = img.mimeType || 'image/png';

                // Debug log to see what we are dealing with
                console.log(`[GLBParser] Image ${index}: ${bufferView.byteLength} bytes, Type: ${mime}`);

                const blob = new Blob([imageBytes], { type: mime });
                return URL.createObjectURL(blob);
            }

            return "";
        });
    }

    static disposeImages(gltf:GLBData) {
        if (gltf.imageUrls) {
            gltf.imageUrls.forEach(url => URL.revokeObjectURL(url));
        }
    }
}