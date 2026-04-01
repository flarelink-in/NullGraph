// null-graph/geometry/Primitives.ts

import {VertexAttribute, VertexLayout} from "./VertexLayout";
import {Geometry} from "./Geometry";

export class Primitives {

    public static createCube(layout: VertexLayout, width: number, height: number, depth: number): Geometry {
        // 1. Calculate half-extents for centering the cube at (0,0,0)
        const w = width / 2;
        const h = height / 2;
        const d = depth / 2;

        // 2. Define the 6 faces.
        // Order: Bottom-Left, Bottom-Right, Top-Right, Top-Left
        const faces = [
            // Front (Z = d)
            { normal: [0, 0, 1], corners: [[-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d]] },
            // Back (Z = -d)
            { normal: [0, 0, -1], corners: [[w, -h, -d], [-w, -h, -d], [-w, h, -d], [w, h, -d]] },
            // Right (X = w)
            { normal: [1, 0, 0], corners: [[w, -h, d], [w, -h, -d], [w, h, -d], [w, h, d]] },
            // Left (X = -w)
            { normal: [-1, 0, 0], corners: [[-w, -h, -d], [-w, -h, d], [-w, h, d], [-w, h, -d]] },
            // Top (Y = h)
            { normal: [0, 1, 0], corners: [[-w, h, d], [w, h, d], [w, h, -d], [-w, h, -d]] },
            // Bottom (Y = -h)
            { normal: [0, -1, 0], corners: [[-w, -h, -d], [w, -h, -d], [w, -h, d], [-w, -h, d]] }
        ];

        // Standard UV mapping for a quad
        const uvs = [[0, 1], [1, 1], [1, 0], [0, 0]];

        // 3. Allocate the exact amount of memory needed
        const numVertices = 24; // 6 faces * 4 vertices
        const vertices = new Float32Array(numVertices * layout.strideFloats);
        let vIdx = 0;

        // 4. Build the Vertex Array dynamically!
        for (let f = 0; f < faces.length; f++) {
            const face = faces[f];

            for (let v = 0; v < 4; v++) {
                const pos = face.corners[v];
                const uv = uvs[v];

                // THE MAGIC: We only write the data the Layout specifically asks for!
                for (const attr of layout.attributes) {
                    switch (attr) {
                        case VertexAttribute.Position:
                            vertices[vIdx++] = pos[0];
                            vertices[vIdx++] = pos[1];
                            vertices[vIdx++] = pos[2];
                            break;
                        case VertexAttribute.Normal:
                            vertices[vIdx++] = face.normal[0];
                            vertices[vIdx++] = face.normal[1];
                            vertices[vIdx++] = face.normal[2];
                            break;
                        case VertexAttribute.UV:
                            vertices[vIdx++] = uv[0];
                            vertices[vIdx++] = uv[1];
                            break;
                        case VertexAttribute.Color:
                            // Default to white if a color attribute is requested
                            vertices[vIdx++] = 1.0; vertices[vIdx++] = 1.0;
                            vertices[vIdx++] = 1.0; vertices[vIdx++] = 1.0;
                            break;
                    }
                }
            }
        }

        // 5. Generate the Indices (Connecting the 4 vertices into 2 triangles per face)
        const indices = new Uint16Array(36); // 6 faces * 6 indices
        let iIdx = 0;

        for (let i = 0; i < 6; i++) {
            const base = i * 4;
            // Triangle 1
            indices[iIdx++] = base + 0;
            indices[iIdx++] = base + 1;
            indices[iIdx++] = base + 2;
            // Triangle 2
            indices[iIdx++] = base + 2;
            indices[iIdx++] = base + 3;
            indices[iIdx++] = base + 0;
        }

        return new Geometry(layout, vertices, indices);
    }

    public static createOctahedron(layout: VertexLayout, width: number, height: number, depth: number): Geometry {
        // 1. Calculate half-extents for centering the octahedron at (0,0,0)
        const w = width / 2;
        const h = height / 2;
        const d = depth / 2;

        // 2. Define the 6 structural vertices of the octahedron
        const top    = [ 0,  h,  0];
        const bottom = [ 0, -h,  0];
        const right  = [ w,  0,  0];
        const left   = [-w,  0,  0];
        const front  = [ 0,  0,  d];
        const back   = [ 0,  0, -d];

        // Helper to calculate the normal of a triangle (using Cross Product)
        const calcNormal = (v0: number[], v1: number[], v2: number[]) => {
            const dx1 = v1[0] - v0[0], dy1 = v1[1] - v0[1], dz1 = v1[2] - v0[2];
            const dx2 = v2[0] - v0[0], dy2 = v2[1] - v0[1], dz2 = v2[2] - v0[2];

            const nx = dy1 * dz2 - dz1 * dy2;
            const ny = dz1 * dx2 - dx1 * dz2;
            const nz = dx1 * dy2 - dy1 * dx2;

            const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
            return [nx / len, ny / len, nz / len];
        };

        // 3. Define the 8 triangular faces using Counter-Clockwise (CCW) winding
        const faces = [
            // Top hemisphere (Y > 0)
            { corners: [right, top, front] },
            { corners: [front, top, left] },
            { corners: [left, top, back] },
            { corners: [back, top, right] },
            // Bottom hemisphere (Y < 0)
            { corners: [front, bottom, right] },
            { corners: [left, bottom, front] },
            { corners: [back, bottom, left] },
            { corners: [right, bottom, back] }
        ];

        // Standard UV mapping for a triangle
        // Top faces point "up", bottom faces point "down" on the texture
        const topUVs = [[1, 0], [0.5, 1], [0, 0]];
        const bottomUVs = [[0, 1], [0.5, 0], [1, 1]];

        // 4. Allocate the exact amount of memory needed
        const numVertices = 24; // 8 faces * 3 vertices (vertices duplicated for flat shading)
        const vertices = new Float32Array(numVertices * layout.strideFloats);
        let vIdx = 0;

        // 5. Build the Vertex Array dynamically!
        for (let f = 0; f < faces.length; f++) {
            const face = faces[f];
            const normal = calcNormal(face.corners[0], face.corners[1], face.corners[2]);

            const isTopHemisphere = f < 4;
            const uvs = isTopHemisphere ? topUVs : bottomUVs;

            for (let v = 0; v < 3; v++) {
                const pos = face.corners[v];
                const uv = uvs[v];

                // THE MAGIC: We only write the data the Layout specifically asks for!
                for (const attr of layout.attributes) {
                    switch (attr) {
                        case VertexAttribute.Position:
                            vertices[vIdx++] = pos[0];
                            vertices[vIdx++] = pos[1];
                            vertices[vIdx++] = pos[2];
                            break;
                        case VertexAttribute.Normal:
                            vertices[vIdx++] = normal[0];
                            vertices[vIdx++] = normal[1];
                            vertices[vIdx++] = normal[2];
                            break;
                        case VertexAttribute.UV:
                            vertices[vIdx++] = uv[0];
                            vertices[vIdx++] = uv[1];
                            break;
                        case VertexAttribute.Color:
                            // Default to white if a color attribute is requested
                            vertices[vIdx++] = 1.0; vertices[vIdx++] = 1.0;
                            vertices[vIdx++] = 1.0; vertices[vIdx++] = 1.0;
                            break;
                    }
                }
            }
        }

        // 6. Generate the Indices (Connecting the 3 vertices into 1 triangle per face)
        const indices = new Uint16Array(24); // 8 faces * 3 indices
        let iIdx = 0;

        for (let i = 0; i < 8; i++) {
            const base = i * 3;
            indices[iIdx++] = base + 0;
            indices[iIdx++] = base + 1;
            indices[iIdx++] = base + 2;
        }

        return new Geometry(layout, vertices, indices);
    }

    public static createPlane(layout: VertexLayout, width: number, height: number): Geometry {
        // 1. Calculate half-extents for centering the plane at (0,0,0)
        const w = width / 2;
        const h = height / 2;

        // 2. Define the 4 corners of the plane (facing +Z, lying on the XY plane)
        // Order: Bottom-Left, Bottom-Right, Top-Right, Top-Left
        // Note: If you want a ground plane instead, swap the Y and Z coordinates
        // (e.g., [-w, 0, -h]) and set the normal to [0, 1, 0].
        const corners = [
            [-w, -h, 0],
            [ w, -h, 0],
            [ w,  h, 0],
            [-w,  h, 0]
        ];

        // The normal for a standard XY plane points straight out toward the camera (+Z)
        const normal = [0, 0, 1];

        // Standard UV mapping for a single quad
        const uvs = [[0, 1], [1, 1], [1, 0], [0, 0]];

        // 3. Allocate memory for exactly 4 vertices
        const numVertices = 4;
        const vertices = new Float32Array(numVertices * layout.strideFloats);
        let vIdx = 0;

        // 4. Build the Vertex Array dynamically based on the layout
        for (let v = 0; v < 4; v++) {
            const pos = corners[v];
            const uv = uvs[v];

            // THE MAGIC: We only write the data the Layout specifically asks for
            for (const attr of layout.attributes) {
                switch (attr) {
                    case VertexAttribute.Position:
                        vertices[vIdx++] = pos[0];
                        vertices[vIdx++] = pos[1];
                        vertices[vIdx++] = pos[2];
                        break;
                    case VertexAttribute.Normal:
                        vertices[vIdx++] = normal[0];
                        vertices[vIdx++] = normal[1];
                        vertices[vIdx++] = normal[2];
                        break;
                    case VertexAttribute.UV:
                        vertices[vIdx++] = uv[0];
                        vertices[vIdx++] = uv[1];
                        break;
                    case VertexAttribute.Color:
                        // Default to white if a color attribute is requested
                        vertices[vIdx++] = 1.0; vertices[vIdx++] = 1.0;
                        vertices[vIdx++] = 1.0; vertices[vIdx++] = 1.0;
                        break;
                }
            }
        }

        // 5. Generate the Indices (Connecting the 4 vertices into 2 triangles using CCW winding)
        const indices = new Uint16Array(6);
        let iIdx = 0;

        // Triangle 1 (Bottom-Left, Bottom-Right, Top-Right)
        indices[iIdx++] = 0;
        indices[iIdx++] = 1;
        indices[iIdx++] = 2;

        // Triangle 2 (Top-Right, Top-Left, Bottom-Left)
        indices[iIdx++] = 2;
        indices[iIdx++] = 3;
        indices[iIdx++] = 0;

        return new Geometry(layout, vertices, indices);
    }
    public static createPyramid(layout: VertexLayout, width: number, height: number, depth: number): Geometry {
        // 1. Calculate half-extents for centering the pyramid at (0,0,0)
        const w = width / 2;
        const h = height / 2;
        const d = depth / 2;

        // 2. Define the 5 structural vertices of the pyramid
        const apex = [ 0,  h,  0];
        const fl   = [-w, -h,  d]; // Front-Left
        const fr   = [ w, -h,  d]; // Front-Right
        const br   = [ w, -h, -d]; // Back-Right
        const bl   = [-w, -h, -d]; // Back-Left

        // Helper to calculate the normal of a face (using Cross Product)
        const calcNormal = (v0: number[], v1: number[], v2: number[]) => {
            const dx1 = v1[0] - v0[0], dy1 = v1[1] - v0[1], dz1 = v1[2] - v0[2];
            const dx2 = v2[0] - v0[0], dy2 = v2[1] - v0[1], dz2 = v2[2] - v0[2];

            const nx = dy1 * dz2 - dz1 * dy2;
            const ny = dz1 * dx2 - dx1 * dz2;
            const nz = dx1 * dy2 - dy1 * dx2;

            const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
            return [nx / len, ny / len, nz / len];
        };

        // 3. Define the 5 faces using Counter-Clockwise (CCW) winding
        const faces = [
            // 4 Side Triangles
            { corners: [fl, fr, apex] }, // Front
            { corners: [fr, br, apex] }, // Right
            { corners: [br, bl, apex] }, // Back
            { corners: [bl, fl, apex] }, // Left
            // 1 Bottom Quad (Facing -Y)
            { corners: [bl, br, fr, fl] }
        ];

        // Standard UV mapping
        const triUVs = [[0, 1], [1, 1], [0.5, 0]]; // Triangle UVs (Bottom-Left, Bottom-Right, Top-Center)
        const quadUVs = [[0, 1], [1, 1], [1, 0], [0, 0]]; // Quad UVs

        // 4. Allocate the exact amount of memory needed
        // 4 triangles * 3 vertices + 1 quad * 4 vertices = 16 total vertices
        const numVertices = 16;
        const vertices = new Float32Array(numVertices * layout.strideFloats);
        let vIdx = 0;

        // 5. Build the Vertex Array dynamically!
        for (let f = 0; f < faces.length; f++) {
            const face = faces[f];
            const isQuad = face.corners.length === 4;
            const uvs = isQuad ? quadUVs : triUVs;

            // Calculate the normal once per flat face
            const normal = calcNormal(face.corners[0], face.corners[1], face.corners[2]);

            for (let v = 0; v < face.corners.length; v++) {
                const pos = face.corners[v];
                const uv = uvs[v];

                // THE MAGIC: Write only what the Layout requests
                for (const attr of layout.attributes) {
                    switch (attr) {
                        case VertexAttribute.Position:
                            vertices[vIdx++] = pos[0];
                            vertices[vIdx++] = pos[1];
                            vertices[vIdx++] = pos[2];
                            break;
                        case VertexAttribute.Normal:
                            vertices[vIdx++] = normal[0];
                            vertices[vIdx++] = normal[1];
                            vertices[vIdx++] = normal[2];
                            break;
                        case VertexAttribute.UV:
                            vertices[vIdx++] = uv[0];
                            vertices[vIdx++] = uv[1];
                            break;
                        case VertexAttribute.Color:
                            vertices[vIdx++] = 1.0; vertices[vIdx++] = 1.0;
                            vertices[vIdx++] = 1.0; vertices[vIdx++] = 1.0;
                            break;
                    }
                }
            }
        }

        // 6. Generate the Indices
        // 4 triangles * 3 indices + 1 quad * 6 indices = 18 total indices
        const indices = new Uint16Array(18);
        let iIdx = 0;
        let baseVertex = 0;

        for (let f = 0; f < faces.length; f++) {
            const isQuad = faces[f].corners.length === 4;

            if (isQuad) {
                // Triangle 1
                indices[iIdx++] = baseVertex + 0;
                indices[iIdx++] = baseVertex + 1;
                indices[iIdx++] = baseVertex + 2;
                // Triangle 2
                indices[iIdx++] = baseVertex + 2;
                indices[iIdx++] = baseVertex + 3;
                indices[iIdx++] = baseVertex + 0;
            } else {
                // Single Triangle
                indices[iIdx++] = baseVertex + 0;
                indices[iIdx++] = baseVertex + 1;
                indices[iIdx++] = baseVertex + 2;
            }

            baseVertex += faces[f].corners.length;
        }

        return new Geometry(layout, vertices, indices);
    }

    // // Generates a sphere with dynamic resolution
    // public static createSphere(layout: VertexLayout, radius: number, widthSegments: number, heightSegments: number): Geometry {
    //     // Trigonometry logic to generate rings of vertices
    // }
}