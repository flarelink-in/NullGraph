import { VertexAttribute, VertexLayout } from "../VertexLayout";
import { Geometry } from "../Geometry";

export function createPyramid(layout: VertexLayout, width: number, height: number, depth: number): Geometry {
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
