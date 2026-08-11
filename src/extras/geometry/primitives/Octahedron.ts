import { VertexAttribute, VertexLayout } from "../VertexLayout";
import { Geometry } from "../Geometry";

export function createOctahedron(layout: VertexLayout, width: number, height: number, depth: number): Geometry {
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
