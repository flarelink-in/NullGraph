import { VertexAttribute, VertexLayout } from "../VertexLayout";
import { Geometry } from "../Geometry";

export function createCube(layout: VertexLayout, width: number, height: number, depth: number): Geometry {
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
