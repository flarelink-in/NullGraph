import { VertexAttribute, VertexLayout } from "../VertexLayout";
import { Geometry } from "../Geometry";

export function createTetrahedron(layout: VertexLayout, radius: number): Geometry {
    const sqrt3 = Math.sqrt(3);
    const r = radius / Math.sqrt(3);

    // 4 vertices of a tetrahedron
    const v0 = [ r,  r,  r];
    const v1 = [-r, -r,  r];
    const v2 = [-r,  r, -r];
    const v3 = [ r, -r, -r];

    // 4 triangular faces
    const faces = [
        { corners: [v2, v1, v0] },
        { corners: [v0, v3, v2] },
        { corners: [v1, v3, v0] },
        { corners: [v2, v3, v1] }
    ];

    const calcNormal = (p0: number[], p1: number[], p2: number[]) => {
        const dx1 = p1[0] - p0[0], dy1 = p1[1] - p0[1], dz1 = p1[2] - p0[2];
        const dx2 = p2[0] - p0[0], dy2 = p2[1] - p0[1], dz2 = p2[2] - p0[2];

        const nx = dy1 * dz2 - dz1 * dy2;
        const ny = dz1 * dx2 - dx1 * dz2;
        const nz = dx1 * dy2 - dy1 * dx2;

        const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
        return [nx / len, ny / len, nz / len];
    };

    const numVertices = 12; // 4 faces * 3 vertices
    const vertices = new Float32Array(numVertices * layout.strideFloats);
    let vIdx = 0;

    const uvs = [[0.5, 1], [0, 0], [1, 0]];

    for (let f = 0; f < faces.length; f++) {
        const face = faces[f];
        const normal = calcNormal(face.corners[0], face.corners[1], face.corners[2]);

        for (let v = 0; v < 3; v++) {
            const pos = face.corners[v];
            const uv = uvs[v];

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

    const indices = new Uint16Array(12);
    for (let i = 0; i < 12; i++) {
        indices[i] = i;
    }

    return new Geometry(layout, vertices, indices);
}
