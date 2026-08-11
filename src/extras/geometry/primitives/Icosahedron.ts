import { VertexAttribute, VertexLayout } from "../VertexLayout";
import { Geometry } from "../Geometry";

export function createIcosahedron(layout: VertexLayout, radius: number): Geometry {
    // Golden ratio
    const t = (1 + Math.sqrt(5)) / 2;

    // Normalize helper
    const normalize = (v: number[]) => {
        const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        return [v[0]/len * radius, v[1]/len * radius, v[2]/len * radius];
    };

    // 12 vertices of an icosahedron
    const rawVerts = [
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ].map(normalize);

    // 20 triangular faces (indices into rawVerts)
    const faces = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]
    ];

    // Normal calculation
    const calcNormal = (v0: number[], v1: number[], v2: number[]) => {
        const dx1 = v1[0] - v0[0], dy1 = v1[1] - v0[1], dz1 = v1[2] - v0[2];
        const dx2 = v2[0] - v0[0], dy2 = v2[1] - v0[1], dz2 = v2[2] - v0[2];

        const nx = dy1 * dz2 - dz1 * dy2;
        const ny = dz1 * dx2 - dx1 * dz2;
        const nz = dx1 * dy2 - dy1 * dx2;

        const len = Math.sqrt(nx*nx + ny*ny + nz*nz);
        return [nx/len, ny/len, nz/len];
    };

    // 20 faces × 3 vertices
    const numVertices = 60;
    const vertices = new Float32Array(numVertices * layout.strideFloats);
    let vIdx = 0;

    // Simple triangle UVs
    const uvs = [[0, 0], [1, 0], [0.5, 1]];

    for (let f = 0; f < faces.length; f++) {
        const [i0, i1, i2] = faces[f];

        const v0 = rawVerts[i0];
        const v1 = rawVerts[i1];
        const v2 = rawVerts[i2];

        const normal = calcNormal(v0, v1, v2);

        const tri = [v0, v1, v2];

        for (let v = 0; v < 3; v++) {
            const pos = tri[v];
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
                        vertices[vIdx++] = 1.0;
                        vertices[vIdx++] = 1.0;
                        vertices[vIdx++] = 1.0;
                        vertices[vIdx++] = 1.0;
                        break;
                }
            }
        }
    }

    // Indices (no sharing → flat shading)
    const indices = new Uint16Array(60);
    for (let i = 0; i < 60; i++) {
        indices[i] = i;
    }

    return new Geometry(layout, vertices, indices);
}
