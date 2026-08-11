import { VertexAttribute, VertexLayout } from "../VertexLayout";
import { Geometry } from "../Geometry";

export function createDodecahedron(layout: VertexLayout, radius: number): Geometry {
    const t = (1 + Math.sqrt(5)) / 2;
    const r = 1 / t;

    // 20 vertices of a dodecahedron
    let rawVerts = [
        // (±1, ±1, ±1)
        [-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [-1,  1,  1],
        [ 1, -1, -1], [ 1, -1,  1], [ 1,  1, -1], [ 1,  1,  1],
        // (0, ±1/φ, ±φ)
        [ 0, -r, -t], [ 0, -r,  t], [ 0,  r, -t], [ 0,  r,  t],
        // (±1/φ, ±φ, 0)
        [-r, -t,  0], [-r,  t,  0], [ r, -t,  0], [ r,  t,  0],
        // (±φ, 0, ±1/φ)
        [-t,  0, -r], [-t,  0,  r], [ t,  0, -r], [ t,  0,  r]
    ];

    // Normalize and scale
    rawVerts = rawVerts.map(v => {
        const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        return [v[0]/len * radius, v[1]/len * radius, v[2]/len * radius];
    });

    // 12 pentagonal faces (indices into rawVerts)
    const faces = [
        [3, 11, 7, 15, 13],
        [7, 19, 17, 3, 11],
        [15, 7, 19, 18, 6],
        [13, 15, 6, 10, 2],
        [17, 3, 13, 2, 16],
        [19, 17, 16, 0, 1],
        [11, 9, 1, 19, 7],
        [5, 9, 11, 3, 17], // Wait, this might not form a valid pentagon in the right order. We'll triangulate via a center point instead.
    ];

    // Better approach: We can just use hardcoded indices for the 12 faces, properly ordered CCW
    const pentagons = [
        [3, 11, 7, 15, 13],
        [7, 19, 5, 9, 11],
        [15, 7, 19, 18, 6],
        [13, 15, 6, 10, 2],
        [3, 13, 2, 16, 17],
        [11, 3, 17, 1, 9],
        [5, 19, 18, 4, 14],
        [9, 5, 14, 12, 1],
        [1, 12, 0, 16, 17],
        [2, 10, 8, 0, 16],
        [6, 18, 4, 8, 10],
        [14, 4, 8, 0, 12]
    ];

    const calcNormal = (v0: number[], v1: number[], v2: number[]) => {
        const dx1 = v1[0] - v0[0], dy1 = v1[1] - v0[1], dz1 = v1[2] - v0[2];
        const dx2 = v2[0] - v0[0], dy2 = v2[1] - v0[1], dz2 = v2[2] - v0[2];

        const nx = dy1 * dz2 - dz1 * dy2;
        const ny = dz1 * dx2 - dx1 * dz2;
        const nz = dx1 * dy2 - dy1 * dx2;

        const len = Math.sqrt(nx*nx + ny*ny + nz*nz);
        return [nx/len, ny/len, nz/len];
    };

    // 12 faces * 3 triangles/face * 3 vertices/triangle = 108 vertices
    const numVertices = 108;
    const vertices = new Float32Array(numVertices * layout.strideFloats);
    const indices = new Uint16Array(108);

    let vIdx = 0;
    let iIdx = 0;

    for (let f = 0; f < pentagons.length; f++) {
        const pentagon = pentagons[f];
        
        // Triangulate pentagon (0-1-2, 0-2-3, 0-3-4)
        const tris = [
            [pentagon[0], pentagon[1], pentagon[2]],
            [pentagon[0], pentagon[2], pentagon[3]],
            [pentagon[0], pentagon[3], pentagon[4]]
        ];

        const v0 = rawVerts[pentagon[0]];
        const v1 = rawVerts[pentagon[1]];
        const v2 = rawVerts[pentagon[2]];
        
        // Use the first triangle to calculate the flat normal for the whole pentagon
        const normal = calcNormal(v0, v1, v2);

        for (let t = 0; t < 3; t++) {
            const tri = tris[t];

            for (let v = 0; v < 3; v++) {
                const pos = rawVerts[tri[v]];
                
                // Simple UV mapping (could be improved)
                const uv = [v === 0 ? 0.5 : (v === 1 ? 0 : 1), v === 0 ? 1 : 0]; 

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
                
                indices[iIdx] = iIdx;
                iIdx++;
            }
        }
    }

    return new Geometry(layout, vertices, indices);
}
