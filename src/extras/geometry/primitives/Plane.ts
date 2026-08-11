import { VertexAttribute, VertexLayout } from "../VertexLayout";
import { Geometry } from "../Geometry";

export function createPlane(layout: VertexLayout, width: number, height: number): Geometry {
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

    // 3. Build BOTH sides. A single quad is one-sided (+Z) and disappears when viewed from
    //    behind or under backface culling — build a back face (-Z) too so it is always visible.
    const vertexData: number[] = [];

    const emitSide = (nz: number) => {
        for (let v = 0; v < 4; v++) {
            const pos = corners[v];
            const uv = uvs[v];
            for (const attr of layout.attributes) {
                switch (attr) {
                    case VertexAttribute.Position:
                        vertexData.push(pos[0], pos[1], pos[2]);
                        break;
                    case VertexAttribute.Normal:
                        vertexData.push(0, 0, nz);
                        break;
                    case VertexAttribute.UV:
                        vertexData.push(uv[0], uv[1]);
                        break;
                    case VertexAttribute.Color:
                        vertexData.push(1.0, 1.0, 1.0, 1.0);
                        break;
                }
            }
        }
    };

    emitSide(1);   // front, normal +Z
    emitSide(-1);  // back,  normal -Z

    const vertices = new Float32Array(vertexData);

    // 4. Front CCW (faces +Z); back reversed (faces -Z), offset by the 4 front vertices.
    const indices = new Uint16Array([
        0, 1, 2,  2, 3, 0,   // front
        4, 6, 5,  6, 4, 7,   // back (reversed winding)
    ]);

    return new Geometry(layout, vertices, indices);
}
