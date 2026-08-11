import { VertexAttribute, VertexLayout } from "../VertexLayout";
import { Geometry } from "../Geometry";

export function createSphere(
    layout: VertexLayout,
    radius: number,
    widthSegments: number,
    heightSegments: number
): Geometry {

    widthSegments = Math.max(3, Math.floor(widthSegments));
    heightSegments = Math.max(2, Math.floor(heightSegments));

    const vertices: number[] = [];
    const indices: number[] = [];

    const grid: number[][] = [];

    let index = 0;

    // Generate vertices
    for (let y = 0; y <= heightSegments; y++) {
        const v = y / heightSegments;
        const theta = v * Math.PI;

        const row: number[] = [];

        for (let x = 0; x <= widthSegments; x++) {
            const u = x / widthSegments;
            const phi = u * Math.PI * 2;

            const sinTheta = Math.sin(theta);
            const cosTheta = Math.cos(theta);
            const sinPhi = Math.sin(phi);
            const cosPhi = Math.cos(phi);

            const px = radius * sinTheta * cosPhi;
            const py = radius * cosTheta;
            const pz = radius * sinTheta * sinPhi;

            const nx = sinTheta * cosPhi;
            const ny = cosTheta;
            const nz = sinTheta * sinPhi;

            // Write attributes based on layout
            for (const attr of layout.attributes) {
                switch (attr) {
                    case VertexAttribute.Position:
                        vertices.push(px, py, pz);
                        break;
                    case VertexAttribute.Normal:
                        vertices.push(nx, ny, nz);
                        break;
                    case VertexAttribute.UV:
                        vertices.push(u, 1 - v);
                        break;
                    case VertexAttribute.Color:
                        vertices.push(1, 1, 1, 1);
                        break;
                }
            }

            row.push(index++);
        }

        grid.push(row);
    }

    // Generate indices
    for (let y = 0; y < heightSegments; y++) {
        for (let x = 0; x < widthSegments; x++) {

            const a = grid[y][x + 1];
            const b = grid[y][x];
            const c = grid[y + 1][x];
            const d = grid[y + 1][x + 1];

            // Two triangles per quad, wound front-face-out (CCW from outside) so the
            // winding agrees with the outward normals and matches the other primitives.
            if (y !== 0) {
                indices.push(a, d, b);
            }

            if (y !== heightSegments - 1) {
                indices.push(b, d, c);
            }
        }
    }

    return new Geometry(
        layout,
        new Float32Array(vertices),
        new Uint16Array(indices)
    );
}
