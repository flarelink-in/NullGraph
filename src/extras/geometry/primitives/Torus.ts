import { VertexAttribute, VertexLayout } from "../VertexLayout";
import { Geometry } from "../Geometry";

export function createTorus(
    layout: VertexLayout,
    radius: number,
    tubeRadius: number,
    radialSegments: number,
    tubularSegments: number
): Geometry {
    // Enforce minimum segment counts to form a complete shape
    radialSegments = Math.max(3, Math.floor(radialSegments));
    tubularSegments = Math.max(3, Math.floor(tubularSegments));

    const vertices: number[] = [];
    const indices: number[] = [];
    const grid: number[][] = [];

    let index = 0;

    // 1. Generate vertices
    // j loops around the cross-section of the tube (radial)
    for (let j = 0; j <= radialSegments; j++) {
        const v = j / radialSegments;
        const phi = v * Math.PI * 2; // Angle around the tube

        const cosPhi = Math.cos(phi);
        const sinPhi = Math.sin(phi);

        const row: number[] = [];

        // i loops around the main ring of the torus (tubular)
        for (let i = 0; i <= tubularSegments; i++) {
            const u = i / tubularSegments;
            const theta = u * Math.PI * 2; // Angle around the center hole

            const cosTheta = Math.cos(theta);
            const sinTheta = Math.sin(theta);

            // Calculate Position
            // (radius + tubeRadius * cosPhi) dictates the distance from the Y-axis center
            const px = (radius + tubeRadius * cosPhi) * cosTheta;
            const py = tubeRadius * sinPhi;
            const pz = (radius + tubeRadius * cosPhi) * sinTheta;

            // Calculate Normal
            // The normal is the normalized vector pointing from the tube's center to the surface
            const nx = cosPhi * cosTheta;
            const ny = sinPhi;
            const nz = cosPhi * sinTheta;

            // THE MAGIC: Write attributes based on layout dynamically
            for (const attr of layout.attributes) {
                switch (attr) {
                    case VertexAttribute.Position:
                        vertices.push(px, py, pz);
                        break;
                    case VertexAttribute.Normal:
                        vertices.push(nx, ny, nz);
                        break;
                    case VertexAttribute.UV:
                        vertices.push(u, 1 - v); // 1-v matches typical Y-down UV mapping
                        break;
                    case VertexAttribute.Color:
                        // Default to white if a color attribute is requested
                        vertices.push(1.0, 1.0, 1.0, 1.0);
                        break;
                }
            }

            row.push(index++);
        }

        grid.push(row);
    }

    // 2. Generate indices
    for (let j = 0; j < radialSegments; j++) {
        for (let i = 0; i < tubularSegments; i++) {
            // Grab the 4 corners of the current quad from the grid
            const a = grid[j][i + 1];
            const b = grid[j][i];
            const c = grid[j + 1][i];
            const d = grid[j + 1][i + 1];

            // Connect the 4 vertices into 2 triangles using Counter-Clockwise (CCW) winding
            indices.push(a, b, d);
            indices.push(b, c, d);
        }
    }

    return new Geometry(
        layout,
        new Float32Array(vertices),
        new Uint32Array(indices) // Uses Uint32Array, which is a bit different from others but fine
    );
}
