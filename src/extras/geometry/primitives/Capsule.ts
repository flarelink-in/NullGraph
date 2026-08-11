import { VertexAttribute, VertexLayout } from "../VertexLayout";
import { Geometry } from "../Geometry";

export function createCapsule(
    layout: VertexLayout,
    radius: number,
    length: number,
    capSegments: number,
    radialSegments: number
): Geometry {
    capSegments = Math.max(1, Math.floor(capSegments));
    radialSegments = Math.max(3, Math.floor(radialSegments));

    const vertices: number[] = [];
    const indices: number[] = [];

    const halfLength = length / 2;
    let index = 0;
    const indexArray: number[][] = [];

    // We generate (capSegments) rings for the top hemisphere, 1 ring for the middle (or reuse equator), and (capSegments) rings for the bottom hemisphere.
    // To cleanly have a cylinder body, we can just split the sphere generation.
    // Total rings = capSegments (top) + 1 (cylinder body) + capSegments (bottom) + 1 = 2 * capSegments + 2 rings
    const rings = 2 * capSegments + 1;

    for (let y = 0; y <= rings; y++) {
        let indexRow: number[] = [];
        
        let v: number;
        let theta: number;
        let yOffset = 0;

        if (y <= capSegments) {
            // Top hemisphere
            v = y / (2 * capSegments);
            theta = v * Math.PI;
            yOffset = halfLength;
        } else {
            // Bottom hemisphere
            v = (y - 1) / (2 * capSegments);
            theta = v * Math.PI;
            yOffset = -halfLength;
        }

        const cosTheta = Math.cos(theta);
        const sinTheta = Math.sin(theta);

        for (let x = 0; x <= radialSegments; x++) {
            const u = x / radialSegments;
            const phi = u * Math.PI * 2;

            const cosPhi = Math.cos(phi);
            const sinPhi = Math.sin(phi);

            // Position
            const px = radius * sinTheta * cosPhi;
            const py = radius * cosTheta + yOffset;
            const pz = radius * sinTheta * sinPhi;

            // Normal
            const nx = sinTheta * cosPhi;
            const ny = cosTheta;
            const nz = sinTheta * sinPhi;

            // UV mapping - stretching the body
            const uvY = (y <= capSegments) 
                ? (1 - v) * (radius / (length + 2 * radius)) // Map top cap
                : ((y === capSegments + 1) ? 1 - v : 1 - v) * (radius / (length + 2 * radius)) + (length / (length + 2 * radius)); // This UV math is simplified for the example

            for (const attr of layout.attributes) {
                switch (attr) {
                    case VertexAttribute.Position:
                        vertices.push(px, py, pz);
                        break;
                    case VertexAttribute.Normal:
                        vertices.push(nx, ny, nz);
                        break;
                    case VertexAttribute.UV:
                        vertices.push(u, 1 - (y / rings)); // Simple proportional UV
                        break;
                    case VertexAttribute.Color:
                        vertices.push(1, 1, 1, 1);
                        break;
                }
            }

            indexRow.push(index++);
        }
        indexArray.push(indexRow);
    }

    for (let y = 0; y < rings; y++) {
        for (let x = 0; x < radialSegments; x++) {
            const a = indexArray[y][x];
            const b = indexArray[y + 1][x];
            const c = indexArray[y + 1][x + 1];
            const d = indexArray[y][x + 1];

            // Wound front-face-out (CCW when viewed from outside), consistent with
            // every other primitive so backface culling keeps the visible surface.
            if (y !== 0) {
                indices.push(a, d, b);
            }
            if (y !== rings - 1) {
                indices.push(b, d, c);
            }
        }
    }

    return new Geometry(
        layout,
        new Float32Array(vertices),
        new Uint32Array(indices)
    );
}
