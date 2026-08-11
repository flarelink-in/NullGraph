import { VertexAttribute, VertexLayout } from "../VertexLayout";
import { Geometry } from "../Geometry";

export function createCylinder(
    layout: VertexLayout,
    radiusTop: number,
    radiusBottom: number,
    height: number,
    radialSegments: number,
    heightSegments: number
): Geometry {
    radialSegments = Math.max(3, Math.floor(radialSegments));
    heightSegments = Math.max(1, Math.floor(heightSegments));

    const vertices: number[] = [];
    const indices: number[] = [];
    const indexArray: number[][] = [];
    const halfHeight = height / 2;

    let index = 0;

    // generate side
    for (let y = 0; y <= heightSegments; y++) {
        const indexRow: number[] = [];
        const v = y / heightSegments;
        const radius = v * (radiusBottom - radiusTop) + radiusTop;

        for (let x = 0; x <= radialSegments; x++) {
            const u = x / radialSegments;
            const theta = u * Math.PI * 2;

            const sinTheta = Math.sin(theta);
            const cosTheta = Math.cos(theta);

            // vertex position
            const px = radius * sinTheta;
            const py = -v * height + halfHeight;
            const pz = radius * cosTheta;

            // normal
            // slope of cylinder side
            const slope = (radiusBottom - radiusTop) / height;
            
            // the normal vector
            let nx = sinTheta;
            let ny = slope;
            let nz = cosTheta;

            // normalize
            const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
            nx /= len;
            ny /= len;
            nz /= len;

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

            indexRow.push(index++);
        }
        indexArray.push(indexRow);
    }

    for (let x = 0; x < radialSegments; x++) {
        for (let y = 0; y < heightSegments; y++) {
            const a = indexArray[y][x];
            const b = indexArray[y + 1][x];
            const c = indexArray[y + 1][x + 1];
            const d = indexArray[y][x + 1];

            // Skip the degenerate triangle when a ring collapses to a point
            // (e.g. a cone has radiusTop === 0), mirroring the sphere's pole handling.
            if (!(radiusTop === 0 && y === 0)) indices.push(a, b, d);
            if (!(radiusBottom === 0 && y === heightSegments - 1)) indices.push(b, c, d);
        }
    }

    // Generate caps
    const generateCap = (top: boolean) => {
        const radius = top ? radiusTop : radiusBottom;
        const sign = top ? 1 : -1;
        const centerIndexStart = index;

        // center vertex
        for (const attr of layout.attributes) {
            switch (attr) {
                case VertexAttribute.Position:
                    vertices.push(0, halfHeight * sign, 0);
                    break;
                case VertexAttribute.Normal:
                    vertices.push(0, sign, 0);
                    break;
                case VertexAttribute.UV:
                    vertices.push(0.5, 0.5);
                    break;
                case VertexAttribute.Color:
                    vertices.push(1, 1, 1, 1);
                    break;
            }
        }
        index++;

        // Edge vertices
        for (let x = 0; x <= radialSegments; x++) {
            const u = x / radialSegments;
            const theta = u * Math.PI * 2;

            const cosTheta = Math.cos(theta);
            const sinTheta = Math.sin(theta);

            const px = radius * sinTheta;
            const py = halfHeight * sign;
            const pz = radius * cosTheta;

            for (const attr of layout.attributes) {
                switch (attr) {
                    case VertexAttribute.Position:
                        vertices.push(px, py, pz);
                        break;
                    case VertexAttribute.Normal:
                        vertices.push(0, sign, 0);
                        break;
                    case VertexAttribute.UV:
                        // Map unit circle to [0,1]x[0,1] bounding box
                        vertices.push((sinTheta + 1) / 2, (sign * cosTheta + 1) / 2);
                        break;
                    case VertexAttribute.Color:
                        vertices.push(1, 1, 1, 1);
                        break;
                }
            }
            index++;
        }

        for (let x = 0; x < radialSegments; x++) {
            const center = centerIndexStart;
            const i = centerIndexStart + 1 + x;

            if (top) {
                indices.push(i, i + 1, center);
            } else {
                indices.push(i + 1, i, center);
            }
        }
    };

    if (radiusTop > 0) generateCap(true);
    if (radiusBottom > 0) generateCap(false);

    return new Geometry(
        layout,
        new Float32Array(vertices),
        new Uint32Array(indices)
    );
}
