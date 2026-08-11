import { VertexAttribute, VertexLayout } from "../VertexLayout";
import { Geometry } from "../Geometry";

export function createRing(
    layout: VertexLayout,
    innerRadius: number,
    outerRadius: number,
    thetaSegments: number
): Geometry {
    thetaSegments = Math.max(3, Math.floor(thetaSegments));

    const vertices: number[] = [];
    const indices: number[] = [];

    // Emit one full loop of (inner, outer) vertex pairs facing +Z or -Z. A flat ring is
    // one-sided by nature, so we build BOTH sides — otherwise it vanishes (looks "half the
    // faces") whenever it is viewed from behind or backface culling is on.
    const emitSide = (nz: number) => {
        for (let i = 0; i <= thetaSegments; i++) {
            const u = i / thetaSegments;
            const theta = u * Math.PI * 2;
            const cosTheta = Math.cos(theta);
            const sinTheta = Math.sin(theta);

            const pushVertex = (radius: number, uvx: number, uvy: number) => {
                for (const attr of layout.attributes) {
                    switch (attr) {
                        case VertexAttribute.Position:
                            vertices.push(radius * cosTheta, radius * sinTheta, 0);
                            break;
                        case VertexAttribute.Normal:
                            vertices.push(0, 0, nz);
                            break;
                        case VertexAttribute.UV:
                            vertices.push(uvx, uvy);
                            break;
                        case VertexAttribute.Color:
                            vertices.push(1, 1, 1, 1);
                            break;
                    }
                }
            };

            // Inner vertex, then outer vertex.
            pushVertex(innerRadius,
                (innerRadius * cosTheta / outerRadius + 1) / 2,
                (innerRadius * sinTheta / outerRadius + 1) / 2);
            pushVertex(outerRadius, (cosTheta + 1) / 2, (sinTheta + 1) / 2);
        }
    };

    const vertsPerSide = (thetaSegments + 1) * 2;

    // Front side: normal +Z, wound CCW so the front face points toward +Z.
    emitSide(1);
    for (let i = 0; i < thetaSegments; i++) {
        const i2 = i * 2;
        indices.push(i2, i2 + 1, i2 + 3);
        indices.push(i2, i2 + 3, i2 + 2);
    }

    // Back side: normal -Z, winding reversed so its front face points toward -Z.
    emitSide(-1);
    for (let i = 0; i < thetaSegments; i++) {
        const i2 = vertsPerSide + i * 2;
        indices.push(i2, i2 + 3, i2 + 1);
        indices.push(i2, i2 + 2, i2 + 3);
    }

    return new Geometry(
        layout,
        new Float32Array(vertices),
        new Uint32Array(indices)
    );
}
