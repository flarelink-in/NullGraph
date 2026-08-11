import { VertexAttribute, VertexLayout } from "../VertexLayout";
import { Geometry } from "../Geometry";
import { createCylinder } from "./Cylinder";

export function createCone(
    layout: VertexLayout,
    radius: number,
    height: number,
    radialSegments: number,
    heightSegments: number
): Geometry {
    // A cone is just a cylinder with a top radius of 0
    return createCylinder(layout, 0, radius, height, radialSegments, heightSegments);
}
