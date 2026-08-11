import { createCube } from "./primitives/Cube";
import { createOctahedron } from "./primitives/Octahedron";
import { createPlane } from "./primitives/Plane";
import { createPyramid } from "./primitives/Pyramid";
import { createIcosahedron } from "./primitives/Icosahedron";
import { createSphere } from "./primitives/Sphere";
import { createTorus } from "./primitives/Torus";
import { createCylinder } from "./primitives/Cylinder";
import { createCapsule } from "./primitives/Capsule";
import { createCone } from "./primitives/Cone";
import { createRing } from "./primitives/Ring";
import { createTetrahedron } from "./primitives/Tetrahedron";
import { createDodecahedron } from "./primitives/Dodecahedron";

export class Primitives {
    public static createCube = createCube;
    public static createOctahedron = createOctahedron;
    public static createPlane = createPlane;
    public static createPyramid = createPyramid;
    public static createIcosahedron = createIcosahedron;
    public static createSphere = createSphere;
    public static createTorus = createTorus;
    
    // New shapes
    public static createCylinder = createCylinder;
    public static createCapsule = createCapsule;
    public static createCone = createCone;
    public static createRing = createRing;
    public static createTetrahedron = createTetrahedron;
    public static createDodecahedron = createDodecahedron;
}