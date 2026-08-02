# Geometry API Usage

The `null-graph/geometry` module is the math and primitive toolbox. It generates vertex/index data
for common shapes, lets you define custom vertex layouts on the fly, and packs everything into
WebGPU-ready buffers. Because NullGraph has no scene graph, geometry is just raw typed-array data
plus a layout description — no nodes, no hierarchy.

```ts
import {
    Primitives, VertexLayout, StandardLayout, CompleteLayout, PositionOnlyLayout,
    Geometry, GeometryBuilder,
} from 'null-graph/geometry';
```

---

## Vertex layouts

A `VertexLayout` describes the interleaved attributes in a vertex buffer and computes the stride
for you. Three presets cover most cases:

| Preset | Attributes | Stride (floats) |
|---|---|---|
| `PositionOnlyLayout` | Position | 3 |
| `StandardLayout` | Position, Normal, UV | 8 |
| `CompleteLayout` | Position, Normal, UV, Color | 11 |

```ts
// Custom layout
const layout = new VertexLayout([
    VertexAttribute.Position,
    VertexAttribute.Normal,
    VertexAttribute.UV,
]);

layout.strideFloats;                // number of floats per vertex
layout.getWebGPUDescriptor();       // GPUVertexBufferLayout[] for createBatch({ vertexLayouts })
```

## Primitives

`Primitives` is a static factory that returns a `Geometry` for each shape. Every method takes a
layout so the output matches the vertex format your shader expects.

```ts
const cube        = Primitives.createCube(StandardLayout, 2, 2, 2);  // 2x2x2 cube
const plane       = Primitives.createPlane(StandardLayout, 10, 10); // 10x10 plane
const sphere      = Primitives.createSphere(StandardLayout, 30,10,10); // radius,widthSegments,heightSegments
const torus       = Primitives.createTorus(StandardLayout, 20,10,20,30); // radius,tubeRadius,radialSegments,tubularSegments
const pyramid     = Primitives.createPyramid(StandardLayout, 2, 2, 2); //width,height,depth
const octahedron  = Primitives.createOctahedron(StandardLayout, 2, 2, 2); //width,height,depth
const icosahedron = Primitives.createIcosahedron(StandardLayout, 1); //radius
```

Available factories: `createCube`, `createPlane`, `createPyramid`, `createOctahedron`,
`createIcosahedron`, `createSphere`, `createTorus`.

## `Geometry`

A `Geometry` holds the CPU-side `vertices` (`Float32Array`) and `indices` (`Uint16Array` or
`Uint32Array`) plus the layout, and uploads them to GPU buffers.

```ts
const geom = Primitives.createCube(StandardLayout, 2, 2, 2);
geom.upload(engine); // creates geom.vertexBuffer and geom.indexBuffer

engine.setBatchGeometry(
    batch,
    geom.vertexBuffer!,
    geom.indexBuffer!,
    geom.indices.length,
);

// Free VRAM when done
geom.destroy();
```

## `GeometryBuilder`

For procedural meshes, `GeometryBuilder` accumulates vertices and indices into growable arrays and
bakes a `Geometry` respecting a given layout — useful for voxel chunks, marching cubes, or any
mesh you generate at runtime.

```ts
const builder = new GeometryBuilder(StandardLayout);
// push positions/normals/uvs into builder.vertices and triangles into builder.indices
// then bake into a Geometry and upload it to the engine
```

> For very large, streaming worlds see `DynamicGeometryManager` and `MegaBufferBuilder` (also
> exported from this module), which pack many meshes into shared mega-buffers to minimize buffer
> churn and state changes.
