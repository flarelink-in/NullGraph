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
bakes them into flat typed arrays — useful for voxel chunks, marching cubes, extruded paths, or any
mesh you generate at runtime. You add vertices (getting back an index) and stitch them into triangles.

### API

```ts
const builder = new GeometryBuilder(StandardLayout);

// addVertex(data) appends one vertex and returns its index
const i0 = builder.addVertex({ position: [-1, -1, 0], normal: [0, 0, 1], uv: [0, 0] });
const i1 = builder.addVertex({ position: [ 1, -1, 0], normal: [0, 0, 1], uv: [1, 0] });
const i2 = builder.addVertex({ position: [ 1,  1, 0], normal: [0, 0, 1], uv: [1, 1] });
const i3 = builder.addVertex({ position: [-1,  1, 0], normal: [0, 0, 1], uv: [0, 1] });

// addTriangle(a, b, c) references vertices by the indices returned above
builder.addTriangle(i0, i1, i2);
builder.addTriangle(i0, i2, i3);

// build() bakes the accumulated data into typed arrays
const baked = builder.build();
// baked = { v: Float32Array, i: Uint16Array, vertexBuffer, indexBuffer, layout }
// (`v`/`vertexBuffer` and `i`/`indexBuffer` are aliases of the same data)
```

`VertexData` accepted by `addVertex`:

```ts
interface VertexData {
    position: [number, number, number];  // required
    normal?:  [number, number, number];  // defaults to [0, 1, 0] if omitted
    uv?:      [number, number];
    color?:   [number, number, number, number];
}
```

### Uploading a built mesh

`build()` returns **plain typed arrays**, not a `Geometry`. Wrap them in a `Geometry` to reuse its
`upload()`/`destroy()` lifecycle:

```ts
const baked = builder.build();
const geom = new Geometry(baked.layout, baked.v, baked.i);
geom.upload(engine);

engine.setBatchGeometry(batch, geom.vertexBuffer!, geom.indexBuffer!, geom.indices.length);
```

Or create the GPU buffers directly if you don't need the `Geometry` wrapper:

```ts
const vertexBuffer = engine.bufferManager.createVertexBuffer(baked.v);
const indexBuffer  = engine.bufferManager.createIndexBuffer(baked.i);
engine.setBatchGeometry(batch, vertexBuffer, indexBuffer, baked.i.length);
```

### Things to know

- **The builder emits `Position (3) + Normal (3) + UV (2)` — i.e. the `StandardLayout` (8 floats).**
  Always pass `uv` when targeting `StandardLayout`; omitting it writes only 6 floats for that vertex
  and desyncs the whole vertex stride.
- **`color` is accepted by `VertexData` but not written** by `addVertex`, so the builder does not
  currently produce `CompleteLayout` geometry. Use `Primitives`/manual `Float32Array`s for colored
  vertices, or push into `builder.vertices` yourself.
- **Indices are `Uint16Array`**, capping a single builder at 65,536 vertices. Split large meshes into
  multiple builders/batches (or write `Uint32Array` indices manually).
- `builder.vertices` and `builder.indices` are plain `number[]` you can also read or mutate directly
  before calling `build()`.

> For very large, streaming worlds see `DynamicGeometryManager` and `MegaBufferBuilder` (also
> exported from this module), which pack many meshes into shared mega-buffers to minimize buffer
> churn and state changes.
