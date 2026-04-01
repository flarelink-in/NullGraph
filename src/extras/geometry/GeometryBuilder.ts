// null-graph/geometry/GeometryBuilder.ts

export interface VertexData {
    position: [number, number, number];
    normal?: [number, number, number];
    uv?: [number, number];
    color?: [number, number, number, number];
}

export class GeometryBuilder {
    public vertices: number[] = [];
    public indices: number[] = [];
    private layout: any;
    private vertexCount: number = 0;

    constructor(layout: any) {
        this.layout = layout;
    }

    // Adds a vertex and returns its index
    addVertex(data: VertexData): number {
        // Assuming StandardLayout order: Position (3), Normal (3), UV (2)
        // You can dynamically check 'this.layout' here if you support multiple formats!

        this.vertices.push(...data.position);

        if (data.normal) {
            this.vertices.push(...data.normal);
        } else {
            this.vertices.push(0, 1, 0); // Default normal
        }

        // If your shader layout expects UVs or Colors, push them here
        if (data.uv) {
            this.vertices.push(...data.uv);
        }

        const currentIndex = this.vertexCount;
        this.vertexCount++;
        return currentIndex;
    }

    // Convenience method for faces
    addTriangle(a: number, b: number, c: number) {
        this.indices.push(a, b, c);
    }

    build() {
        return {
            v: new Float32Array(this.vertices),
            i: new Uint16Array(this.indices),
            // Including these for standard engine compatibility if needed elsewhere
            vertexBuffer: new Float32Array(this.vertices),
            indexBuffer: new Uint16Array(this.indices),
            layout: this.layout
        };
    }
}