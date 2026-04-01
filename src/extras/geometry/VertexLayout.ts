// null-graph/geometry/VertexLayout.ts

export enum VertexAttribute {
    Position = 'position', // float32x3 (12 bytes)
    Normal = 'normal',     // float32x3 (12 bytes)
    UV = 'uv',             // float32x2 (8 bytes)
    Color = 'color',       // float32x4 (16 bytes)
}

export class VertexLayout {
    public attributes: VertexAttribute[];

    // It's helpful to store both! WebGPU needs bytes, but your
    // engine's strideFloats config needs floats.
    public strideBytes: number = 0;
    public strideFloats: number = 0;

    constructor(attributes: VertexAttribute[]) {
        this.attributes = attributes;

        // Calculate the total stride as soon as the layout is created
        for (const attr of this.attributes) {
            switch (attr) {
                case VertexAttribute.Position: this.strideBytes += 12; break;
                case VertexAttribute.Normal:   this.strideBytes += 12; break;
                case VertexAttribute.UV:       this.strideBytes += 8;  break;
                case VertexAttribute.Color:    this.strideBytes += 16; break;
            }
        }
        this.strideFloats = this.strideBytes / 4;
    }

    /**
     * Generates the exact layout object required by device.createRenderPipeline()
     */
    public getWebGPUDescriptor(): GPUVertexBufferLayout[] {
        const gpuAttributes: GPUVertexAttribute[] = [];
        let currentOffset = 0;
        let shaderLocation = 0;

        for (let i = 0; i < this.attributes.length; i++) {
            const attr = this.attributes[i];
            let format: GPUVertexFormat;
            let byteSize: number;

            // Determine the format and size for the current attribute
            switch (attr) {
                case VertexAttribute.Position:
                case VertexAttribute.Normal:
                    format = 'float32x3';
                    byteSize = 12;
                    break;
                case VertexAttribute.UV:
                    format = 'float32x2';
                    byteSize = 8;
                    break;
                case VertexAttribute.Color:
                    format = 'float32x4';
                    byteSize = 16;
                    break;
                default:
                    throw new Error(`Unknown VertexAttribute: ${attr}`);
            }

            // Push it to the attributes array
            gpuAttributes.push({
                shaderLocation: shaderLocation,
                offset: currentOffset,
                format: format
            });

            // Prepare the variables for the NEXT attribute in the loop
            shaderLocation++;
            currentOffset += byteSize;
        }

        // Return the final array containing the buffer layout
        return [{
            arrayStride: this.strideBytes,
            attributes: gpuAttributes
        }];
    }
}

// Built-in common layouts
export const CompleteLayout = new VertexLayout([VertexAttribute.Position, VertexAttribute.Normal, VertexAttribute.UV, VertexAttribute.Color]);
export const StandardLayout = new VertexLayout([VertexAttribute.Position, VertexAttribute.Normal, VertexAttribute.UV]);
export const PositionOnlyLayout = new VertexLayout([VertexAttribute.Position]);