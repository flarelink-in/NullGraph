import {LightSystemConfig} from "./types";

export function getComputeShaderWGSL(config: LightSystemConfig, totalLights: number): string {
    const grid = config.clusterConfig!.grid;
    const maxLightsPerTile = config.clusterConfig!.maxLightsPerTile;

    return `
        struct Camera {
            viewProj: mat4x4<f32>,
            view: mat4x4<f32>,
            proj: mat4x4<f32>,
            invViewProj: mat4x4<f32>,
            invView: mat4x4<f32>,
            invProj: mat4x4<f32>,
            eyePosition: vec3<f32>,
            simTime: f32,
            zNear: f32,
            zFar: f32,
            fovRads: f32,
            aspect: f32,
            _padEnd: vec4<f32>
        };

        struct Light {
            pos_type: vec4<f32>,
            col_int: vec4<f32>,
            dir_rad: vec4<f32>,
            spot_params: vec4<f32>
        };

        struct Cluster {
            start_offset: u32,
            light_count: u32,
        };

        @group(0) @binding(0) var<uniform> camera: Camera;
        @group(0) @binding(1) var<storage, read> globalLights: array<Light>;
        @group(0) @binding(2) var<storage, read_write> clusterGrid: array<Cluster>;
        @group(0) @binding(3) var<storage, read_write> lightIndices: array<u32>;
        @group(0) @binding(4) var<storage, read_write> globalCounter: atomic<u32>;

        const GRID_X = ${grid[0]}u;
        const GRID_Y = ${grid[1]}u;
        const GRID_Z = ${grid[2]}u;
        const MAX_LIGHTS_PER_TILE = ${maxLightsPerTile}u;
        const TOTAL_LIGHTS = ${totalLights}u;

        // Simplified AABB calculation - use frustum planes instead of 8 corners
        fn getClusterAABB(clusterIndex: u32) -> vec4<f32> {
            let z = clusterIndex / (GRID_X * GRID_Y);
            let y = (clusterIndex % (GRID_X * GRID_Y)) / GRID_X;
            let x = clusterIndex % GRID_X;

            // Simplified: just use cluster center and approximate extents
            let zNear = camera.zNear;
            let zFar = camera.zFar;
            
            // Logarithmic depth
            let tileNear = -zNear * pow(zFar / zNear, f32(z) / f32(GRID_Z));
            let tileFar  = -zNear * pow(zFar / zNear, f32(z + 1u) / f32(GRID_Z));
            let tileMid = (tileNear + tileFar) * 0.5;
            let tileExtent = (tileFar - tileNear) * 0.5;

            // Approximate XY extents at this depth
            let tanHalfFov = tan(camera.fovRads * 0.5);
            let yExtent = -tileMid * tanHalfFov / f32(GRID_Y);
            let xExtent = yExtent * camera.aspect / f32(GRID_X);
            
            // Cluster center in view space
            let cx = (f32(x) + 0.5) / f32(GRID_X) * 2.0 - 1.0;
            let cy = (f32(y) + 0.5) / f32(GRID_Y) * 2.0 - 1.0;
            
            return vec4<f32>(
                cx * xExtent * f32(GRID_X),
                cy * yExtent * f32(GRID_Y),
                tileMid,
                length(vec3<f32>(xExtent, yExtent, tileExtent))
            );
        }

        @compute @workgroup_size(64, 1, 1)
        fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let clusterIndex = global_id.x;
            if (clusterIndex >= (GRID_X * GRID_Y * GRID_Z)) { return; }

            // Get approximate cluster sphere
            let clusterSphere = getClusterAABB(clusterIndex);
            let clusterCenter = clusterSphere.xyz;
            let clusterRadius = clusterSphere.w;

            var visibleLights: array<u32, ${maxLightsPerTile}>;
            var visibleCount = 0u;

            for (var i = 0u; i < TOTAL_LIGHTS && visibleCount < MAX_LIGHTS_PER_TILE; i = i + 1u) {
                let light = globalLights[i];
                let lightType = light.pos_type.w;

                if (lightType == 2.0) {
                    visibleLights[visibleCount] = i;
                    visibleCount = visibleCount + 1u;
                    continue;
                }

                // Sphere vs Sphere test (much faster than AABB vs Sphere)
                let lightPosView = (camera.view * vec4<f32>(light.pos_type.xyz, 1.0)).xyz;
                let lightRadius = light.dir_rad.w;
                
                let distToLight = distance(clusterCenter, lightPosView);
                
                if (distToLight <= (clusterRadius + lightRadius)) {
                    visibleLights[visibleCount] = i;
                    visibleCount = visibleCount + 1u;
                }
            }

            let offset = atomicAdd(&globalCounter, visibleCount);
            clusterGrid[clusterIndex] = Cluster(offset, visibleCount);

            for (var i = 0u; i < visibleCount; i = i + 1u) {
                lightIndices[offset + i] = visibleLights[i];
            }
        }
    `;
}