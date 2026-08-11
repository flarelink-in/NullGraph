// light-types.ts

export type LightingTechnique = 'none' | 'standard' | 'tiled-forward' | 'clustered-forward';

export interface LightSystemCapacity {
    point: number;
    directional: number;
    spot: number;
    ambient?: number;
    hemisphere?: number;
    rectArea?: number;
    torch?: number;
}

export interface ClusteredConfig {
    grid: [number, number, number]; // e.g., [16, 9, 24] volumetric tiles
    maxLightsPerTile: number;
}

export interface TiledConfig {
    tileSize: [number, number]; // e.g., [16, 16] pixel tiles in screen space
    maxLightsPerTile: number;
}

export interface LightSystemConfig {
    capacity: LightSystemCapacity;
    technique: LightingTechnique;

    // Only required/used if technique is 'clustered-forward'
    clusterConfig?: ClusteredConfig;

    // Only required/used if technique is 'tiled-forward'
    tiledConfig?: TiledConfig;
}

export interface ShaderLightingLayout {
    wgslStructs: string;
    wgslLogic: string;
}
