
export interface ShaderLightingLayout {
    wgslStructs: string;
    wgslLogic: string;
}

export interface ShaderConfig {
    useSkinning: boolean;
    // Make lighting entirely optional
    lighting?: ShaderLightingLayout;
    // Toon-specific options
    useSmoothRamp?: boolean;
    // Lambert-specific options
    useHalfLambert?: boolean;
}