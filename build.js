import * as esbuild from 'esbuild';
import fs from 'fs';

// 1. Clean dist folder
if (fs.existsSync('dist')) fs.rmSync('dist', { recursive: true });

const isProd = process.argv.includes('--production');

async function build() {
    try {
        await esbuild.build({
            // NEW: Define multiple entry points!
            entryPoints: {
                'index': 'src/core/index.ts',
                'geometry': 'src/extras/geometry/index.ts'
            },
            outdir: 'dist', // NEW: Outputs to dist/index.js and dist/geometry.js
            format: 'esm',
            platform: 'browser',
            bundle: true,
            // NEW: Only generate source maps in development to keep prod clean
            sourcemap: !isProd,
            minify: isProd,
            external: ['gl-matrix'],
        });

        console.log(` NullGraph (Core + Geometry): ${isProd ? 'Production' : 'Development'} build complete.`);
    } catch (error) {
        console.error(' Build failed:', error);
        process.exit(1);
    }
}

build();