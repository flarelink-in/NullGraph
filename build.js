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
                'geometry': 'src/extras/geometry/index.ts',
                'loaders': 'src/extras/loaders/index.ts',
                'materials':'src/extras/materials/index.ts',
                'profiler':'src/extras/profiler/index.ts',
                'debug-ui':'src/extras/debug-ui/index.ts',
                'cameras':'src/extras/cameras/index.ts',
                'lights':'src/extras/lights/index.ts',
                'post':'src/extras/post/index.ts',
                'visibility':'src/extras/visibility/index.ts',
                'shadows':'src/extras/shadows/index.ts',
                'environment':'src/extras/environment/index.ts',
                'picking':'src/extras/picking/index.ts',
                'text':'src/extras/text/index.ts',
                'agents':'src/extras/agents/index.ts',
                'inference':'src/extras/inference/index.ts',
                'physics':'src/extras/physics/index.ts',
                'film':'src/extras/film/index.ts',
            },

            outdir: 'dist', // NEW: Outputs to dist/index.js and dist/geometry.js
            format: 'esm',
            platform: 'browser',
            bundle: true,
            // NEW: Only generate source maps in development to keep prod clean
            sourcemap: !isProd,
            minify: isProd,
            // mediabunny is an optional peer of null-graph/film, imported only when a film encodes.
            // null-graph/film imports the package's own core at run time, so a film's engine is the
            // app's core rather than a second copy bundled into film.js.
            external: ['gl-matrix', 'mediabunny', 'null-graph'],
        });

        console.log(` NullGraph (Core + Geometry + Loaders + Materials + Profiler + Debug-UI + Cameras + Lights + Post + Visibility + Shadows + Environment + Picking + Text + Agents + Inference + Physics + Film): ${isProd ? 'Production' : 'Development'} build complete.`);
    } catch (error) {
        console.error(' Build failed:', error);
        process.exit(1);
    }
}

build();