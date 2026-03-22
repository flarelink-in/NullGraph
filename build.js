import * as esbuild from 'esbuild';
import fs from 'fs';

// 1. Clean dist folder
if (fs.existsSync('dist')) fs.rmSync('dist', { recursive: true });
else fs.mkdirSync('dist');

const isProd = process.argv.includes('--production');

async function build() {
    try {
        await esbuild.build({
            entryPoints: ['src/index.ts'],
            outfile: 'dist/index.js',
            format: 'esm',
            platform: 'browser',
            bundle: true,
            sourcemap: true,
            minify: isProd,
            // Externalize gl-matrix so consumers of your library don't get duplicate copies
            external: ['gl-matrix'],
        });

        console.log(` NullGraph Core: ${isProd ? 'Production' : 'Development'} build complete.`);
    } catch (error) {
        console.error(' Build failed:', error);
        process.exit(1);
    }
}

build();