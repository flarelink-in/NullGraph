import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();

// Check if we want SAB enabled via command line arg
const enableSAB = !process.argv.includes('--no-sab');

if (enableSAB) {
    app.use((req, res, next) => {
        res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
        res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
        res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate, proxy-revalidate");
        next();
    });
}

// Serve the root directory so we can access everything instantly
app.use(express.static(__dirname));

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`\n NullGraph Test Server running at: http://localhost:${PORT}`);
    console.log(` SharedArrayBuffer enabled: ${enableSAB ? 'YES' : 'NO'}\n`);
});