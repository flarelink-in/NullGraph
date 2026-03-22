import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// We copy your test HTML and the built engine bundle
const filesToCopy = [
    { src: 'test/index.html', dest: 'dist-web-test/index.html' },
];

const foldersToCopy = [
    { src: 'test/css', dest: 'dist-web-test/css' },
    { src: 'test/assets', dest: 'dist-web-test/assets' } // If you add GLTF models later
];

function copyDir(src, dest) {
    if (!fs.existsSync(src)) return;
    fs.mkdirSync(dest, { recursive: true });
    let entries = fs.readdirSync(src, { withFileTypes: true });
    for (let entry of entries) {
        let srcPath = path.join(src, entry.name);
        let destPath = path.join(dest, entry.name);
        entry.isDirectory() ? copyDir(srcPath, destPath) : fs.copyFileSync(srcPath, destPath);
    }
}

// Ensure the target directory exists
if (!fs.existsSync('dist-web-test')) fs.mkdirSync('dist-web-test', { recursive: true });

filesToCopy.forEach(file => {
    const srcPath = path.resolve(__dirname, file.src);
    const destPath = path.resolve(__dirname, file.dest);
    if (fs.existsSync(srcPath)) {
        fs.copyFileSync(srcPath, destPath);
        console.log(` Copied: ${file.src}`);
    }
});

foldersToCopy.forEach(folder => {
    const srcPath = path.resolve(__dirname, folder.src);
    const destPath = path.resolve(__dirname, folder.dest);
    copyDir(srcPath, destPath);
    console.log(` Copied folder: ${folder.src}`);
});