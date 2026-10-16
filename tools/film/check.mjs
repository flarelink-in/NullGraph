// D2.1 (docs/offline-films-design.md §10): the scenes' copies of the test site's files, against the site.
//
//   npm run film:check                 lists each copy: the same as the site's, changed there, or missing
//   npm run film:check -- --update     copies the site's files again
//   NG_SITE                            the site's folder (default: the user's local checkout)
//
// Exit 0 when every copy equals the site's file (or after --update); 1 when one differs or is missing.
import { copyFileSync, existsSync, readFileSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const SITE = process.env.NG_SITE ?? 'D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts';
const SCENES = fileURLToPath(new URL('./scenes/', import.meta.url));
const sha256 = file => createHash('sha256').update(readFileSync(file)).digest('hex');

/** Every scene's copies: { scene, file, copy, source, state: 'same' | 'changed' | 'missing in the site' | 'missing here', copyHash }. */
export function compareCopies(site = SITE) {
    const out = [];
    for (const scene of readdirSync(SCENES, { withFileTypes: true }).filter(d => d.isDirectory())) {
        const list = join(SCENES, scene.name, 'copies.json');
        if (!existsSync(list)) continue;
        const { from, files } = JSON.parse(readFileSync(list, 'utf8'));
        for (const file of files) {
            const copy = join(SCENES, scene.name, file), source = join(site, from, file);
            const state = !existsSync(source) ? 'missing in the site' : !existsSync(copy) ? 'missing here'
                : sha256(copy) === sha256(source) ? 'same' : 'changed';
            out.push({ scene: scene.name, file, copy, source, state, copyHash: existsSync(copy) ? sha256(copy) : null });
        }
    }
    return out;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
    const update = process.argv.includes('--update');
    if (!existsSync(SITE)) {
        console.error(`film:check: the site is not at ${SITE} (NG_SITE).`);
        process.exit(1);
    }
    let differ = 0;
    for (const c of compareCopies()) {
        if (c.state !== 'same' && update && existsSync(c.source)) {
            copyFileSync(c.source, c.copy);
            console.log(`updated   ${c.scene}/${c.file}`);
            continue;
        }
        if (c.state !== 'same') differ++;
        console.log(`${c.state.padEnd(9)} ${c.scene}/${c.file}`);
    }
    if (differ) console.log(`\n${differ} cop${differ === 1 ? 'y differs' : 'ies differ'} from the site. ` +
        'Read the site\'s change, then run npm run film:check -- --update, and check the scene still draws as the demo.');
    process.exit(differ ? 1 : 0);
}
