// Shorthand ambient modules for null-graph/film's two dynamic imports, so TypeScript reads neither
// package's declarations while building this entry:
// - 'mediabunny': its declarations reference @types/dom-webcodecs; mediabunny.ts casts the namespace to
//   MediabunnyLike.
// - 'null-graph': the package's own main entry, imported at run time so a film's engine is the same core
//   the app uses (record.ts); its types come from '../../core', as type-only imports.
declare module 'mediabunny';
declare module 'null-graph';
