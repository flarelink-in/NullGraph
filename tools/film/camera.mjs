// D2.1 (docs/offline-films-design.md §9): a camera path from keyframes in orbit terms — target, distance,
// azimuth, polar — evaluated at film time. Since D2.4 the path is null-graph/film's
// (offline-films-browser-design.md §4.2), shared with the browser's films: cubic Hermite with Catmull-Rom
// slopes, eased ends, distance on a log scale, and an optional field of view. Build first.

export { cameraPath } from '../../dist/film.js';
