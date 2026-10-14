// The page of spikes A and B (run.mjs): it hands its canvas to a module worker, as the site's
// EngineCanvas does, and asks the worker for the device switch when the driver says so.
const loads = Number(sessionStorage.getItem('spike-loads') ?? 0) + 1;
sessionStorage.setItem('spike-loads', String(loads));
console.log(`[spike] page load ${loads}`);

const canvas = document.querySelector('canvas');
const offscreen = canvas.transferControlToOffscreen();
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
worker.onerror = (event) => console.error(`[spike] worker error: ${event.message}`);
worker.postMessage({ type: 'INIT', canvas: offscreen, width: canvas.width, height: canvas.height,
    settings: Object.fromEntries(new URLSearchParams(location.search)) }, [offscreen]);
window.spikeSwitch = () => worker.postMessage({ type: 'SWITCH' });
