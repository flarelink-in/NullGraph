// The Node film scenes import node:crypto only to fingerprint their WGSL for the film's report. The page's
// import map resolves node:crypto to this file; a browser film carries the workload's info instead.
export const hash = () => 'browser';
