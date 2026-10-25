// The Node film scenes import node:crypto only to fingerprint their WGSL for the film's report. The spike
// has no report of that kind; run.mjs's bundler resolves node:crypto to this file.
export const hash = () => 'browser';
