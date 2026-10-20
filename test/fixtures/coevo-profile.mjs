// Test workload only; not measured or accepted research settings.
export const coevoProfile = (overrides = {}) => ({
    name: 'test-coevo-profile', K: 8, E: 1, P: 192, W: 16, L: 24, cols: 2, gap: 10,
    T: 300, dt: 1 / 60, thinkEvery: 3, environment: 'contact-v2', contacts: false,
    contactOptions: {}, catchMargin: 0.05, generations: 10, start: 'random', hidden: [16, 16],
    optimizer: { sigma: 0.1, rate: 0.01, decay: 0.005 }, archive: { every: 5, cap: 32, share: 0.5 },
    seeds: { master: 18001, domain: 1 }, ...overrides,
});