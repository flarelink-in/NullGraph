// Reserved grid stream: independent of application rand01 streams, identical u32 arithmetic.
export function gridWindow(i: number, tick: number): number {
    const seed = (i ^ Math.imul(tick, 0x9e3779b9) ^ Math.imul(0xffffffff, 0x85ebca6b)) >>> 0;
    const state = (Math.imul(seed, 747796405) + 2891336453) >>> 0;
    const word = Math.imul((state >>> ((state >>> 28) + 4)) ^ state, 277803737) >>> 0;
    return ((word >>> 22) ^ word) >>> 0;
}

export const GRID_WINDOW_WGSL = `
fn gridWindow(i: u32, tick: u32) -> u32 {
    let seed = i ^ (tick * 0x9e3779b9u) ^ (0xffffffffu * 0x85ebca6bu);
    let hashState = seed * 747796405u + 2891336453u;
    let word = ((hashState >> ((hashState >> 28u) + 4u)) ^ hashState) * 277803737u;
    return (word >> 22u) ^ word;
}`;