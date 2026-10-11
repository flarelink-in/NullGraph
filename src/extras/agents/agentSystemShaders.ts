/** GPU-owned step state; no CPU write varies between copies of a step chain. */
export const AGENT_STATE_WGSL = `
struct AgentState {
    tick: u32,
    overflowSteps: u32,
    overflowTotal: u32,
    _pad: u32,
};
`;

export function buildAgentTickWGSL(): string {
    return `${AGENT_STATE_WGSL}
@group(0) @binding(0) var<storage, read> overflow: u32;
@group(0) @binding(1) var<storage, read_write> state: AgentState;
@compute @workgroup_size(1) fn cs_main() {
    state.tick += 1u;
    if (overflow > 0u) {
        state.overflowSteps += 1u;
        state.overflowTotal += overflow;
    }
}`;
}