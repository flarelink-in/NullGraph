// src/data/engine-demos/demos/AgentsExperiments/HerdWolvesGPU/brain.ts
//
// The rule brain and the two app stages that make the game: catches, then respawn. Live settings
// sit in one caller-owned buffer that the brain and the respawn stage both bind at group 1.
//
// A copy of null-graph's tools/agents/scenario/herd-wolves.mjs, the scenario the networks were
// trained on (null-graph step 10, decision D3). Keep its WGSL texts byte for byte as there:
// - the rule goes through four small functions (wander, coin, resting, tag). herdRule(DRIVER) is
//   the rule the demo runs, and decides what the CPU preview's rule (HerdWolvesPreview/brain.ts)
//   decides;
// - herdRule(LABEL) is the rule without what a network cannot see: no wander, one fixed side on a
//   tie, no rest branch. The networks were trained to copy it, and each <species>.mlp.json carries
//   the SHA-256 of that text (trainedOn.teacher). The demo hashes its own HERD_LABEL against it, so
//   a change to the rule that the networks have not learned shows on the overlay.

/** Float slots of the settings buffer. */
export const SETTINGS = { FLOCKING: 0, FLEEING: 1, REST_SECONDS: 2 } as const;
export const SETTINGS_FLOATS = 4;

/** u32 slots of the stats buffer: [0] catches since the build. */
export const STATS_WORDS = 4;

/** Catch events kept for the flashes: x, z, step of the catch, unused. Oldest overwritten first. */
export const CATCH_RING = 1024;

/** Which of the rule's four functions keep the demo's behaviour (true) or are replaced (false). */
export type HerdSeams = Record<'wander' | 'coin' | 'resting' | 'tag', boolean>;

/** The rule as the demo runs it. */
export const DRIVER: HerdSeams = { wander: true, coin: true, resting: true, tag: false };
/** What the networks copy: no wander, one fixed side on a tie, no rest branch, a tag with the step. */
export const LABEL: HerdSeams = { wander: false, coin: false, resting: false, tag: true };

const SEAMS: Record<keyof HerdSeams, (on: boolean) => string> = {
  wander: (on) =>
    on
      ? `fn wander(i: u32, tick: u32) -> f32 {
    var h = ((i + 1u) * 0x9e3779b1u) ^ (((tick >> 5u) + 1u) * 0x85ebca6bu);
    h = (h ^ (h >> 15u)) * 0x2c1b3c6du;
    return f32(h ^ (h >> 13u)) / 2147483648.0 - 1.0;
}`
      : 'fn wander(i: u32, tick: u32) -> f32 { return 0.0; }',
  coin: (on) => `fn coin(i: u32) -> bool { return ${on ? '(i % 2u) == 0u' : 'true'}; }`,
  resting: (on) =>
    `fn resting(i: u32) -> bool { return ${on ? 'agents[i].state.w > 0.0' : 'false'}; }`,
  tag: (on) =>
    `fn tag(i: u32, tick: u32, why: u32) -> u32 { return ${
      on ? '((tick + 1u) << 8u) | (((agents[i].ids.y >> 1u) & 1u) << 4u) | why' : 'why'
    }; }`,
};

// Prey max speed comes from the species table, not a constant, so the speed slider (setSpecies)
// keeps the alignment rule exact: the sense stage scales relative velocities by the observer's
// maxSpeed. Inside think(), the `species` parameter shadows the table, hence the helper.
const THINK = `
fn think(i: u32, species: u32, tick: u32) {
    var turn = 0.0; var throttle = 0.0; var why = 0u;
    if (species == SPECIES_WOLF) {
        if (resting(i)) {
            setAct(i, ACT_TURN, 0.0); setAct(i, ACT_THROTTLE, -1.0); setReason(i, tag(i, tick, 5u)); return;
        }
        if (obs(i, OBS_PREY_0 + NEAR_PRESENT) > 0.0) {
            turn = 2.5 * atan2(obs(i, OBS_PREY_0 + NEAR_SIDE), obs(i, OBS_PREY_0 + NEAR_FORWARD));
            throttle = 1.0; why = 3u;
        } else { turn = 0.6 * wander(i, tick); throttle = -0.2; }
        let ws = obs(i, OBS_WOLF_0 + NEAR_SIDE); let wf = obs(i, OBS_WOLF_0 + NEAR_FORWARD);
        let wd = length(vec2<f32>(ws, wf));
        if (obs(i, OBS_WOLF_0 + NEAR_PRESENT) > 0.0 && wd < 0.35) { turn -= sign(ws) * (0.35 - wd) * 3.0; }
    } else {
        throttle = 0.1;
        var urgency = 0.0;
        if (settings[${SETTINGS.FLEEING}] > 0.0 && obs(i, OBS_WOLF_0 + NEAR_PRESENT) > 0.0) {
            let ws = obs(i, OBS_WOLF_0 + NEAR_SIDE); let wf = obs(i, OBS_WOLF_0 + NEAR_FORWARD);
            urgency = clamp(1.2 - length(vec2<f32>(ws, wf)), 0.0, 1.0);
            turn = 2.0 * atan2(-ws, -wf) * urgency; throttle = 1.0; why = 2u;
        }
        if (settings[${SETTINGS.FLOCKING}] > 0.0) {
            var n = 0.0; var cs = 0.0; var cf = 0.0; var align = 0.0; var separate = 0.0;
            let maxSpeed = preyMaxSpeed();
            let mySpeed = obs(i, OBS_SELF + SELF_SPEED) * maxSpeed;
            for (var k = 0u; k < 3u; k++) {
                let p = OBS_PREY_0 + k * 5u;
                if (obs(i, p + NEAR_PRESENT) > 0.0) {
                    let s = obs(i, p + NEAR_SIDE); let f = obs(i, p + NEAR_FORWARD);
                    cs += s; cf += f;
                    let vs = obs(i, p + NEAR_VEL_SIDE) * 2.0 * maxSpeed;
                    let vf = obs(i, p + NEAR_VEL_FORWARD) * 2.0 * maxSpeed + mySpeed;
                    if (vs != 0.0 || vf != 0.0) { align += atan2(vs, vf); }
                    let d = length(vec2<f32>(s, f));
                    if (d < 0.3) {
                        var side = s;
                        if (side == 0.0) { side = select(-1.0, 1.0, !coin(i)); }
                        separate -= sign(side) * (0.3 - d) * 4.0;
                    }
                    n += 1.0;
                }
            }
            if (n > 0.0) {
                var forward = cf / n;
                if (forward == 0.0) { forward = 0.000001; }
                let flock = 0.4 * atan2(cs / n, forward) + 0.6 * (align / n) + separate;
                turn += flock * (1.0 - urgency);
                if (why == 0u) { throttle = 0.25; why = 1u; }
            }
        }
        turn += 0.3 * wander(i, tick) * (1.0 - urgency);
    }
    let side = obs(i, OBS_BOUNDS + BOUNDS_SIDE); let forward = obs(i, OBS_BOUNDS + BOUNDS_FORWARD);
    let m = length(vec2<f32>(side, forward));
    if (m > 0.05) {
        var away = select(-1.0, 1.0, coin(i));
        if (abs(side) > 0.05) { away = -sign(side); }
        let wallTurn = away * m * select(1.0, 2.0, forward > 0.0);
        let k = min(1.0, m * 1.5);
        turn = turn * (1.0 - k) + wallTurn * k;
        if (m > 0.4) { why = 4u; }
    }
    setAct(i, ACT_TURN, clamp(turn, -1.0, 1.0));
    setAct(i, ACT_THROTTLE, clamp(throttle, -1.0, 1.0)); setReason(i, tag(i, tick, why));
}`;

/** The rule's WGSL with each of its four functions kept (true) or replaced (false). */
export function herdRule(seams: HerdSeams): string {
  const names = Object.keys(SEAMS) as (keyof HerdSeams)[];
  return `
@group(1) @binding(0) var<storage, read> settings: array<f32>;
fn preyMaxSpeed() -> f32 { return species[SPECIES_PREY].maxSpeed; }
${names.map((name) => SEAMS[name](seams[name])).join('\n')}${THINK}`;
}

/** The rule the demo runs. */
export const HERD_BRAIN = herdRule(DRIVER);
/** The text the networks copy; its SHA-256 is each network's trainedOn.teacher. */
export const HERD_LABEL = herdRule(LABEL);

// Read peer positions/timers, write only own flags. Timers/positions change in the next dispatch.
export const HERD_CATCHES = `
fn step(i: u32, tick: u32) {
    let me = agents[i];
    if (me.ids.x == SPECIES_PREY) {
        let w = neighbours[i * NEIGHBOURS_PER_AGENT + NEIGHBOURS_WOLF];
        if (w != NO_AGENT && agents[w].state.w <= 0.0 && distance(agents[w].posRadius.xz, me.posRadius.xz) < 0.9) {
            agents[i].ids.y |= 256u;
        }
    } else if (me.state.w <= 0.0) {
        let p = neighbours[i * NEIGHBOURS_PER_AGENT + NEIGHBOURS_PREY];
        if (p != NO_AGENT && distance(agents[p].posRadius.xz, me.posRadius.xz) < 0.9) { agents[i].ids.y |= 512u; }
    }
}`;

// Each catch is counted once, here, where the caught flag is consumed, and every `flashEvery`-th
// is recorded in the ring the flashes draw from: catches grow with the wolves, so larger sizes
// sample them and the screen keeps about as many flashes as at 4K. Both writes go to app-owned
// buffers, never to a peer's record.
export const herdRespawnWGSL = (flashEvery: number) => `
@group(1) @binding(0) var<storage, read> settings: array<f32>;
@group(1) @binding(1) var<storage, read_write> stats: array<atomic<u32>, ${STATS_WORDS}>;
@group(1) @binding(2) var<storage, read_write> catchRing: array<vec4<f32>, ${CATCH_RING}>;
fn step(i: u32, tick: u32) {
    if (agents[i].ids.x == SPECIES_WOLF) {
        agents[i].state.w = max(0.0, agents[i].state.w - DT);
        if ((agents[i].ids.y & 512u) != 0u) {
            agents[i].state.w = settings[${SETTINGS.REST_SECONDS}];
            agents[i].ids.y &= ~512u;
        }
        // Frozen act stops immediately, even on a non-thinking step; age still advances.
        if (agents[i].state.w > 0.0) {
            agents[i].ids.y |= 2u;
            agents[i].velHeading.xyz = vec3<f32>(0.0);
        } else { agents[i].ids.y &= ~2u; }
    } else if ((agents[i].ids.y & 256u) != 0u) {
        let n = atomicAdd(&stats[0], 1u);
        if (n % ${flashEvery}u == 0u) {
            catchRing[(n / ${flashEvery}u) % ${CATCH_RING}u] = vec4<f32>(agents[i].posRadius.x, agents[i].posRadius.z, f32(tick), 0.0);
        }
        respawn(i, randomPointInWorld(i, tick), rand01(i, tick, 2u) * 6.2831853 - 3.1415927);
        agents[i].ids.y &= ~256u;
    }
}`;
