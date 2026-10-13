// Step 7, part 2 (docs/agents-cloning-design.md §5): Herd & Wolves as the test-engine site's demo
// runs it (HerdWolvesGPU/brain.ts and index.ts), brought into this repository so the tools can run
// it headless. Until step 10 moves the demo onto this file, the two are separate copies.
//
// One change of shape from the demo's rule: its three hidden inputs and its reason word go through
// four small functions (wander, coin, resting, tag). The DRIVER text keeps them as the demo has
// them, and decides exactly what the demo's rule decides. The LABEL text replaces them, so that
// the action is a function of the species and the observation only: what a network can copy.
//
// No node: imports here: the GPU suite loads this file in a browser as well.
import * as A from '../../../dist/agents.js';

export const HERD_DEFAULTS = Object.freeze({
    count: 4096, thinkEvery: 3, scanBudget: 64, preySpeed: 3.2, wolfSpeed: 4.2,
    restSeconds: 5, flocking: true, fleeing: true, seed: 11,
});
const BASE_HALF = 80;      // m, half the field at 4,096 agents; the field's area grows with the count
const CELL = 8;            // m: grid cell = sense radius
const WOLF_SHARE = 16;     // one agent in 16 is a wolf, at the lowest indices

/** Float slots of the settings buffer. */
export const SETTINGS = Object.freeze({ FLOCKING: 0, FLEEING: 1, REST_SECONDS: 2 });
const SETTINGS_FLOATS = 4;
/** u32 slots of the stats buffer: [0] catches since the build. */
const STATS_WORDS = 4;
/** Catch events the demo draws as flashes. Nothing reads them here; the stage is kept whole. */
const CATCH_RING = 1024;

/** The teacher's reason codes, as the demo's inspector shows them. */
export const REASONS = Object.freeze(['wander', 'flock', 'flee', 'chase', 'wall', 'rest']);

/** Which of the rule's four functions keep the demo's behaviour. */
export const DRIVER = Object.freeze({ wander: true, coin: true, resting: true, tag: false });
/** The label: no wander, one fixed side on a tie, no rest branch, and a tag that carries the step. */
export const LABEL = Object.freeze({ wander: false, coin: false, resting: false, tag: true });
/** The label's decisions, driving: what the rule does without wander and coin-flips (design §8, run C). */
export const LABEL_DRIVING = Object.freeze({ wander: false, coin: false, resting: true, tag: false });

/**
 * A label's reason word: the step it was written in, whether the agent was frozen, and why.
 * The step is stored plus one, so a row no pass has written yet (word 0) reads as tick -1.
 */
export const TAG = Object.freeze({
    pack: (tick, frozen, why) => (((tick + 1) << 8) | ((frozen ? 1 : 0) << 4) | why) >>> 0,
    tick: word => (word >>> 8) - 1,
    frozen: word => (word >>> 4) & 1,
    why: word => word & 15,
});

const SEAMS = {
    wander: on => on ? `fn wander(i: u32, tick: u32) -> f32 {
    var h = ((i + 1u) * 0x9e3779b1u) ^ (((tick >> 5u) + 1u) * 0x85ebca6bu);
    h = (h ^ (h >> 15u)) * 0x2c1b3c6du;
    return f32(h ^ (h >> 13u)) / 2147483648.0 - 1.0;
}` : 'fn wander(i: u32, tick: u32) -> f32 { return 0.0; }',
    coin: on => `fn coin(i: u32) -> bool { return ${on ? '(i % 2u) == 0u' : 'true'}; }`,
    resting: on => `fn resting(i: u32) -> bool { return ${on ? 'agents[i].state.w > 0.0' : 'false'}; }`,
    tag: on => `fn tag(i: u32, tick: u32, why: u32) -> u32 { return ${on
        ? '((tick + 1u) << 8u) | (((agents[i].ids.y >> 1u) & 1u) << 4u) | why' : 'why'}; }`,
};

// The demo's HERD_BRAIN. Prey max speed comes from the species table, so a speed change keeps the
// alignment rule exact; inside think(), the `species` parameter shadows the table, hence the helper.
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
export function herdRule(seams) {
    for (const name of Object.keys(SEAMS)) {
        if (typeof seams?.[name] !== 'boolean') throw new Error(`herd-wolves: herdRule needs a boolean "${name}".`);
    }
    return `
@group(1) @binding(0) var<storage, read> settings: array<f32>;
fn preyMaxSpeed() -> f32 { return species[SPECIES_PREY].maxSpeed; }
${Object.keys(SEAMS).map(name => SEAMS[name](seams[name])).join('\n')}${THINK}`;
}

// The demo's HERD_CATCHES: read peer positions and timers, write only own flags. Timers and
// positions change in the next dispatch.
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

// The demo's herdRespawnWGSL: each catch is counted once, where the caught flag is consumed.
export const herdRespawnWGSL = flashEvery => `
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

/** The demo's definition for a count, scan budget and pair of speeds. */
export function herdDefinition(options = {}) {
    const o = { ...HERD_DEFAULTS, ...options }, half = BASE_HALF * Math.sqrt(o.count / HERD_DEFAULTS.count);
    return A.defineAgents({
        capacity: o.count,
        world: { min: [-half, -half], max: [half, half] },
        cellSize: CELL,
        maxScanPerCell: o.scanBudget,
        species: [
            { name: 'prey', maxSpeed: o.preySpeed, maxTurn: 4, accel: 6, senseRadius: CELL, radius: 0.4 },
            { name: 'wolf', maxSpeed: o.wolfSpeed, maxTurn: 3, accel: 5, senseRadius: CELL, radius: 0.6 },
        ],
        sensors: [A.Sensors.self(), A.Sensors.nearest({ name: 'wolf', species: ['wolf'], k: 1 }),
            A.Sensors.nearest({ name: 'prey', species: ['prey'], k: 3 }), A.Sensors.bounds()],
    });
}

/** A rule brain as the demo attaches it: the WGSL and its settings buffer at group 1. */
export const ruleBrain = (seams, settings) => ({ kind: 'stage', wgsl: herdRule(seams), buffers: [settings], destroy() {} });

/**
 * One build of the demo's simulation, spawned as the demo spawns it for `seed`, with the teacher
 * driving both species. The caller steps it and destroys it. `stats` word 0 counts catches.
 */
export function createHerd(engine, options = {}) {
    const o = { ...HERD_DEFAULTS, ...options }, definition = herdDefinition(o), device = engine.device;
    const wolves = o.count / WOLF_SHARE, owned = [];
    const buffer = (label, size, usage) => {
        const b = device.createBuffer({ label: `Herd ${label}`, size, usage });
        owned.push(b);
        return b;
    };
    const agents = new A.AgentSystem(engine, definition, { thinkEvery: o.thinkEvery });
    try {
        const settings = buffer('settings', SETTINGS_FLOATS * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        const stats = buffer('stats', STATS_WORDS * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        const ring = buffer('catch ring', CATCH_RING * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        const data = new Float32Array(SETTINGS_FLOATS);
        data[SETTINGS.FLOCKING] = o.flocking ? 1 : 0;
        data[SETTINGS.FLEEING] = o.fleeing ? 1 : 0;
        data[SETTINGS.REST_SECONDS] = o.restSeconds;
        device.queue.writeBuffer(settings, 0, data);
        agents.setBrain(ruleBrain(DRIVER, settings));
        agents.addStepStage({ name: 'catches', wgsl: HERD_CATCHES });
        agents.addStepStage({ name: 'respawn', wgsl: herdRespawnWGSL(Math.max(1, o.count / HERD_DEFAULTS.count)),
            buffers: [settings, stats, ring] });
        const inset = definition.world.maxX - 1;
        for (let i = 0; i < o.count; i++) {
            agents.spawn(i, {
                position: [(A.rand01(i, o.seed, 0) * 2 - 1) * inset, 0.08, (A.rand01(i, o.seed, 1) * 2 - 1) * inset],
                species: i < wolves ? 'wolf' : 'prey',
                heading: A.rand01(i, o.seed, 2) * Math.PI * 2 - Math.PI,
            });
        }
        agents.upload();
        const prey = definition.speciesIndex('prey'), wolf = definition.speciesIndex('wolf');
        return {
            agents, definition, options: o, wolves, settings, stats,
            /** The species index of agent i: fixed at spawn, and a respawn keeps it. */
            speciesOf: i => (i < wolves ? wolf : prey),
            /** What the label pass runs: the rule without what a network cannot see. */
            label: { wgsl: herdRule(LABEL), buffers: [settings] },
            /** A rule brain to drive with: 'teacher', or 'label' for the rule without wander and coin-flips. */
            rule: kind => ruleBrain(kind === 'label' ? LABEL_DRIVING : DRIVER, settings),
            destroy() {
                agents.destroy();
                for (const b of owned) b.destroy();
            },
        };
    } catch (error) {
        agents.destroy();
        for (const b of owned) b.destroy();
        throw error;
    }
}

/** What the tools need from a scenario. Only this one exists (design, non-goals). */
export const herdWolves = Object.freeze({
    name: 'herd-wolves',
    defaults: HERD_DEFAULTS,
    reasons: REASONS,
    create: createHerd,
    /** The WGSL a recording depends on, by role. header.json stores a hash of each. */
    texts(options = {}) {
        const count = options.count ?? HERD_DEFAULTS.count;
        return { driver: herdRule(DRIVER), label: herdRule(LABEL), catches: HERD_CATCHES,
            respawn: herdRespawnWGSL(Math.max(1, count / HERD_DEFAULTS.count)) };
    },
    /**
     * What evaluate.mjs reports beside catches. Each is read from one live agent's observation row
     * (`obs` at `row`), and is null where it does not apply.
     */
    measures(definition) {
        const at = name => definition.obsNames.indexOf(name), prey = definition.speciesIndex('prey');
        const speed = at('self.speed'), side = at('prey[0].side'), forward = at('prey[0].forward'), present = at('prey[0].present');
        return [
            { name: 'speed', unit: 'm/s', of: (obs, row, species) => obs[row + speed] * definition.species[species].maxSpeed },
            { name: 'nearest prey', unit: 'm', of: (obs, row, species) => (species === prey && obs[row + present] > 0
                ? Math.hypot(obs[row + side], obs[row + forward]) * definition.species[species].senseRadius : null) },
            { name: 'no prey in sight', unit: 'share', of: (obs, row, species) => (species === prey ? (obs[row + present] > 0 ? 0 : 1) : null) },
        ];
    },
    /** The settings a recording is valid for, beside its definition. */
    settings(options = {}) {
        const o = { ...HERD_DEFAULTS, ...options };
        return { restSeconds: o.restSeconds, flocking: o.flocking, fleeing: o.fleeing, wolfShare: 1 / WOLF_SHARE };
    },
});
