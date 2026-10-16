# NullGraph — every use case, rated (after Phase 2)

**What this is:** the list in [`use-cases-current.md`](./use-cases-current.md), rated again as if
Phase 2 (the milestone after C2) were built, plus the use cases Phase 2 makes possible. Same
columns, same scoring. Written 2026-10-16.

**What "after Phase 2" means here:** the user was unsure what it would cover, so this file assumes
all of it: the agents' Phase 2 in [`agents-design.md`](./agents-design.md) §13 ("Deferred — Phase
2"), the brain work the user set out on 2026-10-15 ([`work/plan.md`](./work/plan.md), scope
changes), the physics plugin's P1–P3 ([`physics/README.md`](./physics/README.md)) and rigged
models (Cycle D, phase 5.4). D2, offline films ([design](./offline-films-design.md)), comes before
Phase 2 in the plan and is assumed built: it makes the videos of C3 and E1. Every use case is
tagged with the pieces it needs (§1), so the table also reads for a partial Phase 2.

**Not this file:** a plan or a schedule. The order of Phase 2's pieces is the user's decision; §6
gives the case for one.

**Effort** is counted on top of the library *at that time*: a use case that needs walls costs no
more once the library has walls.

---

## Summary

**Top 3 after Phase 2** (reasons in §8):

1. **I13 — a venue crowd-flow and evacuation planner on your event platform.** Real, recurring need;
   commercial tools are priced for engineers ($2,350 a year and up) and free ones for researchers;
   your platform holds the ticket counts and entry slots that make the simulation specific. Likely
   $10k–40k in year 1.
2. **One evolution engine, three outlets:** a natural-selection lab for classrooms (E8), an
   evolution video and explainer series (C3, E1) and a sandbox game (G10). Mostly audience at
   first, with real ceilings.
3. **The horde game, upgraded (G1, G2):** walls and base building from NAV, and enemies that evolve
   against the player's build from EVO, sold to the players the first game found.

**What it means for the order of Phase 2 (§6):** walls, obstacles and flow fields (NAV) first; the
agent life cycle and evolution (EVO), with ray eyes, second; flight (3D) third; physics last.

---

## 1. What Phase 2 adds

| Tag | What it adds | Where it is planned |
|---|---|---|
| **NAV** | Walls, obstacles and flow fields that agents steer around and route through; sensors that read app buffers (maps, fields, pheromones), so agents perceive the world | `agents-design.md` §13 (walls, obstacles, flow fields); `plan.md`, 2026-10-15 (a sensor that reads app buffers) |
| **EVO** | The agent life cycle on the GPU: energy and food, birth, death, reproduction and mutation; a genome per agent as a weight-table row | `agents-design.md` §13 (evolution); the record's reserved `genome` field (§4.1) |
| **EYES** | Ray sensors ("1D eyes"), later rendered eyes; what agents see depends on light, night and fog | `agents-design.md` §13 (eyes); `handoff.md`, "Agents that see" |
| **3D** | Flying and swimming agents: height integrated, not only carried | `agents-design.md` §13 |
| **BRAIN+** | A wider `MlpBrain`: a tiled kernel, wider layers, discrete output heads, memory, per-agent weight rows | `plan.md`, 2026-10-15 (brains after C2) |
| **SRV** | Long headless runs on a rented Linux GPU; the best brains replayed in the browser | `agents-design.md` §13; `handoff.md`, "Far horizon" |
| **PHYS** | The physics plugin, P1–P3: a fixed-step bridge to Rapier or Jolt, a GPU broadphase, XPBD particles and cloth | `physics/README.md` |
| **SKIN** | Rigged animals and people drawn through the skinned bake and palettes | Cycle D, phase 5.4 (waiting on assets) |

Also in Phase 2 but not tagged, because they improve quality rather than unlock a use case: render
interpolation, amortized think slices, f16 (detected only).

---

## 2. How to read the ratings

The columns and the score are those of the
[current-state file, §2](./use-cases-current.md#2-how-to-read-the-ratings). Two more:

- **Needs:** the Phase 2 pieces the use case depends on, or is improved by. "—" means none.
- **Δ:** the change in score against today. "new" marks a use case that is not possible today.

---

## 3. All use cases after Phase 2, ranked

| # | ID | Use case | Needs | Effort | Need | Likely /yr | Ceiling /yr | Edge | Reach | Score | Δ |
|---:|---|---|---|---|---:|---|---|---:|---:|---:|---|
| 1 | E1 | ML and evolution explainer series | EVO | S | 4 | $1k–10k (2) | $20k–150k | 3 | 1 | 12 | +4 |
| 2 | I13 | Venue crowd-flow and evacuation planner on your platform | NAV, SKIN | L | 4 | $10k–40k (3) | $100k–500k | 3 | 2 | 11 | new |
| 3 | C1 | Branded interactive web experiences (freelance) | 3D, EVO, PHYS | S/project | 3 | $10k–40k (3) | $80k–250k | 3 | 0 | 11 | +1 |
| 4 | C3 | Evolution video series | EVO | M | 4 | $1k–15k (2) | $100k–500k | 3 | 1 | 10 | +4 |
| 5 | E8 | Natural-selection and evolution lab | EVO, EYES | M | 4 | $1k–15k (2) | $50k–200k | 3 | 1 | 10 | new |
| 6 | G1 | Survivors-like horde game | NAV, EVO, 3D | M | 4 | $1k–15k (2) | $100k–1M+ | 3 | 1 | 10 | 0 |
| 7 | G2 | Horde survival with walls and base building | NAV | M | 4 | $1k–15k (2) | $100k–1M | 3 | 1 | 10 | +2 |
| 8 | I1 | Event photo booth on your platform | — | L | 4 | $2k–25k (2) | $100k–500k | 2 | 2 | 8 | 0 |
| 9 | C2 | Living installations that evolve | EVO, EYES | M | 3 | $0–30k (2) | $50k–250k | 3 | 1 | 8 | +1 |
| 10 | E2 | Ecosystem classroom simulator with a real life cycle | EVO | M | 3 | $1k–10k (2) | $20k–150k | 3 | 1 | 8 | +3 |
| 11 | E9 | Crowd-safety training for event staff | NAV, SKIN | M | 3 | $2k–15k (2) | $40k–150k | 2 | 2 | 8 | new |
| 12 | G4 | Idle swarm game with an evolving colony | EVO | M | 3 | $0.5k–8k (2) | $20k–150k | 3 | 1 | 8 | +1 |
| 13 | D2 | Paid starter kits and templates | NAV, EVO, 3D | S | 2 | $1k–8k (2) | $20k–80k | 3 | 1 | 8 | +2 |
| 14 | E3 | Epidemic-spread simulator with buildings | NAV, EVO | S | 3 | $0–3k (1) | $10k–60k | 2 | 1 | 7 | +1 |
| 15 | E6 | Physics and chemistry with real collisions | PHYS | S | 3 | $0–2k (1) | $5k–40k | 2 | 1 | 7 | +3 |
| 16 | R1 | Collective motion: 3D, obstacles, vision | 3D, NAV, EYES | S | 3 | $0–2k (1) | $10k–60k | 3 | 0 | 7 | +2 |
| 17 | E4 | Learn-WebGPU course built on NullGraph | — | M | 3 | $1k–15k (2) | $50k–500k | 2 | 1 | 7 | 0 |
| 18 | I4 | Architecture/real-estate visualization, indoors too | NAV, SKIN | M | 3 | $5k–25k (2) | $50k–150k | 3 | 0 | 7 | +1 |
| 19 | D3 | Swarm playground: evolve and share creatures | EVO | M | 3 | $0–2k (1) | $10k–60k | 3 | 1 | 6 | +2 |
| 20 | G3 | Massive-battle sandbox | NAV, 3D, PHYS, EYES | L | 3 | $1k–15k (2) | $100k–1M+ | 3 | 1 | 6 | 0 |
| 21 | G10 | Evolution sandbox game | EVO, EYES, SKIN | L | 3 | $2k–20k (2) | $50k–500k | 3 | 1 | 6 | new |
| 22 | G12 | Stealth game where guards really see | EYES, NAV | L | 3 | $1k–15k (2) | $50k–500k | 3 | 1 | 6 | new |
| 23 | C4 | Generative art | 3D, EVO | S | 2 | $0–3k (1) | $5k–50k | 3 | 1 | 6 | +1 |
| 24 | D1 | GPU benchmark site | — | S | 2 | $0–1k (1) | $5k–20k | 3 | 1 | 6 | 0 |
| 25 | D4 | Neural NPC crowds for game studios | BRAIN+ | S | 2 | $0–3k (1) | $5k–40k | 3 | 1 | 6 | 0 |
| 26 | G6 | Ambient aquarium, desktop toy, live wallpaper | 3D, EVO | S | 2 | $0–3k (1) | $5k–40k | 3 | 1 | 6 | +1 |
| 27 | G11 | Colony or god game with thousands of citizens | EVO, NAV, BRAIN+ | XL | 4 | $1k–20k (2) | $100k–1M+ | 2 | 1 | 5 | new |
| 28 | I7 | Drone-show design and previs | 3D | M | 3 | $0–5k (1) | $20k–150k | 3 | 0 | 5 | +2 |
| 29 | R2 | GPU agent-based-modelling toolkit for any GPU | NAV, EVO | M | 3 | $0–3k (1) | $20k–100k | 3 | 0 | 5 | +2 |
| 30 | R4 | Swarm robotics, ground and aerial | NAV, 3D, EYES | M | 3 | $0–3k (1) | $10k–50k | 3 | 0 | 5 | +3 |
| 31 | R10 | Artificial-life and open-ended-evolution platform | EVO, EYES, SRV | M | 3 | $0–5k (1) | $20k–150k | 3 | 0 | 5 | new |
| 32 | G5 | Herding game with real pens | NAV | S | 2 | $0–3k (1) | $10k–100k | 2 | 1 | 5 | 0 |
| 33 | E5 | Classic-model library with ants and slime mould | NAV, EVO | M | 3 | $0–2k (1) | $10k–50k | 2 | 0 | 4 | +2 |
| 34 | R12 | Pedestrian-dynamics research | NAV | M | 3 | $0–3k (1) | $10k–50k | 2 | 0 | 4 | new |
| 35 | C6 | Offline-rendered loops and stock footage | — | S | 2 | $0–1k (1) | $2k–20k | 1 | 1 | 4 | 0 |
| 36 | C7 | Data-journalism: crowd and evacuation explainers | NAV | S | 2 | $0–3k (1) | $5k–30k | 2 | 0 | 4 | +1 |
| 37 | D6 | GPU physics kit for web developers | PHYS | S | 2 | $0–2k (1) | $5k–30k | 1 | 1 | 4 | new |
| 38 | R3 | Imitation-learning testbed | BRAIN+ | S | 2 | ~$0 (1) | $0–10k | 2 | 0 | 4 | 0 |
| 39 | R8 | Math and science-communication visuals | — | S | 2 | ~$0 (1) | $2k–20k | 1 | 1 | 4 | 0 |
| 40 | C5 | Live visuals (VJ) | 3D | M | 2 | $0–2k (1) | $5k–30k | 3 | 1 | 4 | +1 |
| 41 | G7 | Stream-interactive swarm battles | — | M | 2 | $0–3k (1) | $10k–50k | 3 | 1 | 4 | 0 |
| 42 | G15 | Flying-creature games | 3D | M | 2 | $0–3k (1) | $10k–100k | 3 | 1 | 4 | new |
| 43 | I2 | Relight your photo | — | M | 2 | $0–1k (1) | $5k–30k | 3 | 1 | 4 | 0 |
| 44 | I6 | Tracking-data replays | — | M | 2 | $2k–15k (2) | $20k–80k | 2 | 0 | 4 | 0 |
| 45 | R7 | Scientific visualization of simulation output | — | M | 3 | $0–5k (1) | $10k–50k | 1 | 0 | 3 | 0 |
| 46 | I5 | Sun, moon and shadow planner | — | L | 3 | $0–3k (1) | $20k–150k | 2 | 1 | 3 | 0 |
| 47 | E7 | Night-sky and astronomy explorer | — | M | 2 | $0–2k (1) | $5k–30k | 2 | 1 | 3 | 0 |
| 48 | E10 | Traffic and urban-planning classroom | NAV | M | 2 | $0–2k (1) | $5k–30k | 2 | 1 | 3 | new |
| 49 | R16 | Train on a server, play in the browser | SRV, EVO | M | 2 | ~$0 (1) | $10k–50k | 3 | 0 | 3 | new |
| 50 | I16 | Certified evacuation engineering | NAV + validation | XL | 4 | ~$0 (1) | $300k+ | 2 | 0 | 2 | new |
| 51 | A1 | Synthetic training data, including crowds | SKIN, NAV, SRV | L | 3 | $0–3k (1) | $20k–200k | 2 | 0 | 2 | +1 |
| 52 | A2 | On-device vision + 3D (try-on, avatars) | — | L | 3 | $0–3k (1) | $50k–500k | 2 | 0 | 2 | 0 |
| 53 | I8 | Smart-city and digital-twin crowd layers | NAV, SRV | L | 3 | ~$0 (1) | $50k–300k | 2 | 0 | 2 | 0 |
| 54 | I10 | Large-model viewers (CAD, BIM, scans) | — | L | 3 | $0–3k (1) | $20k–100k | 2 | 0 | 2 | 0 |
| 55 | I14 | Footfall and queues: shops, malls, airports, stations, hospitals | NAV | L | 3 | $0–5k (1) | $50k–300k | 2 | 0 | 2 | new |
| 56 | I15 | Warehouse and robot-fleet what-if simulations | NAV | L | 3 | $0–5k (1) | $50k–200k | 2 | 0 | 2 | new |
| 57 | R5 | Research-grade epidemiology | NAV, EVO | L | 3 | ~$0 (1) | $5k–40k | 2 | 0 | 2 | +1 |
| 58 | G13 | RTS with pathfinding on real maps | NAV, EYES | XL | 3 | $0–5k (1) | $50k–500k | 3 | 1 | 2 | new |
| 59 | I11 | Livestock and agritech | NAV, EVO | M | 2 | $0–2k (1) | $5k–30k | 2 | 0 | 2 | +2 |
| 60 | R13 | Movement ecology and landscapes | NAV, EVO | M | 2 | ~$0 (1) | $5k–30k | 2 | 0 | 2 | new |
| 61 | G8 | Swarm-programming game (Screeps-like) | EVO | L | 2 | $0–3k (1) | $20k–200k | 3 | 1 | 2 | 0 |
| 62 | G14 | Breed-your-swarm competitive game | EVO, SRV | L | 2 | $0–3k (1) | $20k–300k | 3 | 1 | 2 | new |
| 63 | A3 | A library of trained behaviours | EVO, SRV, BRAIN+ | M | 1 | $0–2k (1) | $5k–30k | 3 | 1 | 2 | new |
| 64 | I12 | Defence and wargaming swarms | — | XL | 4 | ~$0 (1) | not a solo path | 1 | 0 | 1 | 0 |
| 65 | G9 | Multiplayer .io game with bot swarms | SRV | XL | 3 | $0–3k (1) | $50k–1M | 2 | 1 | 1 | 0 |
| 66 | I9 | Product configurators | — | M | 2 | $0–3k (1) | $20k–80k | 1 | 0 | 1 | 0 |
| 67 | R15 | Soft-matter and granular research | PHYS | M | 2 | ~$0 (1) | $0–10k | 1 | 0 | 1 | new |
| 68 | G16 | Racing and traffic games with AI drivers | NAV, BRAIN+ | L | 2 | $0–3k (1) | $10k–100k | 2 | 1 | 1 | new |
| 69 | G17 | Sports tactics and team AI | BRAIN+ | L | 2 | $0–3k (1) | $10k–50k | 2 | 1 | 1 | new |
| 70 | R11 | Embodied-vision research | EYES | L | 2 | ~$0 (1) | $10k–50k | 3 | 0 | 1 | new |
| 71 | R6 | Multi-agent reinforcement-learning environments | EYES, BRAIN+ | L | 2 | ~$0 (1) | $0–10k | 2 | 0 | 0 | +1 |
| 72 | R14 | Cell and tissue biology | NAV, EVO, PHYS | L | 2 | ~$0 (1) | $5k–30k | 2 | 0 | 0 | new |
| 73 | C8 | Film and TV crowd previs | NAV, SKIN, PHYS | M | 1 | ~$0 (1) | $0–10k | 2 | 0 | 0 | +1 |
| 74 | I17 | Traffic microsimulation | NAV, BRAIN+ | XL | 3 | ~$0 (1) | $50k–300k | 1 | 0 | −1 | new |
| 75 | R9 | Agent-based economics and markets | — | L | 2 | ~$0 (1) | $0–5k | 1 | 0 | −1 | 0 |
| 76 | D5 | A general-purpose web 3D engine | — | XL | 2 | ~$0 (1) | low | 1 | 1 | −2 | 0 |
| — | I3 | Open-air festival crowd preview | | | | | | | | | becomes I13 |

### Quick views

- **Least effort with real need (S, Need ≥ 3):** E1 explainer series, C1 freelance experiences,
  E3 epidemic, E6 physics teaching, R1 collective motion.
- **Highest need (4):** E1, I13, C3, E8, G1, G2, I1, G11, I16 (certification, not a solo path),
  I12 (not a solo path).
- **Most likely money:** I13 and C1 ($10k–40k, tier 3). Eighteen use cases sit in tier 2, among
  them I1, E9, I4, E8, the explainer and video series (E1, C3) and the games (G1–G4, G10–G12).
- **Highest ceilings:** G1, G3, G11 ($100k–1M+), G2 ($100k–1M), I13, I1 and C3 ($100k–500k), I16
  ($300k+, with certification).

---

## 4. What Phase 2 changes for today's use cases

| ID | Use case | Δ | What changes |
|---|---|---|---|
| E1 | Explainer series | +4 | The story becomes "neural creatures evolving live", the most watched kind of simulation content: need 3 → 4, money 1 → 2 |
| C3 | Video content | +4 | The same draw: need 3 → 4, money 1 → 2 |
| E2 | Ecosystem classroom simulator | +3 | Population cycles emerge from energy, food, birth and death instead of scripted spawns: edge 2 → 3, money 1 → 2 |
| E6 | Physics and chemistry | +3 | Real collisions: gas kinetics, phase changes, granular flow. Need 2 → 3, edge 1 → 2 |
| R4 | Swarm robotics | +3 | Obstacles, aerial robots, ray range sensors: need 2 → 3, edge 2 → 3 |
| G2 | Horde game | +2 | Walls and base building, the core of the genre (They Are Billions-style): need 3 → 4 |
| R1 | Collective motion | +2 | Murmurations in 3D, obstacles, flocking by vision: need 2 → 3 |
| R2 | GPU modelling toolkit | +2 | Fields, a life cycle and obstacles cover most classic models: need 2 → 3 |
| I7 | Drone shows | +2 | Formation flight in 3D makes it a design tool, not only previs: need 2 → 3 |
| E5 | Classic-model library | +2 | Ants with pheromones, slime mould, chemotaxis: need 2 → 3 |
| D2 | Starter kits | +2 | More kinds to sell (evacuation, evolution, 3D flocks): money 1 → 2 |
| D3 | Swarm playground | +2 | Evolve creatures and share them: need 2 → 3 |
| I11 | Livestock and agritech | +2 | Terrain, fences, grazing energy: need 1 → 2 |
| C1 | Branded experiences | +1 | Murmurations in 3D and physics add spectacle: edge 2 → 3 |
| G4 | Idle swarm game | +1 | An evolving colony: edge 2 → 3 |
| C2 | Installations | +1 | Installations that evolve over an exhibition: edge 2 → 3 |
| E3 | Epidemic simulator | +1 | Rooms, buildings, births and deaths: edge 1 → 2 |
| I4 | Architecture visualization | +1 | People walking inside buildings: edge 2 → 3 |
| G6 | Aquarium | +1 | Schools in 3D, breeding: edge 2 → 3 |
| C4 | Generative art | +1 | 3D and evolved forms: edge 2 → 3 |
| C5 | Live visuals | +1 | Swarms in 3D: edge 2 → 3 |
| C7 | Data journalism | +1 | Crowd-crush and evacuation explainers: edge 1 → 2 |
| R5, R6, A1, C8 | Epidemiology, RL environments, synthetic data, film previs | +1 each | One more capability each (buildings; eyes and memory; rigged crowds in varied light; walls and people): edge 1 → 2 |
| I3 | Festival crowd preview | — | Superseded by I13, the venue planner |

Unchanged: G1 (Phase 2 deepens it, but the hook was already there), G3, G5, G7–G9, E4, E7, R3,
R7–R9, I1, I2, I5, I6, I8–I10, I12, A2, D1, D4, D5, C6.

---

## 5. New use cases, by domain

### 5.1 Games

| ID | Use case | What you would build | Needs | Who pays, and how | Still missing |
|---|---|---|---|---|---|
| G10 | Evolution sandbox game | Creatures with neural brains eat, breed, mutate and adapt under a real day and night; the player shapes the world and follows lineages | EVO, EYES, SKIN | Steam, Patreon. Comparable estimates: Equilinox ~$556k, Species ~$276k, The Bibites ~$44k gross | Balance over long runs; save files |
| G11 | Colony or god game with thousands of citizens | Citizens who need food and rest, work, are born and die | EVO, NAV, BRAIN+ (discrete actions) | Steam. The genre's hits sell in the millions, but on depth, not headcount | Economy, building, UI |
| G12 | Stealth game where guards really see | Guards with ray eyes; shadows, night and fog hide the player because they change what the eyes see | EYES, NAV | Steam | Level design, guard behaviour |
| G13 | RTS with pathfinding on real maps | Flow fields for whole armies, fog of war from what units see, formations | NAV, EYES | Steam | The rest of an RTS: economy, AI opponents, perhaps multiplayer |
| G14 | Breed-your-swarm competitive game | Players evolve swarms (locally or on a server) and pit them against others'; brains shared as files | EVO, SRV | Steam, cosmetics | Matchmaking, sharing, balance |
| G15 | Flying-creature games | Dragons, birds or fleets in 3D; murmurations as set pieces | 3D | Steam, itch.io | Flight controls and camera |
| G16 | Racing and traffic games with AI drivers | Lanes, junctions, traffic that reacts | NAV, BRAIN+ | Steam | Road networks, lane logic |
| G17 | Sports tactics and team AI | Teams driven by trained policies | BRAIN+ | Steam | The sport's rules, training data |

### 5.2 Education

| ID | Use case | What you would build | Needs | Who pays, and how | Still missing |
|---|---|---|---|---|---|
| E8 | Natural-selection and evolution lab | Camouflage that evolves under real light; mutation rate; drift against selection; speciation; teacher dashboards and lesson plans | EVO, EYES | Schools and districts (licences), Teachers Pay Teachers. Free rival: PhET's Natural Selection | Curriculum alignment, testing on school devices |
| E9 | Crowd-safety training for event staff | Short scenarios: how surges build, why bottlenecks are dangerous, where barriers help; a certificate on your platform | NAV, SKIN | Organizers training stewards, security firms; sold on your platform | Content written with a crowd-safety professional |
| E10 | Traffic and urban-planning classroom | Traffic jams, pedestrian flows and signal timing as experiments | NAV | Teachers, mostly free | Road layouts |

### 5.3 Research and science

| ID | Use case | What you would build | Needs | Who pays, and how | Still missing |
|---|---|---|---|---|---|
| R10 | Artificial-life and open-ended-evolution platform | 100k–1M neural genomes evolving on any GPU; bit-exact replays; long runs on a server, replayed in the browser | EVO, EYES, SRV | Grants and collaborations; the ALife community | Experiment tooling: seeds, lineages, metrics |
| R11 | Embodied-vision research | Agents that evolve or learn to see: eye design, camouflage, hide-and-seek under real lighting | EYES (rendered eyes) | Grants | Rendered eyes come after ray eyes |
| R12 | Pedestrian-dynamics research | Social-force models, bottleneck experiments, comparison with published crowd measurements | NAV | Grants, universities | Validation datasets |
| R13 | Movement ecology and landscapes | Animals moving over resource maps, terrain and corridors | NAV, EVO | Grants | Terrain and land-cover data |
| R14 | Cell and tissue biology | Chemotaxis on chemical fields, colony growth, division | NAV (field sensors), EVO, PHYS | Grants | Biology-specific models |
| R15 | Soft-matter and granular research | XPBD particles, cloth, granular flow | PHYS | — | Specialized solvers exist elsewhere |
| R16 | Train on a server, play in the browser | Evolve brains at 1M agents on a rented GPU; ship the best as `.onnx` files to browsers running the same engine | SRV, EVO | Grants; a showcase | A first run on a Linux GPU |

### 5.4 Events and industry

| ID | Use case | What you would build | Needs | Who pays, and how | Still missing |
|---|---|---|---|---|---|
| I13 | Venue crowd-flow and evacuation planner | Draw a venue (walls, gates, stages, bars, exits); load ticket counts and entry slots from your platform; run 1k–100k attendees; ingress queues, egress times, density heatmaps, bottleneck warnings; a report for the safety plan | NAV; SKIN for presentation (simple shapes suffice for analysis) | Organizers on your platform and small safety consultants. Suggested: $50–300 a month or $100–500 an event; commercial engineering tools cost $2,350 a year and up, and free simulators (JuPedSim, Vadere) are made for researchers | A venue editor; heatmap accumulation (app stages); validation against RiMEA's test cases before any safety claim |
| I14 | Footfall and queues: shops, malls, airports, stations, hospitals | Layouts, queues, flows, what-if changes | NAV | Operators and consultants, who use MassMotion, Legion or AnyLogic | Enterprise sales; floor-plan import |
| I15 | Warehouse and robot-fleet what-if simulations | Aisles, robots, throughput, congestion | NAV | Logistics companies | Robot models, data import |
| I16 | Certified evacuation engineering | Code-compliant analyses for buildings, stadiums and ships | NAV, plus a validation programme | Engineering firms, who buy Pathfinder-class tools | Validation, documentation, acceptance by authorities, liability cover |
| I17 | Traffic microsimulation | Lanes, junctions, signals | NAV, BRAIN+ | Cities and consultancies; SUMO is free | Road networks, a lane model |

### 5.5 AI, data and developer products

| ID | Use case | What you would build | Needs | Who pays, and how | Still missing |
|---|---|---|---|---|---|
| A3 | A library of trained behaviours | Evolved or trained `.onnx` behaviours for crowds, animals and enemies, sold as assets | EVO, SRV, BRAIN+ | Game developers | Proof that anyone buys behaviours |
| D6 | GPU physics kit for web developers | Cloth, particles and a broadphase from the physics plugin, as a drop-in | PHYS | Web developers | Docs, examples |

---

## 6. Which Phase 2 piece pays most

| Piece | Best use cases it opens or lifts (score after Phase 2) | Money it opens (likely → ceiling) | Verdict |
|---|---|---|---|
| **NAV** | I13 venue planner 11 · G2 horde with walls 10 · E9 staff training 8 · I4 architecture indoors 7 · E3 epidemic 7 · R12 pedestrian research 4 · I14–I16 2 | I13 alone $10k–40k → $100k–500k, on your own channel; E9 adds $2k–15k | **First** |
| **EVO** | E1 explainer series 12 · C3 video series 10 · E8 evolution lab 10 · E2 ecosystem 8 · G4 idle 8 · C2 living installations 8 · G10 sandbox game 6 | E8, C3 and G10 together $3k–25k → $100k–500k; mostly audience at first | **Second** |
| **EYES** (rays) | E8 camouflage (with EVO) · G12 stealth 6 · R4 range sensors 5 · R1 vision-based flocking 7 | A stealth game; research novelty | **With EVO** (ray eyes); rendered eyes much later |
| **3D** | C1 brand work 11 · R1 murmurations 7 · G6 aquarium 6 · I7 drone-show design 5 · G15 flying games 4 | Freelance (C1); a niche tool (I7) | Third |
| **BRAIN+** | D4 NPC crowds 6 · G11 colony game 5 · R3 4 | Indirect; colony games need discrete actions | When a chosen game needs it |
| **SRV** | R10 artificial-life platform 5 · R16 3 · G14 2 | Grants | Later |
| **PHYS** | E6 physics teaching 7 · G3 battle debris 6 · D6 physics kit 4 · R15 1 | Small | **Last** |

**Read as an order:** NAV → EVO with ray eyes → 3D → BRAIN+ as games need it → SRV → PHYS.
`agents-design.md` §13 lists evolution, eyes, then walls; the money points the other way, because
walls serve the one product with its own channel. Deciding the order is the user's call.

**On rigged models (SKIN):** the use cases that earn need rigged *people* more than rigged
animals: I4's walkthroughs and E9's training. I13 can start with simple shapes.

---

## 7. Still out of reach after Phase 2

- **Agents driven by language models**, such as a self-narrating simulation: `OrtModel` runs float32
  models only, so token inputs need another runtime.
- **Certified life-safety analysis:** it needs validation, documentation and liability cover, not
  more code (I16).
- **Audiences without WebGPU,** about a third of Poki's players in June 2026: by design, no WebGL
  path.
- **Determinism across GPUs,** for lockstep multiplayer between machines: lockstep is bit-exact on
  one device only.
- **Photoreal offline rendering and ray tracing:** WebGPU has no ray-tracing API.
- **Games built on stacked rigid bodies and joints:** Rapier or Jolt through P1's bridge; GPU rigid
  contacts (P6) are not planned.
- **Learning by gradient in the browser** (reinforcement learning, backpropagation): a non-goal;
  brains are trained offline or evolved.
- **Non-spatial models** (markets, social networks): the grid is spatial.

---

## 8. Top 3 recommendations after Phase 2, and why

### 1. I13 — a venue crowd-flow and evacuation planner on your platform (score 11)

- **What Phase 2 adds makes it possible:** walls, obstacles and flow fields turn agents into
  pedestrians that route through a venue: ingress queues, egress times, density heatmaps,
  bottleneck warnings.
- **The need is real and recurring.** Large events need a crowd-management plan, and the
  commercial tools are built and priced for engineers: Pathfinder $2,350 a year a seat ($7,850
  perpetual), MassMotion historically £5k–£20k a seat, AnyLogic ~$19k (unverified). Free
  simulators exist (JuPedSim, Vadere), but they are made for researchers, not organizers.
- **Your platform holds the data that makes it specific:** ticket counts, entry slots and gates. A
  planner fed by real sales is something no standalone tool has.
- **Nobody serves the small end:** a browser tool on any laptop GPU can sell at $50–300 a month or
  $100–500 an event (a suggestion, to be tested with your organizers).
- **Money:** likely $10k–40k in year 1 (for example 30 organizers × $600 a year ≈ $18k, plus
  per-event plans); $100k–500k if it grows beyond your platform. E9's staff training sells on the
  same engine and channel.
- **Caveat:** present it as a planning and communication aid, not a certified evacuation analysis,
  in the product and its terms, until it passes the RiMEA guideline's test cases (§9).
- **First step, a spike:** a venue with walls and two exits, 10k agents routed by a flow field, and
  the egress time compared with a hand calculation.
- **For Phase 2:** build NAV first.

### 2. One evolution engine, three outlets: the E8 lab, the C3/E1 series, the G10 game (scores 10, 10–12, 6)

- **One build, three uses:** a natural-selection lab for classrooms (evolution is core biology), an
  evolution video and explainer series (evolving neural creatures is among the most watched
  simulation content), and a sandbox game (Equilinox ~$556k, Species ~$276k, The Bibites ~$44k
  gross; estimates).
- **NullGraph's edge is specific:** a genome per agent as a weight-table row, which the record's
  reserved `genome` field was designed for; bit-exact replays, so a class or a viewer can rerun the
  same run; a physical day and night that change what ray eyes see, so camouflage evolves for real.
- **Money:** likely $3k–25k across the three in year 1; $100k–500k ceiling from the game, licences
  and the channel.
- **For Phase 2:** EVO second, with ray eyes (EYES' first step) for camouflage.

### 3. G1 and G2 — the horde game, upgraded (score 10)

- **It compounds.** The game recommended now gains base building and walls from NAV, and enemies
  that evolve against the player's build from EVO: an update or a sequel at M effort, sold to the
  players the first game found.
- **Money:** likely $1k–15k a release; $100k–1M if the series breaks out.

**Alongside them:** freelance brand work (C1, score 11) still pays best per hour, and flocks in 3D
make it more distinctive.

---

## 9. Sources

Price and market comparables are listed in the
[current-state file, §9](./use-cases-current.md#9-sources). Added for this file, read 2026-10-16:

- Artificial-life games, sales estimates: [Equilinox (SteamPulse)](https://steampulse.org/game/853550)
  ~74.7k units, ~$556k · [Species: Artificial Life, Real Evolution (SteamPulse)](https://steampulse.org/game/774541)
  ~18.1k units, ~$276k · [The Bibites (Raijin)](https://raijin.gg/app/2736860/The_Bibites_Digital_Life)
  ~5.5k copies, ~$44k.
- Evacuation-model validation: [Guideline for Crowd Evacuation Simulation — validation with RiMEA
  test scenarios (SciTePress, 2015)](https://scitepress.org/Documents/2015/54367): a simulator
  checked against the guideline's 14 test cases · [RiMEA test cases (PedGo)](https://traffgo-ht.com/en/pedestrians/validation/rimea/index.html) ·
  [test cases (Pedestrian and Evacuation Dynamics wiki)](https://ped.fandom.com/wiki/Test_Cases).
- The repository: [`agents-design.md`](./agents-design.md) §4.1 and §13, [`work/plan.md`](./work/plan.md)
  (scope changes, 2026-10-15), [`physics/README.md`](./physics/README.md), [`handoff.md`](./handoff.md)
  ("Agents that see", "Far horizon").
