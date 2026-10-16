# NullGraph — every use case, rated (current state)

**What this is:** every use found for NullGraph as published today (npm `null-graph` 1.1.1,
2026-10-15), rated by effort to build, market need and money, then ranked. Written 2026-10-16 for
the author, a solo developer. The same list after Phase 2 is in
[`use-cases-after-phase-2.md`](./use-cases-after-phase-2.md).

**Not this file:** a plan. Nothing here changes [`work/plan.md`](./work/plan.md); scope stays the
user's decision.

**How far to trust the money:** estimates for one person, gross, before store cuts and tax. They
lean on public price lists and third-party sales estimates (§9), many of them secondhand. Use them
to compare options, not to forecast income.

---

## Summary

**Top 3 now** (reasons in §8):

1. **G1 — a survivors-like horde game:** more than ten thousand enemies chasing the player. The
   swarm *is* the gameplay, Steam and itch.io reach players directly, and it is 1–2 months on top
   of Herd & Wolves' code. Likely $1k–15k in year 1; a hit reaches six figures.
2. **I1 — the event photo booth on your own event platform:** you already reach the buyers, booths
   are a routine event expense, and real 3D lighting with on-device privacy sets it apart. Likely
   $2k–25k, depending on your platform's volume.
3. **E1 + C3 — "a neural network learned to herd," as an interactive page and short videos:**
   almost built already, and it builds the audience the other two need. Little money directly.

**Cash floor:** freelance WebGPU work (C1, §6) pays the most per hour today, but it sells your time
rather than the library.

**After Phase 2** the best use case becomes a **venue crowd-flow and evacuation planner** on your
platform, which argues for building walls, obstacles and flow fields first
([the other file](./use-cases-after-phase-2.md)).

---

## 1. What the library can do today

Every rating rests on this list. Measured numbers come from the design docs.

**Can:**

- **Simulate agents on the GPU:** sense → think → act every fixed step, with no CPU work per agent.
  Brains are WGSL rules, small dense networks (`MlpBrain`: up to 4 layers, 128 inputs, 64 hidden,
  16 outputs), or any float32 model through ONNX Runtime (`OrtBrain`). Up to 16 species, each with
  its own brain, switched live without a hitch ([agents guide](./api/agents.md)).
- **Scale:** 60 fps from 4K to 16K agents with network brains on an AMD integrated GPU (step 10).
  The demo builds up to 1M agents where the GPU allows. Discrete GPUs are faster but not yet
  measured: step 5's scaling study is deferred.
- **Train and load brains:** the repository's tools record a rule, train a network that copies it
  and export it (about an hour of CPU); `mlpFromOnnx` loads small networks trained in PyTorch or
  scikit-learn ([inference guide](./api/inference.md)).
- **Replay bit-exactly** on one device (lockstep), so experiments are reproducible.
- **Explain:** click an agent and see what it senses and decides (picking, inspector); labels
  follow agents with no CPU work.
- **Draw agents as animated models:** baked rigid or skinned animation, one palette slot per visible
  model, LOD by size on screen, toon and anime looks.
- **Render:** PBR, cascaded and punctual shadows, IBL and probes, physical atmosphere and fog, time
  of day with sun and moon, a night sky with ~9,100 catalogue stars, moon phase and Milky Way;
  16 post effects; GPU culling, meshlets, Hi-Z; SDF/MSDF text; orbit, fly, follow and path cameras.
- **Run anywhere WebGPU runs:** main thread, a worker, or headless in Node through Dawn (render to
  PNG, simulate on a server).
- **Run other ONNX models** (float32) over the same GPU buffers: the base for segmentation and
  depth.

**Cannot yet:**

- Agents move on a plane (XZ); nothing flies.
- No walls, obstacles, pathfinding or flow fields, only world bounds. An app step stage can push
  agents out of shapes, but agents cannot sense them.
- Custom sensors cannot read app buffers (maps, fields, pheromones) or the neighbour lists.
- Births and deaths are CPU spawns and kills, or an agent resetting itself (the demo's respawn); no
  reproduction or evolution on the GPU.
- No physics (rigid bodies, collisions, cloth), scene graph or editor.
- Neighbour queries dominate at scale: 74 ms a step at 64K agents on the AMD iGPU (measured before
  step 5, which moved it by −15% to +9.6%).
- ONNX Runtime costs a 6.4 MB download, up to 1.6 s to start and 3–10× the built-in kernel's GPU
  time; float32 only, so no language models.
- **WebGPU only, by design.** About 68% of Poki's players had WebGPU in June 2026 (§9). Steam on
  desktop, kiosks, iPads on iPadOS 26 and business laptops reach more of their audience than the
  open web does.

**Traction, 2026-10-16:** 154 npm downloads in the month to 2026-10-12, 2,423 in the past year;
102 GitHub stars, 10 forks. Anything that needs an existing audience starts from here.

---

## 2. How to read the ratings

| Column | Meaning |
|---|---|
| **Effort** | To build the use case on the library as it is today, solo with coding assistants, full-time. **XS** under 1 week · **S** 1–3 weeks · **M** 1–2 months · **L** 2–4 months · **XL** 4+ months, or work one person cannot do (a sales team, certification, a content team) |
| **Need** | Market relevance, 1–5. 1 curiosity · 2 enthusiasts or researchers, little money · 3 a real niche with paying buyers, served by existing tools · 4 clear paying demand where NullGraph's angle helps · 5 urgent demand, buyers searching |
| **Likely /yr** | Gross revenue in the first 12 months after launch, for you, with today's audience. The number in brackets is its money tier: 1 under $2k · 2 $2k–10k · 3 $10k–40k · 4 $40k–150k · 5 over $150k, taken from the typical outcome in the range |
| **Ceiling /yr** | Yearly revenue in year 2–3 if it finds its market, bounded by comparables |
| **Edge** | Why NullGraph: 1 any engine does it · 2 NullGraph is clearly better (scale, browser, privacy, determinism, physical sky) · 3 hard to do any other way in a browser |
| **Reach** | Access to buyers: 0 cold sales or academia · 1 a self-serve store or audience platform (Steam, itch.io, portals, YouTube, Teachers Pay Teachers, npm) · 2 your own channel (your event-booking platform) |
| **Score** | 2·Need + 2·Money tier + Edge + Reach − 2·Effort (XS = 1 … XL = 5). A sorting aid, not a verdict. Ties: higher Need first, then lower Effort |

---

## 3. All use cases, ranked

| # | ID | Use case | Domain | Effort | Need | Likely /yr | Ceiling /yr | Edge | Reach | Score |
|---:|---|---|---|---|---:|---|---|---:|---:|---:|
| 1 | G1 | Survivors-like horde game | Games | M | 4 | $1k–15k (2) | $100k–1M+ | 3 | 1 | 10 |
| 2 | C1 | Branded interactive web experiences (freelance) | Creative | S/project | 3 | $10k–40k (3) | $80k–200k | 2 | 0 | 10 |
| 3 | I1 | Event photo booth on your platform | Events | L | 4 | $2k–25k (2) | $100k–500k | 2 | 2 | 8 |
| 4 | E1 | Interactive explainer: a network learned to herd | Education | S | 3 | $0–2k (1) | $10k–80k | 3 | 1 | 8 |
| 5 | G2 | Horde tower defence on open maps | Games | M | 3 | $0.5k–10k (2) | $50k–500k | 3 | 1 | 8 |
| 6 | C2 | Commissioned installations (museums, brands, events) | Creative | M | 3 | $0–25k (2) | $50k–200k | 2 | 1 | 7 |
| 7 | E4 | Learn-WebGPU course built on NullGraph | Education | M | 3 | $1k–15k (2) | $50k–500k | 2 | 1 | 7 |
| 8 | G4 | Idle/incremental swarm game | Games | M | 3 | $0.5k–8k (2) | $20k–150k | 2 | 1 | 7 |
| 9 | E3 | Epidemic-spread simulator | Education | S | 3 | $0–3k (1) | $10k–60k | 1 | 1 | 6 |
| 10 | C3 | Science and AI video content | Creative | M | 3 | $0–3k (1) | $50k–500k | 3 | 1 | 6 |
| 11 | I4 | Architecture/real-estate visualization with life | Industry | M | 3 | $3k–20k (2) | $40k–120k | 2 | 0 | 6 |
| 12 | G3 | Massive-battle sandbox (land or space) | Games | L | 3 | $1k–15k (2) | $100k–1M+ | 3 | 1 | 6 |
| 13 | D1 | GPU benchmark site | Developer | S | 2 | $0–1k (1) | $5k–20k | 3 | 1 | 6 |
| 14 | D2 | Paid starter kits and templates | Developer | S | 2 | $0.5k–5k (1) | $10k–50k | 3 | 1 | 6 |
| 15 | D4 | Neural NPC crowds for game studios | Developer | S | 2 | $0–3k (1) | $5k–40k | 3 | 1 | 6 |
| 16 | E2 | Ecosystem / predator–prey classroom simulator | Education | M | 3 | $0–3k (1) | $20k–150k | 2 | 1 | 5 |
| 17 | C4 | Generative art | Creative | S | 2 | $0–3k (1) | $5k–50k | 2 | 1 | 5 |
| 18 | G5 | Herding/sheepdog game | Games | S | 2 | $0–3k (1) | $10k–100k | 2 | 1 | 5 |
| 19 | G6 | Ambient aquarium, desktop toy, live wallpaper | Games | S | 2 | $0–3k (1) | $5k–30k | 2 | 1 | 5 |
| 20 | R1 | Collective motion and active matter | Research | S | 2 | $0–2k (1) | $10k–50k | 3 | 0 | 5 |
| 21 | I3 | Open-air festival crowd preview | Events | M | 2 | $0–3k (1) | $10k–50k | 3 | 2 | 5 |
| 22 | C6 | Offline-rendered loops and stock footage | Creative | S | 2 | $0–1k (1) | $2k–20k | 1 | 1 | 4 |
| 23 | E6 | Particle physics and chemistry, no collisions | Education | S | 2 | $0–1k (1) | $5k–30k | 1 | 1 | 4 |
| 24 | R3 | Imitation-learning testbed | Research | S | 2 | ~$0 (1) | $0–10k | 2 | 0 | 4 |
| 25 | R8 | Math and science-communication visuals | Research | S | 2 | ~$0 (1) | $2k–20k | 1 | 1 | 4 |
| 26 | D3 | Swarm playground ("Shadertoy for agents") | Developer | M | 2 | $0–1k (1) | $5k–50k | 3 | 1 | 4 |
| 27 | G7 | Stream-interactive swarm battles | Games | M | 2 | $0–3k (1) | $10k–50k | 3 | 1 | 4 |
| 28 | I2 | Relight your photo | Events | M | 2 | $0–1k (1) | $5k–30k | 3 | 1 | 4 |
| 29 | I6 | Tracking-data replays | Industry | M | 2 | $2k–15k (2) | $20k–80k | 2 | 0 | 4 |
| 30 | R7 | Scientific visualization of simulation output | Research | M | 3 | $0–5k (1) | $10k–50k | 1 | 0 | 3 |
| 31 | I5 | Sun, moon and shadow planner | Industry | L | 3 | $0–3k (1) | $20k–150k | 2 | 1 | 3 |
| 32 | C7 | Data-journalism interactives | Creative | S | 2 | $0–3k (1) | $5k–30k | 1 | 0 | 3 |
| 33 | C5 | Live visuals (VJ) | Creative | M | 2 | $0–2k (1) | $5k–30k | 2 | 1 | 3 |
| 34 | E7 | Night-sky and astronomy explorer | Education | M | 2 | $0–2k (1) | $5k–30k | 2 | 1 | 3 |
| 35 | I7 | Drone-light-show previs | Industry | M | 2 | $0–3k (1) | $20k–100k | 3 | 0 | 3 |
| 36 | R2 | GPU agent-based-modelling toolkit for any GPU | Research | M | 2 | $0–2k (1) | $10k–60k | 3 | 0 | 3 |
| 37 | A2 | On-device vision + 3D (try-on, avatars) | AI & data | L | 3 | $0–3k (1) | $50k–500k | 2 | 0 | 2 |
| 38 | I8 | Smart-city and digital-twin crowd layers | Industry | L | 3 | ~$0 (1) | $50k–300k | 2 | 0 | 2 |
| 39 | I10 | Large-model viewers (CAD, BIM, scans) | Industry | L | 3 | $0–3k (1) | $20k–100k | 2 | 0 | 2 |
| 40 | E5 | Classic-model library at GPU scale | Education | M | 2 | $0–2k (1) | $10k–50k | 2 | 0 | 2 |
| 41 | R4 | Swarm-robotics prototyping (ground robots) | Research | M | 2 | $0–3k (1) | $5k–30k | 2 | 0 | 2 |
| 42 | G8 | Swarm-programming game (Screeps-like) | Games | L | 2 | $0–3k (1) | $20k–200k | 3 | 1 | 2 |
| 43 | I12 | Defence and wargaming swarms | Industry | XL | 4 | ~$0 (1) | not a solo path | 1 | 0 | 1 |
| 44 | A1 | Synthetic training data | AI & data | L | 3 | $0–3k (1) | $20k–200k | 1 | 0 | 1 |
| 45 | R5 | Research-grade epidemiology | Research | L | 3 | ~$0 (1) | $5k–30k | 1 | 0 | 1 |
| 46 | G9 | Multiplayer .io game with bot swarms | Games | XL | 3 | $0–3k (1) | $50k–1M | 2 | 1 | 1 |
| 47 | I9 | Product configurators | Industry | M | 2 | $0–3k (1) | $20k–80k | 1 | 0 | 1 |
| 48 | I11 | Livestock and agritech visuals | Industry | M | 1 | ~$0 (1) | $5k–20k | 2 | 0 | 0 |
| 49 | R6 | Multi-agent reinforcement-learning environments | Research | L | 2 | ~$0 (1) | $0–5k | 1 | 0 | −1 |
| 50 | R9 | Agent-based economics and markets | Research | L | 2 | ~$0 (1) | $0–5k | 1 | 0 | −1 |
| 51 | C8 | Film and TV crowd previs | Creative | M | 1 | ~$0 (1) | $0–10k | 1 | 0 | −1 |
| 52 | D5 | A general-purpose web 3D engine | Developer | XL | 2 | ~$0 (1) | low | 1 | 1 | −2 |

### Quick views

- **Least effort with real need (S, Need ≥ 3):** E1 explainer, E3 epidemic simulator, C1 freelance
  experiences.
- **Highest need (4):** G1 horde game, I1 photo booth, I12 defence (not a solo path).
- **Most likely money:** C1 freelance ($10k–40k); then tier 2: G1, G2, G3, G4, E4, I1, I4, I6, C2.
- **Highest ceilings:** G1 and G3 ($100k–1M+), I1 ($100k–500k), G2, E4, C3 and A2 ($50k–500k),
  G9 ($50k–1M, at XL effort).

---

## 4. Use cases by domain

"Library pieces" names the entries each one uses; "Missing today" is what the app or the library
would still need. Phase 2 tags (NAV, EVO, EYES, 3D, BRAIN+, SRV, PHYS) are defined in the
[other file](./use-cases-after-phase-2.md#1-what-phase-2-adds).

### 4.1 Games

| ID | Use case | What you would build | Library pieces | Who pays, and how | Missing today |
|---|---|---|---|---|---|
| G1 | Survivors-like horde game | A top-down arena where thousands to tens of thousands of enemies chase and surround the player; weapons thin the swarm; upgrades between waves | `agents` (a rule brain to chase and separate; an app step stage for damage, death and respawn), LOD in `visibility`, loaders palettes for animated enemies, toon or anime `post` | Players: Steam at $3–10 through a Chromium wrapper such as Electron, itch.io. Web portals reach only their WebGPU share | Arena obstacles (only a push-out in an app stage); enemy counts above 16K unmeasured on most GPUs; art, audio, UI |
| G2 | Horde tower defence on open maps | Waves of thousands march on a base; towers and traps thin them | As G1, plus `picking` to place and select towers | Same stores as G1 | Walls and base building (NAV) |
| G3 | Massive-battle sandbox (land or space) | Place two armies of 10k–100k+ units and watch them clash; space fleets suit a plane | `agents` (a brain per species), animated models, LOD, `cameras`, `text` | Steam at $10–15; Ultimate Epic Battle Simulator 2 sold ~165k copies (est.) | Terrain and obstacles, flight, line of sight |
| G4 | Idle/incremental swarm game | Grow an ant, bee or robot swarm from ten to a million | `agents`, instancing, `text` | Portals (ad revenue split with the portal), Steam | Births are CPU spawns, which is enough at idle pace |
| G5 | Herding/sheepdog game | You are the dog; the sheep flock for real; pen them | Herd & Wolves' simulation as it is | Portals, itch.io | Pens and fences (NAV); a push-out stage can fake them |
| G6 | Ambient aquarium, desktop toy, live wallpaper | Calm schools of fish or flocks of birds; click one to follow it | `agents`, `picking`, `post`, the follow camera | itch.io at $2–5, wallpaper apps | Fish swim on a plane until 3D |
| G7 | Stream-interactive swarm battles | Twitch chat spawns and steers armies live on stream | `agents`, plus a chat bridge in the app | Streamers' tips, sponsorships | The chat integration |
| G8 | Swarm-programming game (Screeps-like) | Players script behaviours (blocks compiled to WGSL rules) and pit their swarms against others | `RuleBrain`, `prepareBrain`, the GPU timeout guard | Steam | A safe sandbox for player code on the GPU; an editor |
| G9 | Multiplayer .io game with bot swarms | A Node server (Dawn) simulates thousands of bots and streams them to players | Headless Node, `agents` | Ads, cosmetics | Networking and servers; Linux GPU runs unverified; determinism holds on one device only |

### 4.2 Education

| ID | Use case | What you would build | Library pieces | Who pays, and how | Missing today |
|---|---|---|---|---|---|
| E1 | Interactive explainer: "a neural network learned to herd" | A page and a short video: the rule, the network trained to copy it, and the same network on ONNX Runtime, switched live with their costs; click an agent to see its inputs and outputs | The Herd & Wolves demo as built in steps 7–10, the inspector | Indirectly: an audience for the course (E4), sponsors, consulting leads | Nothing technical: writing and filming |
| E2 | Ecosystem / predator–prey classroom simulator | Sliders for speeds and behaviours, a live population graph, a teacher guide and lesson plan | `agents`, inspector, `picking`, `text` | Teachers (lesson packs on Teachers Pay Teachers), schools and districts (licences). Free rivals: NetLogo's Wolf Sheep Predation, PhET | A population counter with a small readback (app); births are CPU spawns; testing on school Chromebooks |
| E3 | Epidemic-spread simulator | SIR/SEIR spread by proximity; distancing and vaccination sliders | `agents` (flag bits), app step stages | Teachers, public-health communication, newsrooms | Rooms and buildings (NAV) |
| E4 | Learn-WebGPU course built on NullGraph | 10–20 lessons: buffers, compute, GPU agents, culling, a small network on the GPU | The whole library, [`agent/guide.md`](./agent/guide.md) | Developers at $45–95 a course (Three.js Journey's prices) | Course production and marketing |
| E5 | Classic-model library at GPU scale | Schelling segregation, flocking, opinion dynamics, the traffic ring, diffusion: one URL each | `agents`, `RuleBrain` | Mostly free; small licences for courses | Fields and pheromones need sensors that read app buffers (NAV) |
| E6 | Particle physics and chemistry, no collisions | Diffusion, Brownian motion, gas in a box | `agents` as particles, app stages | Teachers; PhET is free | Particle–particle collisions (PHYS) |
| E7 | Night-sky and astronomy explorer | The real star catalogue (~9,100 stars), moon phase, Milky Way, any latitude and date | `environment` (`NightSky`, `StarField`, `TimeOfDay`) | Hobbyists and teachers; Stellarium is free | Constellation data, UI |

### 4.3 Research and science

| ID | Use case | What you would build | Library pieces | Who pays, and how | Missing today |
|---|---|---|---|---|---|
| R1 | Collective motion and active matter | Flocking, schooling, herding and self-propelled-particle models at 100k+, with bit-exact reruns | `agents`, lockstep `step()` | Grants and collaborations rather than sales | 3D, obstacles, vision (Phase 2) |
| R2 | GPU agent-based-modelling toolkit for any GPU | Parameter sweeps, a headless batch runner, aggregate export, a shareable URL: FLAME GPU's niche without CUDA | `agents`, headless Node, determinism | Grants (NLnet: €5k–50k a project), citations | Sweep and export tooling; app-buffer sensors (NAV) |
| R3 | Imitation-learning testbed | Record a rule, train a copy, deploy it to a population, measure it | [`tools/agents`](../tools/agents/README.md) (record, train, evaluate, export) | Academic | A Python bridge, optional |
| R4 | Swarm-robotics prototyping (ground robots) | Dispersion, coverage and aggregation in 2D | `agents`, `RuleBrain` | Robotics labs (consulting) | Obstacles and range sensors (NAV, EYES) |
| R5 | Research-grade epidemiology | Households, mobility, contact networks | `agents` | Public health, which has tools (Covasim, OpenABM) | Non-spatial contact networks; buildings |
| R6 | Multi-agent reinforcement-learning environments | Batched environments for training | `agents`, headless | — | Training in the loop, Python bindings; JAX- and Isaac-based tools dominate |
| R7 | Scientific visualization of simulation output | Millions of particles from molecular-dynamics or N-body runs, with picking and labels | `visibility`, `picking`, `text`, instancing | Labs (consulting) | File formats, streaming |
| R8 | Math and science-communication visuals | Attractors, reaction–diffusion, fibrations (already on nullgraph.dev) | Core compute and render | Indirect | — |
| R9 | Agent-based economics and markets | Traders and firms as agents | `agents`: a poor fit, since the grid is spatial | Academic | A non-spatial interaction model |

### 4.4 Events and industry

| ID | Use case | What you would build | Library pieces | Who pays, and how | Missing today |
|---|---|---|---|---|---|
| I1 | Event photo booth on your platform | Camera → person segmentation (ONNX Runtime) → the attendee inside a branded 3D set, lit and shadowed, in front of and behind props → a photo or 5 s clip. Nothing leaves the device | `inference` (`OrtRuntime`, `OrtModel`), `materials`, `lights`, `shadows`, `environment` (IBL), `post` | Organizers, per event or for premium sets. Booth rentals bill $200–700 an hour; booth software sells for $49–249 a month | Camera-to-tensor and mask compositing (app compute stages); a small, permissively licensed float32 segmentation model; ONNX Runtime's WebGPU on iPad Safari unverified |
| I2 | Relight your photo | A depth model turns a photo into a surface lit by real lights, shadows and post; works offline | `inference`, `lights`, `shadows`, `post` | Indirectly: a showcase for I1 | Depth models are large (Depth Anything V2 Small, Apache-2.0) |
| I3 | Open-air festival crowd preview | An organizer drops stages, bars and gates on a field; 10k–50k attendees drift between them | `agents` (attractors as rule constants, recompiled off the frame with `prepareBrain`) | A platform feature, for retention | Walls, fences and exits (NAV): no indoor venues |
| I4 | Architecture/real-estate visualization with life | Web walkthroughs with sun and time of day; animated people in plazas and parks | `environment`, `shadows`, IBL, loaders palettes for crowds, `cameras` | Developers and archviz studios, per project ($1.5k–12k solo, $15k–50k at studios) | Rigged human models; walking inside buildings (NAV) |
| I5 | Sun, moon and shadow planner | Sun and moon for any date and latitude; cascaded shadows on buildings and terrain | `environment` (`TimeOfDay`, atmosphere), `shadows` | Architects, home buyers, photographers (subscriptions) | Building and terrain data (OpenStreetMap, elevation tiles) |
| I6 | Tracking-data replays | 10k+ moving robots, vehicles or players replayed, with labels and picking | Instancing, `visibility`, `picking`, `text`, `cameras` | Logistics and sports companies (consulting) | Data connectors |
| I7 | Drone-light-show previs | Load a show; thousands of lights over a real night sky with moon phase and bloom | `environment` (night sky), instancing, `post` | Show operators: SPH Engineering's software costs $28,000; Skybrush has a free tier | Show-file import; formation flight (3D) |
| I8 | Smart-city and digital-twin crowd layers | Crowds in open plazas over city models | `agents`, rendering | Cities and integrators (enterprise) | Streets and buildings as obstacles (NAV); enterprise sales |
| I9 | Product configurators | PBR and IBL product viewers | `materials`, `environment`, `post` | E-commerce | WebGPU-only reach rules it out for shops; no KTX2 or Draco yet |
| I10 | Large-model viewers (CAD, BIM, scans) | GPU culling, meshlets and Hi-Z for very large models | `geometry` (meshlets), `visibility` | Architecture and engineering firms (consulting) | IFC and CAD loaders, streaming |
| I11 | Livestock and agritech visuals | Grazing herds, virtual-fence behaviour | `agents` (herding) | Agritech marketing | Terrain and fences (NAV) |
| I12 | Defence and wargaming swarms | Large unit simulations | `agents` | Governments | Procurement and clearance: not a solo path |

### 4.5 Creative and media

| ID | Use case | What you would build | Library pieces | Who pays, and how | Missing today |
|---|---|---|---|---|---|
| C1 | Branded interactive web experiences | Campaign microsites where 10k–1M creatures or particles react to the cursor and the product, with a video for visitors without WebGPU | `agents`, `post`, `environment`, `text` | Agencies and brands, per project | Portfolio pieces and outreach |
| C2 | Commissioned installations | Touch tables and projections for museums, science centres, brand activations and your organizers' stages: "be the wolf", fish that scatter from visitors, historical scenes with crowds | `agents`, `picking`, `post`; Chrome in kiosk mode | Museums, science centres, brands, organizers; studios charge $15k–45k a station | Camera or sensor input (app); hardware |
| C3 | Science and AI video content | Videos and shorts: "a million agents in a browser", "a network learned to herd" | Headless render (`npm run render`), the demos | YouTube ads, sponsors, Patreon | Filming and editing time. D2, offline films ([design](./offline-films-design.md), 2026-10-16, not built), would render a minute of 1M agents at 60 fps |
| C4 | Generative art | Editions and prints from flocks and flow fields | `agents`, `post`, headless render | Collectors, print buyers | — |
| C5 | Live visuals (VJ) | Audio-reactive swarms in a browser | `agents`, `post` | Gigs; TouchDesigner ($600–2,200) is the incumbent | Audio and MIDI input (app) |
| C6 | Offline-rendered loops and stock footage | Render frames headless, encode them to video | Headless render | Stock-footage sites | The frame-and-ffmpeg pipeline that D2 designs |
| C7 | Data-journalism interactives | Simulations in the style of the Washington Post's 2020 "flatten the curve" piece | `agents` | Newsrooms | Readers without WebGPU |
| C8 | Film and TV crowd previs | Browser previs of crowd shots | Animated crowds | Studios | Incumbents: Golaem in Autodesk's bundle, Unreal for free |

### 4.6 AI and data

| ID | Use case | What you would build | Library pieces | Who pays, and how | Missing today |
|---|---|---|---|---|---|
| A1 | Synthetic training data | Headless renders with random sky, time, fog and materials, with exact masks and depth | Headless render, `environment`, `post` | ML teams; the market was ~$0.6–0.8B in 2025 | Label passes, a randomizer, a dataset writer; not photoreal; Linux GPU runs unverified |
| A2 | On-device vision + 3D (try-on, avatars, fitness) | Pose or segmentation driving 3D, privately | `inference`, rendering | Retail, fitness | Market access (parked in the product plan); models |

### 4.7 Developer products

| ID | Use case | What you would build | Library pieces | Who pays, and how | Missing today |
|---|---|---|---|---|---|
| D1 | GPU benchmark site | "How many agents can your GPU run?" with shareable results | The demo's stress test and its guard (built) | Ads, sponsors; mainly a funnel | Packaging, a results board |
| D2 | Paid starter kits and templates | Horde, army, flock and animated-crowd templates | `agents`, `loaders`, `visibility` | Web game developers at $20–60 (Unity's crowd assets sell for $30–60) | Packaging and docs |
| D3 | Swarm playground ("Shadertoy for agents") | Write rules in the browser, run them, share a URL, fork | `RuleBrain`, `prepareBrain`, the GPU guard | Donations, a pro tier | An editor, hosting, protection from shaders that hang |
| D4 | Neural NPC crowds for game studios | Train in PyTorch → `.onnx` → `MlpBrain`, with a worked example | `inference` (`mlpFromOnnx`), `MlpBrain` | Studios (support, consulting) | The example and its guide |
| D5 | A general-purpose web 3D engine | Competing with three.js, Babylon.js and PlayCanvas for ordinary games and sites | Everything | — | An editor, an ecosystem, WebGL reach: not where the library wins |

---

## 5. Not possible today

| Use case | What is missing | After Phase 2 |
|---|---|---|
| Venue crowd-flow and evacuation planning | Walls, exits, flow fields; agents sensing the map | I13 |
| Footfall and queues in shops, malls, airports, stations, hospitals | The same | I14 |
| RTS on real maps; colony and god games | Pathfinding, flow fields; a life cycle; discrete actions | G13, G11 |
| Evolution games, evolution labs, artificial-life research | Birth, death and mutation on the GPU; a genome per agent | G10, E8, R10 |
| Stealth games with guards that see; camouflage | Ray or rendered eyes | G12, E8 |
| Flocks, schools and drone formations in 3D | Height integrated, flight | G15, R1, I7 |
| Ant trails, chemotaxis, resource maps | Sensors that read app buffers | E5, R13, R14 |
| Collisions, cloth, granular flow | Physics P2–P3 | E6, R15, D6 |
| Train on a server, play in the browser | Runs on a Linux GPU; evolution | R16 |

**Poor fits, Phase 2 or not:** agents driven by language models (`OrtModel` is float32-only),
non-spatial models (markets, social networks), N-body gravity and fluid dynamics (no head start
over any compute code), photoreal offline rendering, audiences without WebGPU.

---

## 6. Ways to earn from the library itself

These cut across the use cases, so they are not ranked with them.

| Model | Likely /yr | Effort to start | Comparables and notes |
|---|---|---|---|
| WebGPU consulting and contract builds (C1, I4, I6, R7) | $10k–60k, part-time | XS | Solo WebGL/WebGPU developers quote $50–120 an hour (some $25–30); solo fixed-price projects run $1.5k–12k. The fastest money, but it sells your time, not the library |
| Courses and workshops (E4) | $1k–15k | M | Three.js Journey: ~50,800 students at $95, the category's outlier |
| Paid kits and templates (D2) | $0.5k–5k | S each | Unity crowd assets sell for $30–60 a seat |
| Sponsorships | $0–3k | XS | GitHub Sponsors income sits in a long tail; disclosed examples: $593 (2023) and ~$12.8k (2022) a year |
| Grants | €0–50k per granted project | S (a proposal) | NLnet NGI Zero: €5k–50k. Residents of the EU and associated countries are preferred among equal proposals; the work must be R&D and fit its open-internet goals |
| A commercial licence for new modules, core staying MIT | $0–20k | S | Needle Engine (per seat), Cosmograph (free for non-commercial use, paid for commercial). Adds friction while adoption is the bottleneck |
| A service on your own platform (I1 now, I13 after Phase 2) | See I1, I13 | L | The highest ceiling, and you own the channel |

---

## 7. What the ratings say, in short

- **The library's edge is clearest where many agents are the point:** horde and battle games,
  crowds on your event platform, explainers and art built on swarms. It is weakest where an
  ordinary 3D viewer would do (configurators, CAD, a general engine): three.js wins there, with
  WebGL reach.
- **Distribution decides more than features.** Your event platform is the one channel you own;
  Steam and itch.io are the self-serve ones that fit WebGPU-only content; portals reach only their
  WebGPU share (~68% on Poki).
- **Research and education create users and credibility, not money.** Free rivals (NetLogo, PhET,
  FLAME GPU) set the price at zero; grants are the realistic money there.
- **Today's audience is small** (154 npm downloads a month), so every plan needs something that
  earns attention. A game, an explainer or a benchmark does.

---

## 8. Top 3 recommendations now, and why

### 1. G1 — a survivors-like horde game (score 10)

- **The swarm is the gameplay.** The genre sells on enemy counts. No off-the-shelf web engine
  simulates and draws thousands of sensing, steering, animated, culled enemies on the GPU;
  NullGraph holds 16K at 60 fps on an integrated GPU. This is its clearest edge.
- **Distribution comes with it:** Steam through a Chromium wrapper on Windows, and itch.io for the
  web build. Players already buy this genre.
- **1–2 months on Herd & Wolves' code:** chase and separation rules, an app stage for damage and
  death, palettes for animated enemies, LOD, the toon look.
- **It markets the library.** "Made with NullGraph" reaches developers through players' videos.
- **Money:** likely $1k–15k in year 1; a breakout reaches $100k–1M+. The appetite for huge counts
  is real (Ultimate Epic Battle Simulator 2: ~$2M gross, est.), but the 2025 Steam median was
  about $249: most games earn little.
- **First step, a one-week spike:** WebGPU inside the wrapper on Windows; enemies at 16K, 64K and
  100K on a mid-range discrete GPU and on the iGPU (16K holds 60 fps there; 64K took 74 ms a step
  in Herd & Wolves). The measured count decides what the game promises.
- **Risks:** a crowded genre; WebGPU-only rules out a third of portal players; low-end GPUs.

### 2. I1 — the event photo booth on your own event platform (score 8)

- **You own the channel.** Organizers on your platform are buyers you already reach, the thing
  almost every solo developer lacks. The product plan already calls it the most directly
  monetizable idea.
- **The spend is proven:** booth rentals bill $200–700 an hour, and booth software sells for
  $49–249 a month.
- **The difference is real 3D:** the attendee stands inside a lit, shadowed branded set, in front
  of and behind props, not on a flat background swap. Nothing leaves the device: private, and no
  server GPU bill, so each extra booth costs you almost nothing.
- **It needs no Phase 2:** `inference`, `lights`, `shadows`, IBL and `post` exist.
- **Money:** likely $2k–25k, driven by events a year × the share that buys × the price (for
  example 500 × 25% × $150 ≈ $19k); $100k–500k as a booth service for others.
- **First step, a 1–2 week spike:** ONNX Runtime's WebGPU on iPad Safari 26 and a laptop; a small,
  permissively licensed float32 person-segmentation model at 15+ fps; a mask composited into a lit
  3D set.
- **Risks:** AI booth effects are becoming common; ONNX Runtime on iPad Safari is unverified; model
  licences (avoid non-commercial ones).

### 3. E1 + C3 — "a neural network learned to herd," as an interactive page and short videos (scores 8 and 6)

- **Nearly free:** steps 7–10 built the whole story. A rule, recorded; a network trained to copy
  it; the same network on the WGSL kernel and on ONNX Runtime, switched live with their costs.
- **Every other path needs an audience,** and today's is small. Simulation explainers travel far:
  the Washington Post's 2020 simulation was reported as its most-read story.
- **The video tool is already your next step:** D2, offline films, designed 2026-10-16, targets one
  minute of 1M Herd & Wolves agents at 60 fps, which no GPU here can show live.
- **It sets up Phase 2,** whose evolution content is the most shareable kind.
- **Money:** little directly ($0–5k), but it feeds players for G1, buyers for a course (E4),
  sponsors and consulting leads.
- **First step:** a 1,500-word page on nullgraph.dev and a 3–5 minute video, posted where WebGPU,
  game and ML developers read.
- **Risk:** content takes steady time, and YouTube grows slowly.

**Alongside them:** freelance branded experiences (C1) pay best per hour today and can fund the
other three. Keep them small, so they do not replace the products.

---

## 9. Sources

Market and price comparables, read 2026-10-16. Most are vendor pages or third-party estimates; the
sales figures for games are estimates, not developer disclosures.

- Crowd and evacuation software: [Pathfinder licensing (Thunderhead)](https://www.thunderheadeng.com/pathfinder/licensing):
  $2,350 a year node-locked, $7,850 perpetual · [MassMotion (AEC Magazine)](https://aecmag.com/news/massmotion/)
  and [MassMotion Flow (AEC Magazine)](https://aecmag.com/news/massmotion-flow/): £20,000 perpetual
  (older), Flow £5,000 · [Architosh, 2015](https://architosh.com/2015/06/aia-show-floor-reports-oasys-massmotion-and-massmotion-flow/amp):
  $30,000 a seat · [AnyLogic pricing, third party, unverified](https://checkthat.ai/brands/anylogic/pricing):
  Professional ~$18,990 · [Golaem in Autodesk's collection (CG Channel, 2025)](https://www.cgchannel.com/2025/03/autodesk-adds-golaem-9-2-to-its-media-entertainment-collection/)
  and [Golaem bridges open-sourced (CG Channel, 2026)](https://www.cgchannel.com/2026/08/autodesk-open-sources-golaem-for-houdini-and-unreal-engine/):
  $2,790 a year, not sold alone.
- Markets: [pedestrian simulation software (Spherical Insights)](https://www.sphericalinsights.com/reports/pedestrian-simulation-software-market):
  ~$1.78B in 2025 · [crowd simulation software (Market Intelo)](https://marketintelo.com/report/crowd-simulation-software-market):
  ~$0.6B in 2025 · [event management software (MarketsandMarkets, via Webull)](https://www.webull.com/news/15243009160733696):
  $16.4B, against [IMARC's $7.2B](https://www.imarcgroup.com/global-event-management-software-market) ·
  [synthetic data generation (360iResearch, via GII)](https://gii.tw/report/ires1471221-synthetic-data-generation-market-by-component.html):
  ~$765M in 2025. Analyst estimates disagree by up to 3×.
- Photo booths: [Snappic on G2](https://www.g2.com/products/snappic-booth/reviews): $69 and $189 a
  month · [Touchpix vs Snappic](https://touchpix.com/compare/touchpix-vs-snappic/) ·
  [Simple Booth tiers (DSLRBooth comparison)](https://dslrbooth.com/compare/lumabooth-vs-simple-booth):
  $49–249 a month · rentals: [Reflects](https://www.gigsalad.com/reflects_dearborn_heights) $200–250
  an hour, [PixelAI Pro](https://www.eventective.com/brooklyn-ny/pixel-ai-pro-796117.html) $300–700
  an hour, [Alexa Corporate](https://www.cliqora.com/photographer/alexa-corporate-photo-booth-and-event-photography-chicago)
  $199+ an hour.
- Games: [indie revenue on Steam (Ziva, citing VG Insights/Alinea)](https://ziva.sh/blogs/indie-game-revenue):
  2025 median ~$249 gross, 66% under $1k, 90% under $50k ·
  [publishing on Poki (Cinevva)](https://app.cinevva.com/guides/publish-game-poki): a first web game
  $500–3,000 a month, 50/50 split on Poki's traffic ·
  [Building web browser games in 2026 (Poki, 2026-06-05)](https://poki.com/blog/building-web-browser-games-2026):
  ~68% of players on WebGPU devices, "treat WebGPU as an enhancement" ·
  [Ultimate Epic Battle Simulator 2 (Raijin)](https://raijin.gg/app/1468720/Ultimate_Epic_Battle_Simulator_2):
  ~164.8k copies, ~$2M gross.
- Education: [Labster pricing](https://www.labster.com/es/precios) and
  [a listing (FitGap)](https://us.fitgap.com/products/labster): ~$3,000 a year for 50 students ·
  [a Gizmos district proposal, 2023](https://mtdiablo.ic-board.com/attachments/69eaf0d9-1ee4-4e30-ba5c-f057b11b4125.pdf):
  $18,565 a year · [Gizmos purchasing](https://help.explorelearning.com/en/articles/9000368-purchasing-gizmos):
  $149 a year for home use · [NetLogo (CCL)](https://ccl.northwestern.edu/education.shtml): free,
  "tens of thousands" of users · [Three.js Journey](https://threejs-journey.com/): ~50,800 students,
  $95; its WebGPU and TSL course $45.
- Services and installations: [WebGL developer rates](https://svilenkovic.com/3d/webgl-developer-rate)
  and [freelance WebGL projects](https://svilenkovic.com/3d/freelance-webgl-developer) (one
  freelancer's figures): $50–120 an hour, projects $1.5k–12k, studios $15k–50k ·
  [three.js forum](https://discourse.threejs.org/t/paid-looking-for-a-three-js-developer/71234):
  $25–30 an hour · [installation costs (Utsubo)](https://www.utsubo.com/blog/interactive-installation-cost-budget-guide):
  $15k–45k a station · [costing exhibitions (Welsh Government)](https://www.gov.wales/costing-exhibitions-toolkit-museums-html) ·
  [TouchDesigner products (Derivative)](https://docs.derivative.ca/TouchDesigner_Products): $600 and
  $2,200 · [SPH Engineering drone show technologies](https://shop.sphengineering.com/collections/drone-show-technologies):
  $28,000 plus $9,200 a year · [Skybrush](https://skybrush.io/diy/): free community edition.
- Developer products and licensing: [GPU Instancer – Crowd Animations (Unity Asset Store)](https://assetstore.unity.com/packages/tools/animation/gpu-instancer-crowd-animations-145114):
  $60 a seat · [Needle Engine licensing](https://cloud.needle.tools/faq/what-are-the-licensing-options-for-commercial-or-enterprise-use) ·
  [Cosmograph licensing](https://cosmograph.app/licensing/).
- Funding: [NLnet guide for applicants](https://nlnet.nl/core/guideforapplicants/): €5,000–50,000 ·
  [GitHub Sponsors study (arXiv 2111.13323)](https://arxiv.org/abs/2111.13323v1) ·
  disclosed sponsor income: [~$12.8k in 2022](https://dev.to/azu/my-github-sponsors-revenue-2022-38ab),
  [$593 in 2023](https://old.lgug2z.com/articles/github-sponsorship-breakdown-for-2023/).
- Research tools: [FLAME GPU (University of Sheffield)](https://paulrichmond.shef.ac.uk/cs/FLAMEGPU):
  CUDA, grant-funded.
- Reach of simulation explainers: [Simulating a pandemic (DataJournalism.com)](https://datajournalism.com/read/longreads/simulating-a-pandemic) ·
  [Dezeen on the Washington Post simulator](https://www.dezeen.com/2020/03/26/the-washington-post-coronavirus-inforgraphics-harry-stevens/amp/).
- WebGPU support: [web.dev](https://web.dev/blog/webgpu-supported-major-browsers): Safari 26,
  Firefox 141 on Windows.
- This library: [npm downloads API](https://api.npmjs.org/downloads/point/last-month/null-graph)
  and the GitHub API, read 2026-10-16. Capabilities and measurements: [`api/agents.md`](./api/agents.md),
  [`api/inference.md`](./api/inference.md), [`agents-design.md`](./agents-design.md),
  [`work/plan.md`](./work/plan.md), [`handoff.md`](./handoff.md) ("After Cycle B — the ONNX/AI
  product plan", "Far horizon").
