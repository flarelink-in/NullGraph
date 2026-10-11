# Validation and open platform pointers

Run these commands from a library repository checkout after `npm install` and `npm run build:lib`.
They are not scripts installed into arbitrary applications by the engine package.

```sh
npm run test:agent-kit
npm run test:gpu -- --only render:quick-start --only render:instancing --only render:culling --only visibility
```

To inspect one scene, use `npm run render -- test/gpu/scenes/culling.mjs --out culling.png
--json culling.json --frames 2` on one command line. The [runner](../../test/gpu/render.mjs)
captures device errors and loss through the [Node host](../../test/gpu/node-gpu.mjs), waits
for submitted work, writes PNG and JSON, and rejects uniform frames unless explicitly allowed.
Exit codes: 0 rendered, 1 recorded GPU error or uniform frame, 2 invalid arguments or scene failure.
Device/bootstrap and other uncaught failures also fail; do not infer success from an absent report.

## What each layer proves

| Layer | Evidence | Not proved |
|---|---|---|
| Agent-kit checks | Links, version identity, example/doc equality, example registration, budget | GPU execution |
| Starter renders | No recorded GPU errors; non-uniform output | Correct cull counts or requested appearance |
| Visibility GPU suite | Draw counts at 1k/13k; GPU frustum vs CPU mirror; Hi-Z wall controls | Every application integration |
| Human/image review | Camera composition, visible artifacts and requested content | General energy stability |
| Fresh-agent evaluation | Ability to discover, build, debug and validate unaided | Not run merely by engine tests |

The [visibility suite](../../test/visibility-gpu/run.js) contains the semantic assertions.
The starter scenes do not themselves expose GPU readbacks or claim to reproduce those assertions.
Their smoke tests run in the Node registry, not the browser registry.

## Platform coverage remains open

The runner depends on the development-only `webgpu` package and a usable GPU adapter. See
the Node host for its current binary-loading diagnostics. No additional platform coverage is
claimed by this kit. If Dawn cannot run, use the existing [browser runner](../../test/gpu/browser-run.mjs)
and [visibility browser harness](../../test/visibility-gpu/index.html) as investigation pointers.
Browser checks do not automatically run these Node starter modules. Do not silently skip a
failed GPU run or report a platform as tested because documentation mentions it.

## Before/after discovery evaluation (still pending)

Use independent fresh agent contexts and identical prompts/resources except for kit access.
Ask for a batched instance scene and a GPU-culling scene. Record installed engine version,
kit commit/tag, prompt, elapsed time, docs fetched, corrections and GPU availability. Grade
with actual renders, image review and task-specific assertions, not self-reported success.
Do not call an implementation session its own independent baseline. Preserve failed attempts.

## Release checklist

1. Review and commit the intended library changes; exclude unrelated frontend work.
2. Confirm package manifest 1.1.1 and run the kit and GPU checks against a fresh build.
3. Create and publish `v1.1.1` from that reviewed commit. Do not tag a dirty tree as if it includes it.
4. Verify the tagged root index, nested guide and example text can be fetched anonymously.
5. Add verified tag-specific URLs to the package README before publication. Until then its
   branch URL is explicitly development-only, not a version-matching guarantee.
6. Dry-run packaging: Option 1 ships README, engine and declarations, not this kit.
7. If Option 2 is separately approved, ship only the curated kit and its required examples;
   resolution subpaths locate files, they do not automatically read instructions or ship targets.