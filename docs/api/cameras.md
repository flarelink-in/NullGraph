# Cameras API Usage

The `null-graph/cameras` module provides ready-to-use camera controllers. Each controller is split
into two intentionally decoupled pieces so the same math can run on the main thread *or* inside a
Web Worker:

- **State classes** (`OrbitalCameraState`, `FlyCameraState`, `FollowCameraState`,
  `PathCameraState`) — pure math. No DOM, no events. They hold parameters, clamp them to limits,
  and produce eye/target positions for `Camera.updateView()`.
- **Event proxies** (`OrbitalEventProxy`, `FlyEventProxy`) — thin DOM input listeners. They
  capture pointer/wheel/key events and emit **plain numbers**, so they can feed a state object
  living on another thread. Follow and Path cameras are driven by game logic and need no proxy.

```ts
import {
    OrbitalCameraState, OrbitalEventProxy,
    FlyCameraState, FlyEventProxy,
    FollowCameraState, PathCameraState,
} from 'null-graph/cameras';
import { Camera } from 'null-graph';
```

---

## `OrbitalCameraState`

Holds and mutates the orbit parameters. All angles are in radians.

```ts
const orbit = new OrbitalCameraState({
    target: [0, 1, 0],   // point the camera looks at
    distance: 7,         // radius of the orbit
    azimuthAngle: 0,     // horizontal angle
    polarAngle: 1.2,     // vertical angle (~68°, looking slightly down)

    // Optional clamps (sensible defaults shown)
    minDistance: 2,
    maxDistance: 20,
    minPolarAngle: 0.01,             // avoids gimbal lock at the north pole
    maxPolarAngle: Math.PI - 0.01,   // full freedom under the model
    minAzimuthAngle: -Infinity,      // endless horizontal spin
    maxAzimuthAngle: Infinity,
});
```

| Method | Description |
|---|---|
| `applyRotation(deltaAzimuth, deltaPolar)` | Adds rotation deltas and clamps to the angle limits. |
| `applyZoom(deltaDistance)` | Adds a zoom delta and clamps to `[minDistance, maxDistance]`. |
| `getPosition()` | Returns the final `[x, y, z]` eye position in world space. |

## `OrbitalEventProxy`

Attaches pointer and wheel listeners to a canvas and forwards abstract input deltas.

```ts
const proxy = new OrbitalEventProxy({
    canvas,
    sensitivity: 0.01,       // rotation scale (default 0.01)
    zoomSensitivity: 0.05,   // zoom scale (default 0.05)
    onInput: (type, x, y) => {
        if (type === 'ROTATE') orbit.applyRotation(x, y);
        if (type === 'ZOOM')   orbit.applyZoom(y);
    },
});

// Call proxy.destroy() to remove all listeners when tearing down.
```

---

## `FlyCameraState` + `FlyEventProxy`

First-person free-fly: drag to look, hold W/S (forward), A/D (strafe), Q/E (vertical) to move.
Movement keys are continuous, so the proxy exposes them as **polled axes** rather than events.

```ts
const fly = new FlyCameraState({
    position: [0, 2, -8],
    yaw: 0,              // radians; 0 looks toward +Z
    pitch: 0,            // radians; positive looks up
    moveSpeed: 6,        // world units per second
    // minPitch / maxPitch default to just short of straight down/up
});

const flyProxy = new FlyEventProxy({
    canvas,
    sensitivity: 0.005,
    onInput: (_type, x, y) => fly.applyLook(x, y),
});

function frame(dt: number) {
    const [forward, right, up] = flyProxy.getMoveAxes(); // each -1..1
    fly.applyMove(forward, right, up, dt);
    camera.updateView(fly.getPosition(), fly.getTarget());
}
```

| Method | Description |
|---|---|
| `applyLook(deltaYaw, deltaPitch)` | Adds look deltas; pitch is clamped short of the poles. |
| `applyMove(forward, right, up, dt)` | Moves relative to the view direction, scaled by `moveSpeed * dt`. |
| `getPosition()` / `getTarget()` | Eye position and a point one unit along the look direction. |

## `FollowCameraState`

Third-person chase camera. Game logic pushes the followed point in every frame; `update(dt)`
eases the camera toward `point + offset` with frame-rate-independent exponential damping.

```ts
const follow = new FollowCameraState({
    offset: [0, 3, -6],     // camera position relative to the followed point
    lookOffset: [0, 1, 0],  // aim above the feet
    stiffness: 5,           // damping per second; higher snaps faster
});

function frame(dt: number) {
    follow.setTarget(player.x, player.y, player.z);
    follow.update(dt);
    camera.updateView(follow.getPosition(), follow.getTarget());
}
// follow.snap() teleports the camera instantly (level start, respawn).
```

## `PathCameraState`

Cinematic dolly along a Catmull-Rom spline through waypoints. Advance it with time or scrub
`progress` directly (e.g. from scroll position).

```ts
const path = new PathCameraState({
    points: [[0, 2, -10], [8, 3, 0], [0, 5, 10], [-8, 3, 0]],
    duration: 12,            // seconds for one traversal
    loop: true,              // wrap and close the spline into a ring
    // lookTarget: [0, 1, 0] // fixed aim; omit to face the direction of travel
});

function frame(dt: number) {
    path.advance(dt);        // or: path.progress = scrollFraction;
    camera.updateView(path.getPosition(), path.getTarget());
}
```

---

## Putting it together

Drive a core `Camera` from the orbit state every frame:

```ts
const camera = new Camera(75, canvas.width / canvas.height, 0.1, 1000);

function frame() {
    camera.updateView(orbit.getPosition(), orbit.target);
    engine.updateCamera(camera);
    engine.render();
    requestAnimationFrame(frame);
}
frame();
```

Because `OrbitalEventProxy` only emits numbers and `OrbitalCameraState` is pure math, you can run
the proxy on the main thread and post its deltas to a worker that owns the state and the engine —
keeping the camera fully inside the Data-Oriented, off-main-thread model.
