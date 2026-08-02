# Cameras API Usage

The `null-graph/cameras` module provides ready-to-use camera controllers for orbiting, panning,
and zooming around a scene. It is split into two intentionally decoupled pieces so the same math
can run on the main thread *or* inside a Web Worker:

- **`OrbitalCameraState`** — pure spherical-coordinate math. No DOM, no events. It holds the
  target, distance, and angles, clamps them to limits, and produces a Cartesian eye position.
- **`OrbitalEventProxy`** — a thin DOM input listener. It captures pointer/wheel events and emits
  **plain numbers** through a callback, so it can feed a state object living on another thread.

```ts
import { OrbitalCameraState, OrbitalEventProxy } from 'null-graph/cameras';
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
