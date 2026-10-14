// src/core/BufferManager.ts
var BufferManager = class {
  device;
  constructor(device) {
    this.device = device;
  }
  // Creates a VBO (Vertex Buffer Object)
  createVertexBuffer(data) {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      // Align memory to 4 bytes as required by WebGPU
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
  }
  // Creates an IBO (Index Buffer Object)
  createIndexBuffer(data) {
    const buffer = this.device.createBuffer({
      size: Math.ceil(data.byteLength / 4) * 4,
      // Ensure 4-byte alignment
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    if (data instanceof Uint16Array) {
      new Uint16Array(buffer.getMappedRange()).set(data);
    } else {
      new Uint32Array(buffer.getMappedRange()).set(data);
    }
    buffer.unmap();
    return buffer;
  }
  /**
   * A `dispatchWorkgroupsIndirect` argument buffer — `count` triples of `[x, y, z]` u32,
   * writable by a compute stage and readable as dispatch arguments.
   *
   * Exists so the correct usage mask is the default path rather than something to remember:
   * `INDIRECT` alone cannot be filled, `STORAGE` alone cannot be dispatched from, and getting
   * either wrong fails as a dropped dispatch with no error and a symptom two subsystems away.
   * `COPY_DST` is included so `preClearBuffers` can reset the triple, which is the ordinary way
   * a counter-driven stage starts its frame.
   *
   * Initialized to `[0, 0, 0]` — a stage whose producer has not run yet dispatches nothing,
   * which is the safe direction. The alternative, uninitialized memory read as a workgroup
   * count, is a hang.
   *
   * **The producing kernel must clamp.** Nothing here can validate a value the GPU wrote, so
   * keeping each component at or below `device.limits.maxComputeWorkgroupsPerDimension` is the
   * kernel's job — `min(computed, cap)` in the shader that writes the triple.
   */
  createDispatchIndirectBuffer(count = 1, label) {
    if (!Number.isInteger(count) || count < 1) {
      throw new Error(`BufferManager: createDispatchIndirectBuffer count ${count} must be a positive integer.`);
    }
    const buffer = this.device.createBuffer({
      label: label ?? "NullGraph_DispatchIndirect",
      size: count * 12,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Uint32Array(buffer.getMappedRange()).fill(0);
    buffer.unmap();
    return buffer;
  }
};

// src/core/RenderPass.ts
var RenderPassNode = class {
  name;
  isMainScreenPass;
  batches = [];
  colorAttachments;
  depthStencilAttachment;
  /** Attachment size in pixels — see {@link RenderPassConfig.size}. */
  size;
  constructor(config) {
    this.name = config.name;
    this.isMainScreenPass = config.isMainScreenPass;
    this.colorAttachments = config.colorAttachments || [];
    this.depthStencilAttachment = config.depthStencilAttachment;
    this.size = config.size;
  }
  addBatch(batch) {
    this.batches.push(batch);
  }
  removeBatch(batch) {
    const index = this.batches.indexOf(batch);
    if (index !== -1) {
      this.batches.splice(index, 1);
    }
  }
};

// src/core/types.ts
var RenderBatch = class {
  pipeline;
  storageBuffer;
  bindGroup;
  stride = 0;
  // Geometry
  vertexBuffer = null;
  indexBuffer = null;
  indexCount = 0;
  indexFormat = "uint16";
  // Instancing
  currentInstanceCount = 0;
  maxInstanceCount = 0;
  // Indirect Drawing
  isIndirect = false;
  indirectBuffer;
  sourceStorageBuffer;
  /**
   * Per-batch viewport in attachment pixels (shadows-atlas-design.md §12.1). Copied from
   * `PipelineConfig.viewport` at creation; mutable so a system (the shadow atlas) can retarget
   * a batch when its tile moves without rebuilding the pipeline.
   */
  viewport;
  // Render-specific Custom Data
  customBindGroups = {};
};
var ComputeStageNode = class {
  name;
  computePipeline;
  customBindGroups = {};
  preClearBuffers;
  /**
   * Set `false` to skip this stage — its `preClearBuffers` **and** its dispatch — for a frame,
   * without splicing it out of the sequence array. Mutable and checked per frame by
   * `RenderPipeline`, so a system that owns many stages can gate them cheaply.
   *
   * The shadow atlas uses this: a cached tile that is not being re-rendered should not pay for
   * its caster cull either (shadows-atlas-design.md §8). Defaults to `true`, so every existing
   * stage runs exactly as before.
   */
  enabled = true;
  /**
   * Buffer holding `[x, y, z]` workgroup counts — see {@link ComputeStageConfig.indirectBuffer}.
   * Mutable, so a system can retarget a stage at a different count buffer without rebuilding the
   * pipeline (the same reason `RenderBatch.viewport` is mutable).
   */
  indirectBuffer;
  /** Byte offset into {@link indirectBuffer}. */
  indirectOffset = 0;
  _dispatchCount;
  constructor(config) {
    this.name = config.name;
    this._dispatchCount = config.dispatchCount || [1, 1, 1];
    this.indirectBuffer = config.indirectBuffer;
    this.indirectOffset = config.indirectOffset ?? 0;
    this.preClearBuffers = config.preClearBuffers || [];
  }
  /** True when this stage takes its workgroup count from a buffer rather than from the CPU. */
  get isIndirect() {
    return this.indirectBuffer !== void 0;
  }
  get dispatchCount() {
    if (typeof this._dispatchCount === "function") {
      return this._dispatchCount();
    }
    return this._dispatchCount;
  }
};

// src/core/WebGPUContext.ts
var PROGRESSION_FEATURES = [
  "subgroups",
  "shader-f16",
  "bgra8unorm-storage",
  "dual-source-blending"
];
var PROGRESSION_LIMITS = [
  "maxStorageBuffersPerShaderStage",
  "maxStorageBufferBindingSize",
  "maxBufferSize"
];
var HEADLESS_SCREEN_FORMAT = "rgba8unorm";
var WebGPUContext = class {
  device;
  /**
   * The canvas context — `undefined` when the engine was initialised without a canvas, in which
   * case the main-screen pass renders into {@link screenTexture} instead. Read the frame's
   * target through {@link getScreenView}, which covers both.
   */
  context;
  format;
  depthTexture;
  cameraUniformBuffer;
  /**
   * The offscreen main-screen target when there is no canvas; `undefined` otherwise. Usage is
   * `RENDER_ATTACHMENT | COPY_SRC`, so a test or a render-to-PNG tool can read the frame back —
   * readback itself stays out of the engine (it is legal in a test, forbidden in the frame loop).
   */
  screenTexture;
  screenView;
  /**
   * `false` when the device was adopted through `init(…, { device })`. NullGraph never destroys
   * a device it did not create — the owner (ONNX Runtime, another renderer, a test harness)
   * still has work on it.
   */
  ownsDevice = true;
  /**
   * Whether {@link depthTexture} carries `TEXTURE_BINDING` and can therefore be sampled
   * (environment-aerial-design.md §9.1). Opt in at {@link init} with `{ samplableDepth: true }`.
   *
   * Off by default, and the default is the point: adding `TEXTURE_BINDING` can disable some
   * drivers' depth-compression fast paths for that texture, and it costs that on EVERY frame
   * whether or not anything samples it. Paying at construction for what you use beats paying
   * per frame for what you might.
   *
   * Three consumers want it, which is why it is one flag and not three: aerial perspective's
   * apply pass, `HiZPyramid` (which allocates a separate r32float pyramid today precisely
   * because this is unsampleable), and a depth pre-pass.
   */
  depthSamplable = false;
  /** Negotiated optional-feature snapshot, populated during {@link init}. */
  capabilities;
  width = 1;
  height = 1;
  /** True when the main-screen pass renders into {@link screenTexture} rather than a canvas. */
  get isHeadless() {
    return this.context === void 0;
  }
  /**
   * Initializes the WebGPU context with fallback-safe feature negotiation.
   *
   * @param canvas The canvas to present to, or `null` to render the main-screen pass into an
   *   offscreen {@link screenTexture} (Node/Dawn, tests, render-to-PNG). `null` needs
   *   `options.width` and `options.height`.
   * @param options See {@link EngineInitOptions}. `options.device` adopts an existing device.
   */
  async init(canvas, options = {}) {
    validateInitOptions(canvas, options);
    if (options.device) {
      this.adoptDevice(options.device, options.desiredFeatures);
    } else {
      await this.requestDevice(options);
    }
    this.cameraUniformBuffer = this.device.createBuffer({
      label: "Global Camera Uniform",
      size: 432,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.width = options.width ?? canvas?.width ?? 1;
    this.height = options.height ?? canvas?.height ?? 1;
    this.depthSamplable = options.samplableDepth ?? false;
    if (canvas) {
      this.context = canvas.getContext("webgpu");
      this.format = options.format ?? navigator.gpu.getPreferredCanvasFormat();
      this.context.configure({
        device: this.device,
        format: this.format,
        alphaMode: "premultiplied"
      });
      this.setupDepthTexture(canvas.width, canvas.height);
    } else {
      this.format = options.format ?? HEADLESS_SCREEN_FORMAT;
      this.setupScreenTexture(this.width, this.height);
      this.setupDepthTexture(this.width, this.height);
    }
  }
  /**
   * The view the main-screen pass renders into this frame: the canvas's current texture, or the
   * offscreen {@link screenTexture} when there is no canvas. The offscreen view is created once
   * per size, so the headless frame path allocates nothing here.
   */
  getScreenView() {
    if (this.context) return this.context.getCurrentTexture().createView();
    return this.screenView;
  }
  resize(width, height) {
    if (this.device) {
      this.width = width;
      this.height = height;
      if (this.isHeadless) this.setupScreenTexture(width, height);
      this.setupDepthTexture(width, height);
    }
  }
  /** Request an adapter and a device, negotiating optional features. The canvas-era path, unchanged. */
  async requestDevice(options) {
    const gpu = globalThis.navigator?.gpu;
    if (!gpu) {
      throw new Error("WebGPU unavailable: navigator.gpu is missing. Install one (Node: Dawn) or pass init(\u2026, { device }).");
    }
    let adapter = null;
    const powerPref = options.powerPreference ?? "high-performance";
    try {
      adapter = await gpu.requestAdapter({
        powerPreference: powerPref
      });
      if (!adapter) {
        console.warn(
          `NullGraph: No adapter found with powerPreference "${powerPref}". Falling back to default.`
        );
        adapter = await gpu.requestAdapter();
      }
    } catch (e) {
      console.warn(
        `NullGraph: Adapter request failed with powerPreference. Falling back to default. Error:`,
        e
      );
      adapter = await gpu.requestAdapter();
    }
    if (!adapter) {
      throw new Error("No appropriate GPU adapter found.");
    }
    const info = adapter.info;
    console.log(`NullGraph: Using GPU \u2014 ${describeAdapter(info)}`);
    const requested = /* @__PURE__ */ new Set();
    if (options.desiredFeatures) {
      for (const feature of options.desiredFeatures) {
        if (adapter.features.has(feature)) {
          requested.add(feature);
        } else {
          console.warn(
            `NullGraph: Optional feature "${feature}" is not supported by this device.`
          );
        }
      }
    }
    const requiredLimits = {};
    if (options.autoNegotiate ?? true) {
      for (const feature of PROGRESSION_FEATURES) {
        if (adapter.features.has(feature)) requested.add(feature);
      }
      for (const limit of PROGRESSION_LIMITS) {
        const value = adapter.limits[limit];
        if (typeof value === "number") requiredLimits[limit] = value;
      }
    }
    this.device = await adapter.requestDevice({
      requiredFeatures: [...requested],
      requiredLimits
    });
    this.ownsDevice = true;
    this.capabilities = buildCapabilities(this.device, info);
  }
  /**
   * Adopt a device another library created (ONNX Runtime Web, a test harness, another
   * renderer). Nothing is negotiated — features were fixed when its owner created it — so the
   * capability snapshot is simply what the device has. The baseline path needs no optional
   * feature, so whatever the owner requested is enough to render; it only decides which
   * accelerated paths light up.
   */
  adoptDevice(device, desiredFeatures) {
    this.device = device;
    this.ownsDevice = false;
    if (desiredFeatures) {
      for (const feature of desiredFeatures) {
        if (!device.features.has(feature)) {
          console.warn(`NullGraph: Optional feature "${feature}" is not enabled on the adopted device \u2014 its owner must request it.`);
        }
      }
    }
    const info = device.adapterInfo;
    console.log(`NullGraph: Adopted an existing GPU device \u2014 ${describeAdapter(info)}`);
    this.capabilities = buildCapabilities(device, info);
  }
  setupScreenTexture(width, height) {
    if (this.screenTexture) this.screenTexture.destroy();
    this.screenTexture = this.device.createTexture({
      label: "NullGraph_Screen",
      size: [Math.max(1, width), Math.max(1, height)],
      format: this.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
    });
    this.screenView = this.screenTexture.createView();
  }
  setupDepthTexture(width, height) {
    if (this.depthTexture) this.depthTexture.destroy();
    const dWidth = Math.max(1, width);
    const dHeight = Math.max(1, height);
    this.depthTexture = this.device.createTexture({
      size: [dWidth, dHeight],
      format: "depth24plus",
      usage: this.depthSamplable ? GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING : GPUTextureUsage.RENDER_ATTACHMENT
    });
  }
};
function validateInitOptions(canvas, options) {
  if (options.device) {
    if (options.powerPreference !== void 0) {
      throw new Error("NullGraph.init: powerPreference does nothing on an adopted device \u2014 set it where the device is requested.");
    }
    if (options.autoNegotiate !== void 0) {
      throw new Error("NullGraph.init: autoNegotiate does nothing on an adopted device \u2014 its features are fixed; read engine.capabilities.");
    }
  }
  if (!canvas) {
    const { width, height } = options;
    if (width === void 0 || height === void 0) {
      throw new Error("NullGraph.init: headless init (canvas = null) needs options.width and options.height.");
    }
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
      throw new Error(
        `NullGraph.init: headless screen size must be positive integers, got ${width} x ${height}.`
      );
    }
  }
}
function describeAdapter(info) {
  if (!info) return "unknown adapter";
  return [info.vendor, info.architecture, info.description].filter(Boolean).join(" ") || "unknown adapter";
}
function buildCapabilities(device, info) {
  const enabled = /* @__PURE__ */ new Set();
  for (const f of device.features) enabled.add(f);
  const subgroups = enabled.has("subgroups");
  return {
    features: enabled,
    subgroups,
    shaderF16: enabled.has("shader-f16"),
    bgra8unormStorage: enabled.has("bgra8unorm-storage"),
    dualSourceBlending: enabled.has("dual-source-blending"),
    // subgroupMinSize/MaxSize live on GPUAdapterInfo and stay optional until every
    // browser reports them; treat "subgroups off" or "unreported" as 0.
    subgroupMinSize: subgroups ? info?.subgroupMinSize ?? 0 : 0,
    subgroupMaxSize: subgroups ? info?.subgroupMaxSize ?? 0 : 0,
    has: (feature) => enabled.has(feature)
  };
}

// src/core/BatchManager.ts
var DISPATCH_INDIRECT_BYTES = 12;
function validateIndirectDispatch(config) {
  const buffer = config.indirectBuffer;
  if (!buffer) {
    if (config.indirectOffset !== void 0) {
      throw new Error(
        `ComputeStage "${config.name}": indirectOffset was set without an indirectBuffer. The offset addresses a buffer that does not exist; the stage would dispatch from dispatchCount and silently ignore it.`
      );
    }
    return;
  }
  const offset = config.indirectOffset ?? 0;
  if (!Number.isInteger(offset) || offset < 0 || offset % 4 !== 0) {
    throw new Error(
      `ComputeStage "${config.name}": indirectOffset ${offset} must be a non-negative multiple of 4 \u2014 the dispatch arguments are three u32.`
    );
  }
  if ((buffer.usage & GPUBufferUsage.INDIRECT) === 0) {
    throw new Error(
      `ComputeStage "${config.name}": indirectBuffer lacks GPUBufferUsage.INDIRECT, so the dispatch cannot read its workgroup count. Allocate it with bufferManager.createDispatchIndirectBuffer(), or add INDIRECT | STORAGE to its usage.`
    );
  }
  if ((buffer.usage & GPUBufferUsage.STORAGE) === 0 && (buffer.usage & GPUBufferUsage.COPY_DST) === 0) {
    throw new Error(
      `ComputeStage "${config.name}": indirectBuffer has neither STORAGE nor COPY_DST usage, so nothing can ever write its workgroup count and the dispatch is permanently zero. Add STORAGE for a compute stage to fill it, or COPY_DST to write it from the CPU.`
    );
  }
  if (buffer.size < offset + DISPATCH_INDIRECT_BYTES) {
    throw new Error(
      `ComputeStage "${config.name}": indirectBuffer is ${buffer.size} bytes; reading three u32 at offset ${offset} needs ${offset + DISPATCH_INDIRECT_BYTES}.`
    );
  }
}
var BatchManager = class {
  constructor(ctx) {
    this.ctx = ctx;
  }
  /**
   * Creates a pure rendering batch. No hidden compute execution.
   */
  createBatch(pass, config) {
    const batch = new RenderBatch();
    batch.stride = config.strideFloats;
    batch.isIndirect = config.isIndirect || false;
    batch.viewport = config.viewport;
    if (!pass.isMainScreenPass && !pass.size && pass.batches.length > 0) {
      const scoped = batch.viewport !== void 0;
      if (pass.batches.some((b) => b.viewport !== void 0 !== scoped)) {
        throw new Error(
          `BatchManager: pass "${pass.name}" mixes viewport-scoped and full-pass batches but has no size, so the full viewport cannot be restored between them (viewport state persists across draws in a pass). Give the pass a size: [width, height], or give every batch in it an explicit viewport.`
        );
      }
    }
    const label = config.label ?? `${pass.name} #${pass.batches.length}`;
    const shaderModule = this.ctx.device.createShaderModule({ label, code: config.shaderCode });
    const formats = config.depthOnly ? [] : config.targetFormats || [config.targetFormat || this.ctx.format];
    const colorTargets = formats.map((fmt) => {
      const target = { format: fmt };
      if (config.blend) {
        target.blend = config.blend;
      }
      return target;
    });
    const expectsDepth = pass.depthStencilAttachment !== void 0 || pass.isMainScreenPass;
    batch.pipeline = this.ctx.device.createRenderPipeline({
      label,
      layout: "auto",
      vertex: {
        module: shaderModule,
        entryPoint: "vs_main",
        buffers: config.vertexLayouts || []
      },
      ...config.depthOnly ? {} : {
        fragment: {
          module: shaderModule,
          entryPoint: "fs_main",
          targets: colorTargets
        }
      },
      primitive: {
        topology: config.topology || "triangle-list"
      },
      ...expectsDepth ? {
        depthStencil: {
          depthWriteEnabled: config.depthWriteEnabled !== void 0 ? config.depthWriteEnabled : true,
          depthCompare: config.depthCompare || "less",
          format: "depth24plus"
        }
      } : {}
    });
    batch.storageBuffer = this.ctx.device.createBuffer({
      size: config.maxInstances * config.strideFloats * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    if (batch.isIndirect) {
      batch.indirectBuffer = this.ctx.device.createBuffer({
        size: 5 * 4,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
      batch.sourceStorageBuffer = config.sharedSourceBuffer || this.ctx.device.createBuffer({
        size: config.maxInstances * config.strideFloats * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
      });
    }
    batch.bindGroup = this.ctx.device.createBindGroup({
      layout: batch.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: config.cameraBuffer || this.ctx.cameraUniformBuffer } },
        { binding: 1, resource: { buffer: batch.storageBuffer } }
      ]
    });
    batch.maxInstanceCount = config.maxInstances;
    if (config.material) {
      batch.customBindGroups[1] = config.material.applyToBatch(batch);
    }
    pass.addBatch(batch);
    return batch;
  }
  /**
   * NEW: Explicitly create a separate, global compute execution stage.
   */
  createComputeStage(config) {
    validateIndirectDispatch(config);
    const stage = new ComputeStageNode(config);
    const computeModule = this.ctx.device.createShaderModule({ label: config.name, code: config.shaderCode });
    stage.computePipeline = this.ctx.device.createComputePipeline({
      label: config.name,
      layout: "auto",
      compute: { module: computeModule, entryPoint: "cs_main" }
    });
    return stage;
  }
  /**
   * As createComputeStage, but the pipeline comes from createComputePipelineAsync: its compile
   * does not hold the caller (agents-ort-design.md §12.1). A shader that does not compile rejects,
   * naming the stage, with the device's message; it raises no uncaptured error.
   */
  async createComputeStageAsync(config) {
    validateIndirectDispatch(config);
    const stage = new ComputeStageNode(config), device = this.ctx.device;
    device.pushErrorScope("validation");
    const module = device.createShaderModule({ label: config.name, code: config.shaderCode });
    const pipeline = device.createComputePipelineAsync({ label: config.name, layout: "auto", compute: { module, entryPoint: "cs_main" } });
    const scope = device.popErrorScope();
    try {
      stage.computePipeline = await pipeline;
    } catch (error) {
      throw new Error(`NullGraph: compute stage "${config.name}" did not compile: ${(await scope ?? error).message}`);
    }
    return stage;
  }
  /**
   * NEW: Attaches a bind group to a standalone compute stage.
   */
  setComputePassBindGroup(stage, entries, groupIndex = 0) {
    stage.customBindGroups[groupIndex] = this.ctx.device.createBindGroup({
      layout: stage.computePipeline.getBindGroupLayout(groupIndex),
      entries
    });
  }
  clearBatch(pass, batch) {
    pass.removeBatch(batch);
    if (batch.storageBuffer) batch.storageBuffer.destroy();
    if (batch.indirectBuffer) batch.indirectBuffer.destroy();
  }
  setBatchGeometry(batch, vertexBuffer, indexBuffer, indexCount, format = "uint16") {
    batch.vertexBuffer = vertexBuffer;
    batch.indexBuffer = indexBuffer;
    batch.indexCount = indexCount;
    batch.indexFormat = format;
  }
  updateBatchData(batch, rawData, instanceCount) {
    batch.currentInstanceCount = instanceCount;
    const targetBuffer = batch.isIndirect && batch.sourceStorageBuffer ? batch.sourceStorageBuffer : batch.storageBuffer;
    this.ctx.device.queue.writeBuffer(targetBuffer, 0, rawData.buffer, rawData.byteOffset, instanceCount * batch.stride * 4);
  }
  createTextureBindGroup(batch, textureView, sampler, extraEntries = [], groupIndex = 1) {
    const views = Array.isArray(textureView) ? textureView : [textureView];
    const entries = [{ binding: 0, resource: sampler }];
    views.forEach((view, i) => {
      entries.push({ binding: i + 1, resource: view });
    });
    if (extraEntries.length > 0) entries.push(...extraEntries);
    const bindGroup = this.ctx.device.createBindGroup({
      layout: batch.pipeline.getBindGroupLayout(groupIndex),
      entries
    });
    batch.customBindGroups[groupIndex] = bindGroup;
    return bindGroup;
  }
  setRenderPassBindGroup(batch, entries, groupIndex = 1) {
    if (batch.pipeline) {
      batch.customBindGroups[groupIndex] = this.ctx.device.createBindGroup({
        layout: batch.pipeline.getBindGroupLayout(groupIndex),
        entries
      });
    }
  }
  attachMaterial(batch, material, groupIndex = 1) {
    const bindGroup = material.applyToBatch(batch);
    batch.customBindGroups[groupIndex] = bindGroup;
    return bindGroup;
  }
};

// src/core/RenderPipeline.ts
var RenderPipeline = class {
  constructor(ctx) {
    this.ctx = ctx;
  }
  /**
   * Executes a clean sequence of compute stages and render passes linearly.
   */
  render(pipelineSequence) {
    if (pipelineSequence.length === 0) return;
    const commandEncoder = this.ctx.device.createCommandEncoder();
    let activeComputePass = null;
    for (const node of pipelineSequence) {
      if (node instanceof ComputeStageNode) {
        if (!node.enabled) continue;
        if (node.preClearBuffers.length > 0) {
          if (activeComputePass) {
            activeComputePass.end();
            activeComputePass = null;
          }
          for (const clear of node.preClearBuffers) {
            this.ctx.device.queue.writeBuffer(
              clear.buffer,
              clear.offset,
              clear.data
            );
          }
        }
        if (!activeComputePass) {
          activeComputePass = commandEncoder.beginComputePass();
        }
        activeComputePass.setPipeline(node.computePipeline);
        for (const groupIndexStr in node.customBindGroups) {
          const index = parseInt(groupIndexStr, 10);
          activeComputePass.setBindGroup(index, node.customBindGroups[index]);
        }
        if (node.indirectBuffer) {
          activeComputePass.dispatchWorkgroupsIndirect(node.indirectBuffer, node.indirectOffset);
        } else {
          const counts = node.dispatchCount;
          activeComputePass.dispatchWorkgroups(counts[0], counts[1], counts[2]);
        }
      } else if (node instanceof RenderPassNode) {
        if (activeComputePass) {
          activeComputePass.end();
          activeComputePass = null;
        }
        let colorAttachments = node.colorAttachments;
        let depthStencilAttachment = node.depthStencilAttachment;
        if (node.isMainScreenPass) {
          colorAttachments = [{
            view: this.ctx.getScreenView(),
            clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1 },
            loadOp: "clear",
            storeOp: "store"
          }];
          depthStencilAttachment = {
            view: this.ctx.depthTexture.createView(),
            depthClearValue: 1,
            depthLoadOp: "clear",
            depthStoreOp: "store"
          };
        }
        const passEncoder = commandEncoder.beginRenderPass({
          colorAttachments,
          ...depthStencilAttachment ? { depthStencilAttachment } : {}
        });
        let narrowed = false;
        const passSize = node.isMainScreenPass ? [this.ctx.width, this.ctx.height] : node.size ?? [this.ctx.width, this.ctx.height];
        for (const batch of node.batches) {
          if (!batch.isIndirect && batch.currentInstanceCount === 0) continue;
          if (batch.viewport && (batch.viewport[2] <= 0 || batch.viewport[3] <= 0)) continue;
          passEncoder.setPipeline(batch.pipeline);
          passEncoder.setBindGroup(0, batch.bindGroup);
          if (batch.viewport) {
            const [vx, vy, vw, vh] = batch.viewport;
            passEncoder.setViewport(vx, vy, vw, vh, 0, 1);
            passEncoder.setScissorRect(vx, vy, vw, vh);
            narrowed = true;
          } else if (narrowed && passSize) {
            passEncoder.setViewport(0, 0, passSize[0], passSize[1], 0, 1);
            passEncoder.setScissorRect(0, 0, passSize[0], passSize[1]);
            narrowed = false;
          }
          if (batch.customBindGroups) {
            for (const groupIndexStr in batch.customBindGroups) {
              const index = parseInt(groupIndexStr, 10);
              passEncoder.setBindGroup(index, batch.customBindGroups[index]);
            }
          }
          if (batch.vertexBuffer) passEncoder.setVertexBuffer(0, batch.vertexBuffer);
          if (batch.indexBuffer) passEncoder.setIndexBuffer(batch.indexBuffer, batch.indexFormat);
          if (batch.isIndirect && batch.indirectBuffer) {
            passEncoder.drawIndexedIndirect(batch.indirectBuffer, 0);
          } else if (batch.vertexBuffer && batch.indexBuffer) {
            passEncoder.drawIndexed(batch.indexCount, batch.currentInstanceCount, 0, 0, 0);
          } else {
            passEncoder.draw(3, batch.currentInstanceCount, 0, 0);
          }
        }
        passEncoder.end();
      }
    }
    if (activeComputePass) {
      activeComputePass.end();
    }
    this.ctx.device.queue.submit([commandEncoder.finish()]);
  }
};

// src/core/TextureManager.ts
var TextureManager = class {
  device;
  // Cache to prevent loading the exact same texture into VRAM twice
  textureCache = /* @__PURE__ */ new Map();
  samplerCache = /* @__PURE__ */ new Map();
  solidTextures = /* @__PURE__ */ new Map();
  // A default sampler we can reuse for most standard 3D models
  defaultSampler;
  constructor(device) {
    this.device = device;
    this.defaultSampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat"
    });
  }
  /**
   * Fetches an image, decodes it, uploads it to the GPU, and returns a View.
   */
  async load(url, options = {}) {
    const isSRGB = options.sRGB ?? false;
    const cacheKey = `${url}_${isSRGB ? "srgb" : "linear"}`;
    if (this.textureCache.has(cacheKey)) {
      return this.textureCache.get(cacheKey).createView();
    }
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const blob = await response.blob();
      let imageBitmap;
      try {
        imageBitmap = await createImageBitmap(blob, {
          colorSpaceConversion: isSRGB ? "default" : "none"
        });
      } catch (decodeError) {
        imageBitmap = await createImageBitmap(blob);
      }
      const format = isSRGB ? "rgba8unorm-srgb" : "rgba8unorm";
      const texture = this.device.createTexture({
        size: [imageBitmap.width, imageBitmap.height, 1],
        format,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
      });
      this.device.queue.copyExternalImageToTexture(
        { source: imageBitmap },
        { texture },
        [imageBitmap.width, imageBitmap.height]
      );
      this.textureCache.set(cacheKey, texture);
      return texture.createView();
    } catch (error) {
      console.warn(`[TextureManager] Failed to load/decode texture. Using fallback. URL: ${url}`, error);
      return isSRGB ? this.fallbackWhite : this.fallbackWhite;
    }
  }
  /**
   * Gets a custom sampler, caching it based on its stringified properties.
   */
  getSampler(descriptor) {
    const key = JSON.stringify(descriptor);
    if (this.samplerCache.has(key)) {
      return this.samplerCache.get(key);
    }
    const sampler = this.device.createSampler(descriptor);
    this.samplerCache.set(key, sampler);
    return sampler;
  }
  /**
   * Clears all textures from VRAM. Call this when destroying a scene.
   */
  clearCache() {
    this.textureCache.forEach((texture) => texture.destroy());
    this.textureCache.clear();
    this.samplerCache.clear();
  }
  /** Generates a 1x1 solid color texture on the GPU */
  getSolidTexture(r, g, b, a = 255) {
    const key = `${r}_${g}_${b}_${a}`;
    if (this.solidTextures.has(key)) return this.solidTextures.get(key);
    const tex = this.device.createTexture({
      size: [1, 1, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });
    this.device.queue.writeTexture(
      { texture: tex },
      new Uint8Array([r, g, b, a]),
      { bytesPerRow: 4 },
      [1, 1, 1]
    );
    const view = tex.createView();
    this.solidTextures.set(key, view);
    return view;
  }
  get fallbackWhite() {
    return this.getSolidTexture(255, 255, 255, 255);
  }
  get fallbackNormal() {
    return this.getSolidTexture(128, 128, 255, 255);
  }
};

// src/core/NullGraph.ts
var NullGraph = class {
  // Public API Contracts
  device;
  bufferManager;
  textureManager;
  // We expose the raw context as readonly so users can access core buffers
  // like ctx.cameraUniformBuffer for their custom compute stages.
  ctx = new WebGPUContext();
  batchManager;
  renderPipeline;
  // Legacy tracking arrays (Optional in DOD, but kept for memory management / fallback)
  passes = [];
  batches = [];
  computeStages = [];
  /** The canvas being presented to, or `null` when the engine renders headless. */
  canvas = null;
  // Engine State
  isDestroyed = false;
  /**
   * Acquire (or adopt) a device and set up the main-screen target.
   *
   * @param canvas The canvas to present to, or `null` to render headless — the main-screen pass
   *   then draws into {@link screenTexture}, which is how the engine runs in Node (Dawn), in
   *   tests, and in render-to-PNG tooling. `null` needs `options.width` and `options.height`.
   * @param options See {@link EngineInitOptions}. `{ device }` adopts an existing device (e.g.
   *   ONNX Runtime Web's `await ort.env.webgpu.device`) so the two share buffers with no copy.
   *   `samplableDepth` is required by `AerialPerspective.getApplyPass()` when it uses the
   *   engine's own depth buffer.
   */
  async init(canvas, options = {}) {
    try {
      this.canvas = canvas;
      await this.ctx.init(canvas, options);
      this.device = this.ctx.device;
      this.bufferManager = new BufferManager(this.device);
      this.batchManager = new BatchManager(this.ctx);
      this.renderPipeline = new RenderPipeline(this.ctx);
      this.textureManager = new TextureManager(this.device);
      this.isDestroyed = false;
      return {
        success: true,
        enabledFeatures: this.device.features
      };
    } catch (e) {
      console.error("NullGraph Initialization Failed:", e);
      return {
        success: false,
        enabledFeatures: /* @__PURE__ */ new Set(),
        error: e instanceof Error ? e.message : "Unknown Error"
      };
    }
  }
  resize(width, height) {
    this.ctx.resize(width, height);
  }
  get width() {
    return this.ctx.width;
  }
  get height() {
    return this.ctx.height;
  }
  /**
   * The negotiated optional-feature snapshot (subgroups, shader-f16, storage formats,
   * dual-source blending, subgroup sizes). Valid only after a successful {@link init}.
   * See {@link GPUCapabilities}.
   */
  get capabilities() {
    return this.ctx.capabilities;
  }
  /**
   * The offscreen texture the main-screen pass renders into when the engine was initialised
   * without a canvas (`init(null, …)`); `undefined` with a canvas. `RENDER_ATTACHMENT |
   * COPY_SRC`, in `engine.ctx.format` (`'rgba8unorm'` unless `options.format` said otherwise),
   * so a test can `copyTextureToBuffer` the finished frame.
   */
  get screenTexture() {
    return this.ctx.screenTexture;
  }
  clearPasses() {
    this.passes = [];
  }
  clearBatches() {
    this.batches = [];
  }
  // =========================================================
  // PIPELINE NODE CREATION
  // =========================================================
  createPass(config) {
    const pass = new RenderPassNode(config);
    this.passes.push(pass);
    return pass;
  }
  createComputeStage(config) {
    const stage = this.batchManager.createComputeStage(config);
    this.computeStages.push(stage);
    return stage;
  }
  /**
   * The same stage as createComputeStage, compiled with createComputePipelineAsync: frames can go
   * on while it compiles. A shader that does not compile rejects with an error naming the stage.
   */
  async createComputeStageAsync(config) {
    const stage = await this.batchManager.createComputeStageAsync(config);
    this.computeStages.push(stage);
    return stage;
  }
  createBatch(pass, config) {
    const batch = this.batchManager.createBatch(pass, config);
    this.batches.push(batch);
    return batch;
  }
  clearBatch(pass, batch) {
    if (typeof this.batchManager.clearBatch === "function") {
      this.batchManager.clearBatch(pass, batch);
    }
    const globalIndex = this.batches.indexOf(batch);
    if (globalIndex !== -1) {
      this.batches.splice(globalIndex, 1);
    }
  }
  // =========================================================
  // DATA & RESOURCE BINDING
  // =========================================================
  setBatchGeometry(batch, vertexBuffer, indexBuffer, indexCount, format = "uint16") {
    this.batchManager.setBatchGeometry(batch, vertexBuffer, indexBuffer, indexCount, format);
  }
  updateBatchData(batch, rawData, instanceCount) {
    this.batchManager.updateBatchData(batch, rawData, instanceCount);
  }
  createTextureBindGroup(batch, textureView, sampler, extraEntries = [], groupIndex = 1) {
    return this.batchManager.createTextureBindGroup(batch, textureView, sampler, extraEntries, groupIndex);
  }
  /**
   * Binds custom data exclusively to a RenderBatch's graphics pipeline.
   */
  setRenderPassBindGroup(batch, entries, groupIndex = 1) {
    return this.batchManager.setRenderPassBindGroup(batch, entries, groupIndex);
  }
  /**
   * NEW: Binds custom data exclusively to a standalone ComputeStageNode.
   */
  setComputePassBindGroup(stage, entries, groupIndex = 0) {
    return this.batchManager.setComputePassBindGroup(stage, entries, groupIndex);
  }
  updateCamera(camera) {
    this.device.queue.writeBuffer(
      this.ctx.cameraUniformBuffer,
      0,
      camera.bufferData.buffer,
      camera.bufferData.byteOffset,
      camera.bufferData.byteLength
      // <--- USE THIS INSTEAD OF 20 * 4
    );
  }
  //----------
  // =========================================================
  // EXECUTION
  // =========================================================
  /**
   * Executes the WebGPU render loop.
   * @param pipelineSequence An explicit, linear array of compute and render stages.
   */
  render(pipelineSequence) {
    if (this.isDestroyed) return;
    const sequenceToRender = pipelineSequence || this.passes;
    this.renderPipeline.render(sequenceToRender);
  }
};

// src/core/Camera.ts
import { mat4 } from "gl-matrix";
var Camera = class {
  // Core Matrices
  viewProjectionMatrix = mat4.create();
  viewMatrix = mat4.create();
  projectionMatrix = mat4.create();
  // Inverse Matrices (The Future-Proofing)
  inverseViewProjectionMatrix = mat4.create();
  inverseViewMatrix = mat4.create();
  inverseProjectionMatrix = mat4.create();
  // The single, contiguous memory block (108 floats = 432 bytes)
  bufferData = new Float32Array(108);
  fovDegrees;
  near;
  far;
  currentAspect;
  constructor(fovDegrees, aspect, near, far) {
    this.fovDegrees = fovDegrees;
    this.near = near;
    this.far = far;
    this.currentAspect = aspect;
    this.updateProjection(aspect);
  }
  /*
  to update time in the padding slot directly call this,it sets in camera buffer
   */
  updateTime(simTime) {
    this.bufferData[99] = simTime;
  }
  updateProjection(aspect) {
    this.currentAspect = aspect;
    const fovRads = this.fovDegrees * Math.PI / 180;
    mat4.perspectiveZO(this.projectionMatrix, fovRads, aspect, this.near, this.far);
    mat4.invert(this.inverseProjectionMatrix, this.projectionMatrix);
    mat4.multiply(this.viewProjectionMatrix, this.projectionMatrix, this.viewMatrix);
    mat4.invert(this.inverseViewProjectionMatrix, this.viewProjectionMatrix);
    this.bufferData.set(this.viewProjectionMatrix, 0);
    this.bufferData.set(this.projectionMatrix, 32);
    this.bufferData.set(this.inverseViewProjectionMatrix, 48);
    this.bufferData.set(this.inverseProjectionMatrix, 80);
    this.bufferData[100] = this.near;
    this.bufferData[101] = this.far;
    this.bufferData[102] = fovRads;
    this.bufferData[103] = aspect;
  }
  updateView(eye, target, up = [0, 1, 0]) {
    mat4.lookAt(this.viewMatrix, eye, target, up);
    mat4.invert(this.inverseViewMatrix, this.viewMatrix);
    mat4.multiply(this.viewProjectionMatrix, this.projectionMatrix, this.viewMatrix);
    mat4.invert(this.inverseViewProjectionMatrix, this.viewProjectionMatrix);
    this.bufferData.set(this.viewProjectionMatrix, 0);
    this.bufferData.set(this.viewMatrix, 16);
    this.bufferData.set(this.inverseViewProjectionMatrix, 48);
    this.bufferData.set(this.inverseViewMatrix, 64);
    this.bufferData.set(eye, 96);
  }
};
export {
  BatchManager,
  BufferManager,
  Camera,
  ComputeStageNode,
  DISPATCH_INDIRECT_BYTES,
  HEADLESS_SCREEN_FORMAT,
  NullGraph,
  PROGRESSION_FEATURES,
  PROGRESSION_LIMITS,
  RenderBatch,
  RenderPassNode,
  RenderPipeline,
  WebGPUContext
};
//# sourceMappingURL=index.js.map
