// Trainer-owned lifecycle: errors cannot be turned into successful artifacts by a caller's logger.
import { appendLog, writeStatus } from './coevolve-checkpoint.mjs';

export async function managedRun(config, body) {
    const resources = [];
    let completedUpdates = 0, result, failure;
    const own = resource => { resources.push(resource); return resource; };
    writeStatus(config.targetDir, { state: 'incomplete', completedUpdates });
    try {
        result = await body({ own, progress: n => { completedUpdates = n; } });
    } catch (error) { failure = error; }
    finally {
        for (const resource of resources.reverse()) {
            try { await resource.destroy(); }
            catch (error) { failure ??= error; }
        }
    }
    if (failure) {
        const record = { state: 'incomplete', completedUpdates, error: String(failure.stack ?? failure) };
        try { appendLog(config.targetDir, { event: 'failure', ...record }); writeStatus(config.targetDir, record); }
        catch (error) { throw new AggregateError([failure, error], 'coevolve: run and failure persistence failed'); }
        throw failure;
    }
    writeStatus(config.targetDir, { state: 'complete', completedUpdates });
    return result;
}

export function watchDevice(device) {
    const errors = [];
    const listener = event => errors.push(String(event.error?.message ?? event.error));
    device.addEventListener('uncapturederror', listener);
    device.lost.then(info => {
        if (info.reason !== 'destroyed') errors.push(`device lost (${info.reason}): ${info.message}`);
    }, error => errors.push(String(error)));
    return {
        errors,
        async check({ final = false } = {}) {
            await device.queue.onSubmittedWorkDone();
            // Error event delivery is asynchronous, including on an externally supplied device.
            await new Promise(resolve => setTimeout(resolve, final ? 50 : 0));
            if (errors.length) throw new Error(`coevolve: WebGPU failure: ${errors.join('; ')}`);
        },
        destroy() { device.removeEventListener('uncapturederror', listener); },
    };
}

/** Resolve pipeline compilation without executing an unscored simulation step. */
export async function buildCompiled(device, build) {
    const original = device.createComputePipeline;
    const pending = [];
    device.createComputePipeline = function (descriptor) {
        const pipeline = original.call(this, descriptor);
        pending.push(this.createComputePipelineAsync(descriptor));
        return pipeline;
    };
    let value, error;
    const start = performance.now();
    try { value = build(); }
    catch (caught) { error = caught; }
    finally { device.createComputePipeline = original; }
    const compiled = await Promise.allSettled(pending);
    error ??= compiled.find(result => result.status === 'rejected')?.reason;
    if (error) { value?.destroy(); throw error; }
    return { value, setupCompileMs: performance.now() - start };
}