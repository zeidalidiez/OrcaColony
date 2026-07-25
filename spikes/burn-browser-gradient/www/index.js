import init, {
  run_gradient,
  run_gradient_cpu,
  run_lora_gradient,
  run_lora_gradient_cpu,
} from "./pkg/orcacolony_burn_browser_gradient.js";

const button = document.querySelector("#run");
const output = document.querySelector("#output");
const pageParams = new URLSearchParams(location.search);
const fragmentParams = new URLSearchParams(location.hash.slice(1));
const pinnedCoordinator = document
  .querySelector('meta[name="orcacolony-coordinator"]')
  .content.trim();
const pinnedCampaign = document
  .querySelector('meta[name="orcacolony-campaign"]')
  .content.trim();
const configuredCoordinator = pinnedCoordinator || pageParams.get("coordinator");
const coordinatorBase = new URL(configuredCoordinator || location.origin);
if (!new Set(["http:", "https:"]).has(coordinatorBase.protocol)) {
  throw new Error("coordinator must use HTTP or HTTPS");
}

function coordinatorUrl(path) {
  const url = new URL(path, coordinatorBase);
  if (url.origin !== coordinatorBase.origin) {
    throw new Error("coordinator API URLs must remain on the pinned origin");
  }
  return url.href;
}

function workerIdFromPage() {
  return (
    pageParams.get("worker") ??
    pageParams.get("loop") ??
    pageParams.get("cpu") ??
    pageParams.get("cpu-loop")
  );
}

function setText(selector, value) {
  document.querySelector(selector).textContent = value;
}

function formatInteger(value) {
  return new Intl.NumberFormat().format(Number(value));
}

function formatBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const index = Math.min(
    units.length - 1,
    Math.floor(Math.log(Math.max(bytes, 1)) / Math.log(1024)),
  );
  const scaled = bytes / 1024 ** index;
  return `${scaled.toFixed(index === 0 || scaled >= 100 ? 0 : 1)} ${units[index]}`;
}

function shortHash(value) {
  return value ? `${String(value).slice(0, 10)}…${String(value).slice(-6)}` : "—";
}

function replaceChildrenWithText(container, text, className = "note") {
  const element = document.createElement("span");
  element.className = className;
  element.textContent = text;
  container.replaceChildren(element);
}

function renderEvaluation(entries) {
  const chart = document.querySelector("#evaluation-chart");
  const empty = document.querySelector("#evaluation-empty");
  if (!entries.length) {
    chart.hidden = true;
    empty.hidden = false;
    return;
  }
  empty.hidden = true;
  chart.hidden = false;
  const losses = entries.map((entry) => Number(entry.mean_loss));
  const minimum = Math.min(...losses);
  const maximum = Math.max(...losses);
  const span = Math.max(maximum - minimum, Math.max(maximum, 1) * 0.02);
  const xFor = (index) => 28 + (entries.length === 1 ? 296 : (index * 592) / (entries.length - 1));
  const yFor = (loss) => 24 + ((maximum + span * 0.15 - loss) * 136) / (span * 1.3);
  document.querySelector("#evaluation-line").setAttribute(
    "points",
    entries.map((entry, index) => `${xFor(index)},${yFor(Number(entry.mean_loss))}`).join(" "),
  );
  const pointGroup = document.querySelector("#evaluation-points");
  pointGroup.replaceChildren();
  const svgNamespace = "http://www.w3.org/2000/svg";
  entries.forEach((entry, index) => {
    const point = document.createElementNS(svgNamespace, "circle");
    point.setAttribute("class", "chart-point");
    point.setAttribute("cx", xFor(index));
    point.setAttribute("cy", yFor(Number(entry.mean_loss)));
    point.setAttribute("r", 4);
    const title = document.createElementNS(svgNamespace, "title");
    title.textContent = `Step ${entry.step}: loss ${Number(entry.mean_loss).toFixed(5)}`;
    point.append(title);
    pointGroup.append(point);
  });
}

function renderDashboard(dashboard) {
  if (pinnedCampaign && dashboard.campaign.id !== pinnedCampaign) {
    throw new Error("coordinator campaign does not match this static worker release");
  }
  const {
    campaign,
    checkpoint,
    contributors,
    dataset,
    evaluation_gate: evaluationGate,
    evaluations,
    model,
    progress,
    public_ledger: ledger,
    resource_observations: resourceObservations,
  } = dashboard;
  setText("#campaign-id", campaign.id);
  setText("#campaign-state", campaign.state.replaceAll("_", " "));
  setText("#steps-value", `${progress.completed_steps} / ${progress.target_steps}`);
  document.querySelector("#steps-progress").style.width = `${Math.min(100, (100 * progress.completed_steps) / Math.max(1, progress.target_steps))}%`;
  setText("#tokens-value", formatInteger(progress.accepted_tokens));
  setText("#assignments-detail", `${formatInteger(progress.accepted_assignments)} of ${formatInteger(progress.target_assignments)} assignments accepted`);
  setText("#contributors-value", formatInteger(contributors.active_count));
  setText("#contributors-detail", `${contributors.anonymous_count} contributing privately`);
  const runtimeSeconds = resourceObservations?.runtime_seconds?.gradient_compute ?? 0;
  const transfer = resourceObservations?.transfer_bytes ?? {};
  const totalTransferBytes = Object.values(transfer).reduce(
    (total, value) => total + Number(value ?? 0),
    0,
  );
  const memory = resourceObservations?.memory_bytes ?? {};
  const peakMemoryBytes = Math.max(
    Number(memory.peak_wasm_linear ?? 0),
    Number(memory.peak_process_rss ?? 0),
    Number(memory.peak_js_heap_used ?? 0),
  );
  setText("#compute-value", `${Number(runtimeSeconds).toFixed(2)} s`);
  setText(
    "#compute-detail",
    `${formatInteger(resourceObservations?.worker_reports ?? 0)} measured assignments`,
  );
  setText("#transfer-value", formatBytes(totalTransferBytes));
  setText("#memory-value", formatBytes(peakMemoryBytes));
  setText(
    "#memory-detail",
    peakMemoryBytes ? "Peak observed worker allocation" : "Worker API unavailable",
  );
  setText(
    "#storage-value",
    formatBytes(resourceObservations?.coordinator_storage_bytes ?? 0),
  );

  const lastEvaluation = evaluations.at(-1);
  setText("#loss-value", lastEvaluation ? Number(lastEvaluation.mean_loss).toFixed(4) : "—");
  setText(
    "#loss-detail",
    lastEvaluation
      ? `Step ${lastEvaluation.step} · perplexity ${Number(lastEvaluation.perplexity).toFixed(1)}`
      : "Frozen held-out profile",
  );
  setText("#model-title", `${model.layers}-layer causal transformer`);
  setText("#model-parameters", formatInteger(model.parameter_count));
  setText("#model-architecture", `${model.width}d · ${model.heads} heads · ${model.context_length} ctx`);
  setText("#dataset-name", dataset.source.dataset ?? "Frozen dataset");
  const licenseLink = document.querySelector("#dataset-license");
  licenseLink.textContent = dataset.source.license ?? "Project fixture";
  licenseLink.removeAttribute("href");
  if (dataset.source.license_url) {
    try {
      const licenseUrl = new URL(dataset.source.license_url);
      if (new Set(["http:", "https:"]).has(licenseUrl.protocol)) {
        licenseLink.href = licenseUrl.href;
      }
    } catch {
      // The frozen license identifier remains visible even if its URL is malformed.
    }
  }
  setText("#dataset-revision", shortHash(dataset.revision));
  document.querySelector("#dataset-revision").title = dataset.revision;
  setText("#checkpoint-sha", shortHash(checkpoint.sha256));
  document.querySelector("#checkpoint-sha").title = checkpoint.sha256 ?? "";
  const checkpointDownload = document.querySelector("#checkpoint-download");
  checkpointDownload.hidden = !checkpoint.download_url;
  if (checkpoint.download_url) {
    checkpointDownload.href = coordinatorUrl(checkpoint.download_url);
  }

  const statusDot = document.querySelector("#status-dot");
  statusDot.classList.add("live");
  setText("#connection-status", campaign.state === "campaign_complete" ? "Campaign complete" : "Coordinator live");
  if (!workerIdFromPage() || campaign.state === "campaign_complete") {
    button.disabled = true;
    button.textContent =
      campaign.state === "campaign_complete"
        ? "Campaign complete"
        : "Approved worker link required";
  }
  renderEvaluation(evaluations);
  const evaluationGateElement = document.querySelector("#evaluation-gate");
  evaluationGateElement.hidden = !evaluationGate;
  if (evaluationGate) {
    const observed = evaluationGate.observed_improvement_from_initialization;
    const observedText =
      observed === undefined ? "awaiting checkpoints" : `${observed.toFixed(4)} observed`;
    evaluationGateElement.textContent =
      `Declared gate: ${evaluationGate.state} · ${observedText} · ` +
      `${evaluationGate.minimum_improvement_from_initialization.toFixed(4)} required`;
  }

  const acknowledgements = document.querySelector("#acknowledgements");
  acknowledgements.replaceChildren();
  for (const acknowledgement of contributors.acknowledgements) {
    const pill = document.createElement("span");
    pill.className = "ack";
    pill.textContent = `${acknowledgement.display_name} · ${formatInteger(acknowledgement.accepted_tokens)} tokens`;
    acknowledgements.append(pill);
  }
  if (contributors.anonymous_count > 0) {
    const pill = document.createElement("span");
    pill.className = "ack";
    pill.textContent = `${contributors.anonymous_count} private contributor${contributors.anonymous_count === 1 ? "" : "s"}`;
    acknowledgements.append(pill);
  }
  if (!acknowledgements.children.length) {
    replaceChildrenWithText(acknowledgements, "No accepted contributions yet.");
  }

  const ledgerElement = document.querySelector("#ledger");
  ledgerElement.replaceChildren();
  for (const entry of ledger.slice(-12).reverse()) {
    const row = document.createElement("div");
    row.className = "list-row";
    const identity = document.createElement("strong");
    identity.textContent = entry.credit;
    const amount = document.createElement("span");
    amount.textContent = `${formatInteger(entry.accepted_tokens)} tokens · step ${entry.checkpoint_step}`;
    const runtime = document.createElement("span");
    runtime.textContent = entry.runtime_backend;
    row.append(identity, amount, runtime);
    ledgerElement.append(row);
  }
  if (!ledgerElement.children.length) {
    replaceChildrenWithText(ledgerElement, "No accepted work yet.", "empty");
  }
}

async function loadDashboard() {
  try {
    const dashboard = await fetchOk(coordinatorUrl("/api/v1/dashboard"), "json");
    renderDashboard(dashboard);
    window.orcacolonyDashboard = dashboard;
  } catch (error) {
    document.querySelector("#status-dot").classList.remove("live");
    setText("#connection-status", "Coordinator unavailable");
  }
}

function show(message) {
  output.textContent = typeof message === "string" ? message : JSON.stringify(message, null, 2);
}

async function fetchOk(url, kind = "arrayBuffer", options) {
  return (await fetchMeasured(url, kind, options)).value;
}

async function fetchMeasured(url, kind = "arrayBuffer", options) {
  const started = performance.now();
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.text();
    const error = new Error(`${url}: HTTP ${response.status}: ${body}`);
    error.status = response.status;
    error.responseBody = body;
    throw error;
  }
  const body = await response.arrayBuffer();
  const value =
    kind === "json"
      ? JSON.parse(new TextDecoder().decode(body))
      : body;
  return {
    value,
    bytes: body.byteLength,
    seconds: (performance.now() - started) / 1000,
  };
}

function optionalByteCount(value) {
  return Number.isSafeInteger(value) && value >= 0 ? value : null;
}

function parseSafetensors(buffer) {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLength = Number(view.getBigUint64(0, true));
  const headerText = new TextDecoder().decode(bytes.subarray(8, 8 + headerLength));
  const header = JSON.parse(headerText);
  const dataStart = 8 + headerLength;
  const tensors = new Map();

  for (const [name, descriptor] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (descriptor.dtype !== "F32") throw new Error(`${name}: expected F32, got ${descriptor.dtype}`);
    const [start, end] = descriptor.data_offsets;
    const raw = bytes.slice(dataStart + start, dataStart + end);
    tensors.set(name, {
      shape: descriptor.shape,
      values: new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4),
    });
  }
  return tensors;
}

function compareGradients(expectedBuffer, actualBuffer) {
  const expected = parseSafetensors(expectedBuffer);
  const actual = parseSafetensors(actualBuffer);
  const expectedNames = [...expected.keys()].sort();
  const actualNames = [...actual.keys()].sort();
  if (JSON.stringify(expectedNames) !== JSON.stringify(actualNames)) {
    throw new Error(`gradient names differ: expected ${expectedNames.length}, actual ${actualNames.length}`);
  }

  let squaredError = 0;
  let squaredExpected = 0;
  let dot = 0;
  let squaredActual = 0;
  let maxAbsoluteError = 0;
  let valueCount = 0;
  let worstTensor = null;

  for (const name of expectedNames) {
    const left = expected.get(name);
    const right = actual.get(name);
    if (JSON.stringify(left.shape) !== JSON.stringify(right.shape)) {
      throw new Error(`${name}: shape mismatch ${left.shape} vs ${right.shape}`);
    }
    if (left.values.length !== right.values.length) throw new Error(`${name}: length mismatch`);

    let tensorMax = 0;
    for (let index = 0; index < left.values.length; index += 1) {
      const expectedValue = left.values[index];
      const actualValue = right.values[index];
      if (!Number.isFinite(actualValue)) throw new Error(`${name}[${index}] is ${actualValue}`);
      const difference = actualValue - expectedValue;
      const absolute = Math.abs(difference);
      tensorMax = Math.max(tensorMax, absolute);
      maxAbsoluteError = Math.max(maxAbsoluteError, absolute);
      squaredError += difference * difference;
      squaredExpected += expectedValue * expectedValue;
      squaredActual += actualValue * actualValue;
      dot += expectedValue * actualValue;
      valueCount += 1;
    }
    if (!worstTensor || tensorMax > worstTensor.max_absolute_error) {
      worstTensor = { name, max_absolute_error: tensorMax };
    }
  }

  return {
    tensor_count: expectedNames.length,
    value_count: valueCount,
    cosine_similarity: dot / Math.sqrt(squaredExpected * squaredActual),
    relative_l2_error: Math.sqrt(squaredError / squaredExpected),
    root_mean_square_error: Math.sqrt(squaredError / valueCount),
    max_absolute_error: maxAbsoluteError,
    worst_tensor: worstTensor,
  };
}

async function run() {
  button.disabled = true;
  const continuous = pageParams.has("loop") || pageParams.has("cpu-loop");
  const cpuWorkerId = pageParams.get("cpu") ?? pageParams.get("cpu-loop");
  const workerId = workerIdFromPage();
  const workerToken = fragmentParams.get("token");
  const connected = pageParams.has("connected") || workerId !== null;
  if (connected && coordinatorBase.origin !== location.origin && !pinnedCoordinator) {
    throw new Error(
      "remote worker execution requires an operator-pinned coordinator origin",
    );
  }
  const requestedBackend =
    (cpuWorkerId ? "cpu" : pageParams.get("backend")) ??
    fragmentParams.get("backend") ??
    (navigator.gpu ? "webgpu" : "cpu");
  if (!new Set(["cpu", "webgpu"]).has(requestedBackend)) {
    throw new Error(`unsupported backend: ${requestedBackend}`);
  }
  show(connected ? "Loading the coordinator assignment…" : "Loading WASM and the M0 fixture…");
  let runAgain = false;
  try {
    if (requestedBackend === "webgpu" && !navigator.gpu) {
      throw new Error("WebGPU is unavailable in this browser");
    }
    const started = performance.now();
    const assignmentUrl = workerId
      ? coordinatorUrl(`/api/v1/assignment?worker_id=${encodeURIComponent(workerId)}`)
      : coordinatorUrl("/api/v1/assignment");
    const assignmentMeasurement = await fetchMeasured(
      connected ? assignmentUrl : "./fixture/fixture.json",
      "json",
      connected && workerToken
        ? { headers: { "X-Orca-Worker-Token": workerToken } }
        : undefined,
    );
    const manifest = assignmentMeasurement.value;
    if (connected && pinnedCampaign && manifest.campaign_id !== pinnedCampaign) {
      throw new Error("assignment campaign does not match this static worker release");
    }
    const peftMode =
      manifest.format === "orcacolony_lora_fixture_v1" ||
      manifest.training_method === "frozen-base-lora";
    const modelUrl = connected
      ? coordinatorUrl(manifest.model_url)
      : peftMode
        ? "./fixture/base.safetensors"
        : "./fixture/model.safetensors";
    const gradientUrl = connected
      ? coordinatorUrl(manifest.oracle_gradient_url)
      : "./fixture/gradients.safetensors";
    const wasmInitStarted = performance.now();
    const wasmRuntime = await init();
    const wasmInitSeconds = (performance.now() - wasmInitStarted) / 1000;
    const artifactFetchStarted = performance.now();
    const adapterPromise = peftMode
      ? fetchMeasured(
          connected
            ? coordinatorUrl(manifest.adapter_url)
            : "./fixture/adapter.safetensors",
        )
      : Promise.resolve({ value: null, bytes: 0, seconds: 0 });
    const [modelMeasurement, gradientMeasurement, adapterMeasurement] = await Promise.all([
      fetchMeasured(modelUrl),
      fetchMeasured(gradientUrl),
      adapterPromise,
    ]);
    const artifactFetchSeconds = (performance.now() - artifactFetchStarted) / 1000;
    const model = modelMeasurement.value;
    const expectedGradients = gradientMeasurement.value;
    const adapter = adapterMeasurement.value;
    if (connected) {
      const resources = manifest.resource_profile;
      if (
        modelMeasurement.bytes !== resources.model_download_bytes ||
        adapterMeasurement.bytes !== resources.adapter_download_bytes ||
        gradientMeasurement.bytes !== resources.oracle_gradient_download_bytes
      ) {
        throw new Error("downloaded artifact sizes do not match the assignment resource profile");
      }
    }
    const [batchSize, sequenceLength] = manifest.input_shape;
    const modelSpec =
      manifest.model ??
      {
        vocab_size: 4096,
        context_length: 128,
        d_model: 128,
        num_heads: 2,
        num_layers: 4,
        d_ff: 512,
      };
    show(
      `Running Burn ${peftMode ? "LoRA " : ""}forward/backward and reading ` +
        `${peftMode ? "adapter" : "all"} gradients from ${requestedBackend}…`,
    );
    const gradientComputeStarted = performance.now();
    let result;
    if (peftMode) {
      const runLoraGradient =
        requestedBackend === "cpu" ? run_lora_gradient_cpu : run_lora_gradient;
      result = await runLoraGradient(
        new Uint8Array(model),
        new Uint8Array(adapter),
        Int32Array.from(manifest.input_ids),
        Int32Array.from(manifest.target_ids),
        batchSize,
        sequenceLength,
        modelSpec.vocab_size,
        modelSpec.context_length,
        modelSpec.d_model,
        modelSpec.num_heads,
        modelSpec.num_layers,
        modelSpec.d_ff,
        manifest.adapter.rank,
        manifest.adapter.alpha,
      );
    } else {
      const runGradient = requestedBackend === "cpu" ? run_gradient_cpu : run_gradient;
      result = await runGradient(
        new Uint8Array(model),
        Int32Array.from(manifest.input_ids),
        Int32Array.from(manifest.target_ids),
        batchSize,
        sequenceLength,
        modelSpec.vocab_size,
        modelSpec.context_length,
        modelSpec.d_model,
        modelSpec.num_heads,
        modelSpec.num_layers,
        modelSpec.d_ff,
      );
    }
    const gradientComputeSeconds = (performance.now() - gradientComputeStarted) / 1000;
    const actualGradientBytes = result.gradients();
    const gradientMetrics = compareGradients(expectedGradients, actualGradientBytes.buffer);
    const expectedLossSum = connected ? manifest.expected_loss_sum : manifest.loss_sum;
    const lossAbsoluteError = Math.abs(result.loss_sum - expectedLossSum);
    const lossRelativeError = lossAbsoluteError / Math.abs(expectedLossSum);
    const memory = performance.memory;
    const workerTelemetry = {
      format: "orcacolony_worker_telemetry_v1",
      runtime_seconds: {
        assignment_fetch: assignmentMeasurement.seconds,
        runtime_init: wasmInitSeconds,
        artifact_fetch: artifactFetchSeconds,
        gradient_compute: gradientComputeSeconds,
      },
      transfer_bytes: {
        assignment: assignmentMeasurement.bytes,
        model: modelMeasurement.bytes,
        adapter: adapterMeasurement.bytes,
        oracle_gradient: gradientMeasurement.bytes,
        result: actualGradientBytes.byteLength,
      },
      memory_bytes: {
        wasm_linear: optionalByteCount(wasmRuntime?.memory?.buffer?.byteLength),
        process_peak_rss: null,
        js_heap_used: optionalByteCount(memory?.usedJSHeapSize),
        js_heap_limit: optionalByteCount(memory?.jsHeapSizeLimit),
        device_capacity: optionalByteCount(
          navigator.deviceMemory === undefined
            ? null
            : navigator.deviceMemory * 1024 * 1024 * 1024,
        ),
      },
    };
    const summary = {
      backend:
        requestedBackend === "cpu"
          ? "Burn 0.21 Autodiff<NdArray<f32, i32>>"
          : "Burn 0.21 Autodiff<Wgpu<f32, i32>>",
      mode: connected ? "connected-worker" : "local-parity",
      training_method: peftMode ? "frozen-base-lora" : "dense",
      worker_id: workerId,
      batch_shape: manifest.input_shape,
      model_parameter_count: peftMode
        ? (manifest.base?.parameter_count ?? manifest.parameter_count)
        : manifest.parameter_count,
      trainable_parameter_count: peftMode
        ? manifest.adapter.value_count
        : manifest.parameter_count,
      model: modelSpec,
      expected_loss_sum: expectedLossSum,
      browser_loss_sum: result.loss_sum,
      loss_weight_sum: result.loss_weight_sum,
      loss_absolute_error: lossAbsoluteError,
      loss_relative_error: lossRelativeError,
      gradients: gradientMetrics,
      resource_observation: workerTelemetry,
      elapsed_ms: Math.round(performance.now() - started),
    };
    summary.provisional_parity =
      summary.loss_relative_error <= 0.002 &&
      gradientMetrics.cosine_similarity >= 0.999 &&
      gradientMetrics.relative_l2_error <= 0.01;
    if (connected) {
      show("Uploading the complete gradient to the coordinator…");
      const headers = {
        "Content-Type": "application/octet-stream",
        "X-Orca-Checkpoint-Sha256": manifest.checkpoint_sha256,
        "X-Orca-Loss-Sum": String(result.loss_sum),
        "X-Orca-Loss-Weight-Sum": String(result.loss_weight_sum),
        "X-Orca-Runtime-Backend":
          requestedBackend === "cpu" ? "burn-ndarray-f32" : "burn-webgpu-f32",
        "X-Orca-Worker-Telemetry": JSON.stringify(workerTelemetry),
      };
      if (manifest.lease_token) headers["X-Orca-Lease-Token"] = manifest.lease_token;
      const response = await fetch(coordinatorUrl(manifest.result_url), {
        method: "POST",
        headers,
        body: actualGradientBytes,
      });
      const receipt = await response.json();
      if (!response.ok) throw new Error(`coordinator rejected result: ${receipt.error}`);
      summary.coordinator = receipt;
      summary.connected_step_complete =
        receipt.step_complete ?? (receipt.accepted === true && receipt.step === 1);
    }
    if (continuous) {
      const campaignStatus = await fetchOk(coordinatorUrl("/api/v1/status"), "json");
      summary.campaign = campaignStatus;
      summary.campaign_complete = campaignStatus.state === "campaign_complete";
      runAgain = !summary.campaign_complete;
    }
    window.orcacolonyResult = summary;
    console.log("ORCACOLONY_RESULT", JSON.stringify(summary));
    show(summary);
    await loadDashboard();
    result.free();
  } catch (error) {
    const retryableConflict =
      continuous &&
      error?.status === 409 &&
      /no assignment available|campaign target is complete/.test(error.responseBody ?? "");
    if (retryableConflict) {
      const campaignStatus = await fetchOk(coordinatorUrl("/api/v1/status"), "json");
      const waiting = {
        waiting_for_assignment: campaignStatus.state !== "campaign_complete",
        campaign_complete: campaignStatus.state === "campaign_complete",
        campaign: campaignStatus,
      };
      window.orcacolonyResult = waiting;
      show(waiting);
      runAgain = !waiting.campaign_complete;
      return;
    }
    const failure = { error: error instanceof Error ? error.stack || error.message : String(error) };
    window.orcacolonyResult = failure;
    console.error("ORCACOLONY_ERROR", failure.error);
    show(failure);
  } finally {
    button.disabled = false;
    if (runAgain) window.setTimeout(run, 1000);
  }
}

button.addEventListener("click", run);
loadDashboard();
window.setInterval(loadDashboard, 3000);
if (
  pageParams.has("autorun") ||
  pageParams.has("connected") ||
  pageParams.has("worker") ||
  pageParams.has("cpu") ||
  pageParams.has("loop") ||
  pageParams.has("cpu-loop")
) run();
