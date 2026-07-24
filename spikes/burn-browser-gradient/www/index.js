import init, { run_gradient } from "./pkg/orcacolony_burn_browser_gradient.js";

const button = document.querySelector("#run");
const output = document.querySelector("#output");
const pageParams = new URLSearchParams(location.search);
const fragmentParams = new URLSearchParams(location.hash.slice(1));

function show(message) {
  output.textContent = typeof message === "string" ? message : JSON.stringify(message, null, 2);
}

async function fetchOk(url, kind = "arrayBuffer", options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response[kind]();
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
  const workerId = pageParams.get("worker");
  const workerToken = fragmentParams.get("token");
  const connected = pageParams.has("connected") || workerId !== null;
  show(connected ? "Loading the coordinator assignment…" : "Loading WASM and the M0 fixture…");
  try {
    if (!navigator.gpu) throw new Error("WebGPU is unavailable in this browser");
    const started = performance.now();
    const assignmentUrl = workerId
      ? `/api/v1/assignment?worker_id=${encodeURIComponent(workerId)}`
      : "/api/v1/assignment";
    const manifest = await fetchOk(
      connected ? assignmentUrl : "./fixture/fixture.json",
      "json",
      connected && workerToken
        ? { headers: { "X-Orca-Worker-Token": workerToken } }
        : undefined,
    );
    const modelUrl = connected ? manifest.model_url : "./fixture/model.safetensors";
    const gradientUrl = connected
      ? manifest.oracle_gradient_url
      : "./fixture/gradients.safetensors";
    const [model, expectedGradients] = await Promise.all([
      fetchOk(modelUrl),
      fetchOk(gradientUrl),
      init(),
    ]);
    const [batchSize, sequenceLength] = manifest.input_shape;
    show("Running Burn forward/backward and reading all gradients from WebGPU…");
    const result = await run_gradient(
      new Uint8Array(model),
      Int32Array.from(manifest.input_ids),
      Int32Array.from(manifest.target_ids),
      batchSize,
      sequenceLength,
    );
    const actualGradientBytes = result.gradients();
    const gradientMetrics = compareGradients(expectedGradients, actualGradientBytes.buffer);
    const expectedLossSum = connected ? manifest.expected_loss_sum : manifest.loss_sum;
    const lossAbsoluteError = Math.abs(result.loss_sum - expectedLossSum);
    const lossRelativeError = lossAbsoluteError / Math.abs(expectedLossSum);
    const summary = {
      backend: "Burn 0.21 Autodiff<Wgpu<f32, i32>>",
      mode: connected ? "connected-worker" : "local-parity",
      worker_id: workerId,
      batch_shape: manifest.input_shape,
      model_parameter_count: manifest.parameter_count,
      expected_loss_sum: expectedLossSum,
      browser_loss_sum: result.loss_sum,
      loss_weight_sum: result.loss_weight_sum,
      loss_absolute_error: lossAbsoluteError,
      loss_relative_error: lossRelativeError,
      gradients: gradientMetrics,
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
      };
      if (manifest.lease_token) headers["X-Orca-Lease-Token"] = manifest.lease_token;
      const response = await fetch(manifest.result_url, {
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
    window.orcacolonyResult = summary;
    console.log("ORCACOLONY_RESULT", JSON.stringify(summary));
    show(summary);
    result.free();
  } catch (error) {
    const failure = { error: error instanceof Error ? error.stack || error.message : String(error) };
    window.orcacolonyResult = failure;
    console.error("ORCACOLONY_ERROR", failure.error);
    show(failure);
  } finally {
    button.disabled = false;
  }
}

button.addEventListener("click", run);
if (pageParams.has("autorun") || pageParams.has("connected") || pageParams.has("worker")) run();
