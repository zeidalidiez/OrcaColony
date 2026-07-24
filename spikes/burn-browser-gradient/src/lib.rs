use burn::{
    backend::{
        Autodiff,
        ndarray::{NdArray, NdArrayDevice},
        wgpu::{Wgpu, WgpuDevice, graphics::AutoGraphicsApi, init_setup_async},
    },
    module::{Module, Param},
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{
        Int, Tensor, TensorData,
        backend::{AutodiffBackend, Backend as BackendTrait},
        module::attention,
        ops::AttentionModuleOptions,
    },
};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use js_sys::Uint8Array;
use safetensors::tensor::{Dtype, TensorView, serialize};
use wasm_bindgen::prelude::*;

const LAYER_NORM_EPSILON: f64 = 1.0e-5;

#[derive(Clone, Copy)]
struct ModelSpec {
    vocab_size: usize,
    context_length: usize,
    d_model: usize,
    num_heads: usize,
    num_layers: usize,
    d_ff: usize,
}

type BaseBackend = Wgpu<f32, i32>;
type TrainingBackend = Autodiff<BaseBackend>;
type CpuBaseBackend = NdArray<f32, i32>;
type CpuTrainingBackend = Autodiff<CpuBaseBackend>;

#[derive(Module, Debug)]
struct SelfAttention<B: BackendTrait> {
    qkv: Linear<B>,
    output: Linear<B>,
}

impl<B: BackendTrait> SelfAttention<B> {
    fn new(spec: ModelSpec, device: &B::Device) -> Self {
        Self {
            qkv: LinearConfig::new(spec.d_model, 3 * spec.d_model).init(device),
            output: LinearConfig::new(spec.d_model, spec.d_model).init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>, spec: ModelSpec) -> Tensor<B, 3> {
        let [batch_size, sequence_length, _] = input.dims();
        let qkv = self.qkv.forward(input);
        let chunks = qkv.chunk(3, 2);
        let head_dim = spec.d_model / spec.num_heads;

        let query = chunks[0]
            .clone()
            .reshape([batch_size, sequence_length, spec.num_heads, head_dim])
            .swap_dims(1, 2);
        let key = chunks[1]
            .clone()
            .reshape([batch_size, sequence_length, spec.num_heads, head_dim])
            .swap_dims(1, 2);
        let value = chunks[2]
            .clone()
            .reshape([batch_size, sequence_length, spec.num_heads, head_dim])
            .swap_dims(1, 2);

        let context = attention(
            query,
            key,
            value,
            None,
            None,
            AttentionModuleOptions {
                is_causal: true,
                ..Default::default()
            },
        )
        .swap_dims(1, 2)
        .reshape([batch_size, sequence_length, spec.d_model]);

        self.output.forward(context)
    }
}

#[derive(Module, Debug)]
struct Mlp<B: BackendTrait> {
    input: Linear<B>,
    output: Linear<B>,
}

impl<B: BackendTrait> Mlp<B> {
    fn new(spec: ModelSpec, device: &B::Device) -> Self {
        Self {
            input: LinearConfig::new(spec.d_model, spec.d_ff).init(device),
            output: LinearConfig::new(spec.d_ff, spec.d_model).init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.output.forward(gelu_tanh(self.input.forward(input)))
    }
}

#[derive(Module, Debug)]
struct DecoderBlock<B: BackendTrait> {
    attention_norm: LayerNorm<B>,
    attention: SelfAttention<B>,
    mlp_norm: LayerNorm<B>,
    mlp: Mlp<B>,
}

impl<B: BackendTrait> DecoderBlock<B> {
    fn new(spec: ModelSpec, device: &B::Device) -> Self {
        let norm = || {
            LayerNormConfig::new(spec.d_model)
                .with_epsilon(LAYER_NORM_EPSILON)
                .init(device)
        };
        Self {
            attention_norm: norm(),
            attention: SelfAttention::new(spec, device),
            mlp_norm: norm(),
            mlp: Mlp::new(spec, device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>, spec: ModelSpec) -> Tensor<B, 3> {
        let hidden = input.clone()
            + self
                .attention
                .forward(self.attention_norm.forward(input), spec);
        hidden.clone() + self.mlp.forward(self.mlp_norm.forward(hidden))
    }
}

#[derive(Module, Debug)]
struct VolunteerDecoder<B: BackendTrait> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    blocks: Vec<DecoderBlock<B>>,
    final_norm: LayerNorm<B>,
}

impl<B: BackendTrait> VolunteerDecoder<B> {
    fn new(spec: ModelSpec, device: &B::Device) -> Self {
        Self {
            token_embedding: EmbeddingConfig::new(spec.vocab_size, spec.d_model).init(device),
            position_embedding: EmbeddingConfig::new(spec.context_length, spec.d_model)
                .init(device),
            blocks: (0..spec.num_layers)
                .map(|_| DecoderBlock::new(spec, device))
                .collect(),
            final_norm: LayerNormConfig::new(spec.d_model)
                .with_epsilon(LAYER_NORM_EPSILON)
                .init(device),
        }
    }

    fn forward(&self, input_ids: Tensor<B, 2, Int>, spec: ModelSpec) -> Tensor<B, 3> {
        let [batch_size, sequence_length] = input_ids.dims();
        assert!(sequence_length <= spec.context_length);

        let positions = Tensor::<B, 1, Int>::arange(0..sequence_length as i64, &input_ids.device())
            .reshape([1, sequence_length])
            .repeat_dim(0, batch_size);
        let mut hidden =
            self.token_embedding.forward(input_ids) + self.position_embedding.forward(positions);
        for block in &self.blocks {
            hidden = block.forward(hidden, spec);
        }
        let hidden = self.final_norm.forward(hidden);
        let hidden = hidden.reshape([batch_size * sequence_length, spec.d_model]);
        hidden
            .matmul(self.token_embedding.weight.val().transpose())
            .reshape([batch_size, sequence_length, spec.vocab_size])
    }
}

fn gelu_tanh<B: BackendTrait, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
    let cubic = input.clone() * input.clone() * input.clone();
    let inner = (input.clone() + cubic * 0.044_715) * 0.797_884_560_802_865_4;
    input * 0.5 * (inner.tanh() + 1.0)
}

struct OwnedTensor {
    name: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

async fn push_1d<B>(
    tensors: &mut Vec<OwnedTensor>,
    name: String,
    parameter: &Param<Tensor<B, 1>>,
    grads: &mut B::Gradients,
) -> Result<(), JsValue>
where
    B: AutodiffBackend<FloatElem = f32, IntElem = i32>,
{
    let gradient = parameter
        .val()
        .grad_remove(grads)
        .ok_or_else(|| js_error(format!("missing gradient for {name}")))?;
    let [length] = gradient.dims();
    let data = gradient
        .into_data_async()
        .await
        .map_err(|error| js_error(format!("readback failed for {name}: {error}")))?;
    let values = data
        .to_vec::<f32>()
        .map_err(|error| js_error(format!("decode failed for {name}: {error}")))?;
    tensors.push(OwnedTensor {
        name,
        shape: vec![length],
        bytes: f32_le_bytes(values),
    });
    Ok(())
}

async fn push_2d<B>(
    tensors: &mut Vec<OwnedTensor>,
    name: String,
    parameter: &Param<Tensor<B, 2>>,
    grads: &mut B::Gradients,
    transpose_to_pytorch: bool,
) -> Result<(), JsValue>
where
    B: AutodiffBackend<FloatElem = f32, IntElem = i32>,
{
    let gradient = parameter
        .val()
        .grad_remove(grads)
        .ok_or_else(|| js_error(format!("missing gradient for {name}")))?;
    let gradient = if transpose_to_pytorch {
        gradient.transpose()
    } else {
        gradient
    };
    let [rows, columns] = gradient.dims();
    let data = gradient
        .into_data_async()
        .await
        .map_err(|error| js_error(format!("readback failed for {name}: {error}")))?;
    let values = data
        .to_vec::<f32>()
        .map_err(|error| js_error(format!("decode failed for {name}: {error}")))?;
    tensors.push(OwnedTensor {
        name,
        shape: vec![rows, columns],
        bytes: f32_le_bytes(values),
    });
    Ok(())
}

async fn collect_gradients<B>(
    model: &VolunteerDecoder<B>,
    grads: &mut B::Gradients,
) -> Result<Vec<OwnedTensor>, JsValue>
where
    B: AutodiffBackend<FloatElem = f32, IntElem = i32>,
{
    let mut tensors = Vec::new();
    push_2d(
        &mut tensors,
        "token_embedding.weight".into(),
        &model.token_embedding.weight,
        grads,
        false,
    )
    .await?;
    push_2d(
        &mut tensors,
        "position_embedding.weight".into(),
        &model.position_embedding.weight,
        grads,
        false,
    )
    .await?;

    for (index, block) in model.blocks.iter().enumerate() {
        push_1d(
            &mut tensors,
            format!("blocks.{index}.attention_norm.weight"),
            &block.attention_norm.gamma,
            grads,
        )
        .await?;
        push_1d(
            &mut tensors,
            format!("blocks.{index}.attention_norm.bias"),
            block.attention_norm.beta.as_ref().expect("layer norm bias"),
            grads,
        )
        .await?;
        push_2d(
            &mut tensors,
            format!("blocks.{index}.attention.qkv.weight"),
            &block.attention.qkv.weight,
            grads,
            true,
        )
        .await?;
        push_1d(
            &mut tensors,
            format!("blocks.{index}.attention.qkv.bias"),
            block.attention.qkv.bias.as_ref().expect("linear bias"),
            grads,
        )
        .await?;
        push_2d(
            &mut tensors,
            format!("blocks.{index}.attention.output.weight"),
            &block.attention.output.weight,
            grads,
            true,
        )
        .await?;
        push_1d(
            &mut tensors,
            format!("blocks.{index}.attention.output.bias"),
            block.attention.output.bias.as_ref().expect("linear bias"),
            grads,
        )
        .await?;
        push_1d(
            &mut tensors,
            format!("blocks.{index}.mlp_norm.weight"),
            &block.mlp_norm.gamma,
            grads,
        )
        .await?;
        push_1d(
            &mut tensors,
            format!("blocks.{index}.mlp_norm.bias"),
            block.mlp_norm.beta.as_ref().expect("layer norm bias"),
            grads,
        )
        .await?;
        push_2d(
            &mut tensors,
            format!("blocks.{index}.mlp.0.weight"),
            &block.mlp.input.weight,
            grads,
            true,
        )
        .await?;
        push_1d(
            &mut tensors,
            format!("blocks.{index}.mlp.0.bias"),
            block.mlp.input.bias.as_ref().expect("linear bias"),
            grads,
        )
        .await?;
        push_2d(
            &mut tensors,
            format!("blocks.{index}.mlp.2.weight"),
            &block.mlp.output.weight,
            grads,
            true,
        )
        .await?;
        push_1d(
            &mut tensors,
            format!("blocks.{index}.mlp.2.bias"),
            block.mlp.output.bias.as_ref().expect("linear bias"),
            grads,
        )
        .await?;
    }

    push_1d(
        &mut tensors,
        "final_norm.weight".into(),
        &model.final_norm.gamma,
        grads,
    )
    .await?;
    push_1d(
        &mut tensors,
        "final_norm.bias".into(),
        model.final_norm.beta.as_ref().expect("layer norm bias"),
        grads,
    )
    .await?;
    Ok(tensors)
}

fn serialize_gradients(tensors: &[OwnedTensor]) -> Result<Vec<u8>, JsValue> {
    let views = tensors
        .iter()
        .map(|tensor| {
            let view = TensorView::new(Dtype::F32, tensor.shape.clone(), &tensor.bytes)
                .map_err(|error| js_error(format!("invalid gradient tensor: {error}")))?;
            Ok((tensor.name.as_str(), view))
        })
        .collect::<Result<Vec<_>, JsValue>>()?;
    serialize(views, None).map_err(|error| js_error(format!("safetensors export failed: {error}")))
}

fn f32_le_bytes(values: Vec<f32>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn js_error(message: impl AsRef<str>) -> JsValue {
    JsValue::from_str(message.as_ref())
}

#[wasm_bindgen]
pub struct GradientRun {
    loss_sum: f32,
    loss_weight_sum: u32,
    gradient_bytes: Vec<u8>,
}

#[wasm_bindgen]
impl GradientRun {
    #[wasm_bindgen(getter)]
    pub fn loss_sum(&self) -> f32 {
        self.loss_sum
    }

    #[wasm_bindgen(getter)]
    pub fn loss_weight_sum(&self) -> u32 {
        self.loss_weight_sum
    }

    pub fn gradients(&self) -> Uint8Array {
        Uint8Array::from(self.gradient_bytes.as_slice())
    }
}

async fn run_gradient_with_backend<B>(
    model_bytes: Vec<u8>,
    input_ids: Vec<i32>,
    target_ids: Vec<i32>,
    batch_size: usize,
    sequence_length: usize,
    spec: ModelSpec,
    device: &B::Device,
) -> Result<GradientRun, JsValue>
where
    B: AutodiffBackend<FloatElem = f32, IntElem = i32>,
{
    if spec.vocab_size == 0
        || spec.context_length == 0
        || spec.d_model == 0
        || spec.num_heads == 0
        || spec.num_layers == 0
        || spec.d_ff == 0
        || spec.d_model % spec.num_heads != 0
    {
        return Err(js_error("model dimensions are invalid"));
    }
    if sequence_length > spec.context_length {
        return Err(js_error("sequence length exceeds model context length"));
    }
    let loss_weight_sum = batch_size
        .checked_mul(sequence_length)
        .ok_or_else(|| js_error("batch dimensions overflow"))?;
    if input_ids.len() != loss_weight_sum || target_ids.len() != loss_weight_sum {
        return Err(js_error(
            "fixture tensor lengths do not match batch dimensions",
        ));
    }

    let mut model = VolunteerDecoder::<B>::new(spec, device);
    let mut store = SafetensorsStore::from_bytes(Some(model_bytes))
        .with_from_adapter(PyTorchToBurnAdapter)
        .with_key_remapping(r"^blocks\.(\d+)\.mlp\.0\.", "blocks.$1.mlp.input.")
        .with_key_remapping(r"^blocks\.(\d+)\.mlp\.2\.", "blocks.$1.mlp.output.");
    model
        .load_from(&mut store)
        .map_err(|error| js_error(format!("checkpoint load failed: {error}")))?;

    let inputs = Tensor::<B, 2, Int>::from_data(
        TensorData::new(input_ids, [batch_size, sequence_length]),
        device,
    );
    let targets =
        Tensor::<B, 1, Int>::from_data(TensorData::new(target_ids, [loss_weight_sum]), device);
    let logits = model
        .forward(inputs, spec)
        .reshape([loss_weight_sum, spec.vocab_size]);
    let log_probabilities = burn::tensor::activation::log_softmax(logits, 1);
    let selected = log_probabilities.gather(1, targets.reshape([loss_weight_sum, 1]));
    let loss = selected.sum().neg();
    let loss_sum = loss
        .clone()
        .into_scalar_async()
        .await
        .map_err(|error| js_error(format!("loss readback failed: {error}")))?;
    let mut grads = loss.backward();
    let tensors = collect_gradients(&model, &mut grads).await?;
    let gradient_bytes = serialize_gradients(&tensors)?;

    Ok(GradientRun {
        loss_sum,
        loss_weight_sum: loss_weight_sum as u32,
        gradient_bytes,
    })
}

#[wasm_bindgen]
pub async fn run_gradient(
    model_bytes: Vec<u8>,
    input_ids: Vec<i32>,
    target_ids: Vec<i32>,
    batch_size: usize,
    sequence_length: usize,
    vocab_size: usize,
    context_length: usize,
    d_model: usize,
    num_heads: usize,
    num_layers: usize,
    d_ff: usize,
) -> Result<GradientRun, JsValue> {
    console_error_panic_hook::set_once();
    let device = WgpuDevice::default();
    init_setup_async::<AutoGraphicsApi>(&device, Default::default()).await;
    run_gradient_with_backend::<TrainingBackend>(
        model_bytes,
        input_ids,
        target_ids,
        batch_size,
        sequence_length,
        ModelSpec {
            vocab_size,
            context_length,
            d_model,
            num_heads,
            num_layers,
            d_ff,
        },
        &device,
    )
    .await
}

#[wasm_bindgen]
pub async fn run_gradient_cpu(
    model_bytes: Vec<u8>,
    input_ids: Vec<i32>,
    target_ids: Vec<i32>,
    batch_size: usize,
    sequence_length: usize,
    vocab_size: usize,
    context_length: usize,
    d_model: usize,
    num_heads: usize,
    num_layers: usize,
    d_ff: usize,
) -> Result<GradientRun, JsValue> {
    console_error_panic_hook::set_once();
    let device = NdArrayDevice::Cpu;
    run_gradient_with_backend::<CpuTrainingBackend>(
        model_bytes,
        input_ids,
        target_ids,
        batch_size,
        sequence_length,
        ModelSpec {
            vocab_size,
            context_length,
            d_model,
            num_heads,
            num_layers,
            d_ff,
        },
        &device,
    )
    .await
}
