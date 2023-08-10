use derives::*;
use dfdx::{
    dtypes::Dtype,
    shapes::Dim,
    tensor::{HasErr, PutTape, SplitTape, WithEmptyTape},
    tensor_ops::{Device, TryAdd},
};

use crate::layer_norm1d::{DeviceLayerNorm1D, LayerNorm1D};
use crate::linear::Linear;
use crate::multi_head_attention::{DeviceMultiHeadAttention, MultiHeadAttention};
use crate::relu::ReLU;
use crate::residual_add::ResidualAdd;

#[derive(Clone, Debug, Sequential)]
pub struct FeedForward<Model: Dim, F: Dim> {
    pub l1: Linear<Model, F>,
    pub act1: ReLU,
    pub l2: Linear<F, Model>,
}

#[derive(Clone, Debug, Sequential)]
pub struct EncoderBlock<Model: Dim, NumHeads: Dim, F: Dim> {
    pub self_attn: ResidualAdd<MultiHeadAttention<Model, NumHeads>>,
    pub norm1: LayerNorm1D<Model>,
    pub ff: ResidualAdd<FeedForward<Model, F>>,
    pub norm2: LayerNorm1D<Model>,
}

impl<Model: Dim, NumHeads: Dim, F: Dim> EncoderBlock<Model, NumHeads, F> {
    pub fn new(model: Model, num_heads: NumHeads, f: F) -> Self {
        EncoderBlock {
            self_attn: ResidualAdd(MultiHeadAttention::new(model, num_heads, model, model)),
            norm1: LayerNorm1D(model),
            ff: ResidualAdd(FeedForward {
                l1: Linear::new(model, f),
                act1: ReLU,
                l2: Linear::new(f, model),
            }),
            norm2: LayerNorm1D(model),
        }
    }
}

#[derive(Clone, Debug, CustomModule)]
pub struct DecoderBlock<Model: Dim, NumHeads: Dim, F: Dim> {
    #[module]
    pub self_attn: ResidualAdd<MultiHeadAttention<Model, NumHeads>>,
    #[module]
    pub norm1: LayerNorm1D<Model>,
    #[module]
    pub mh_attn: MultiHeadAttention<Model, NumHeads>,
    #[module]
    pub norm2: LayerNorm1D<Model>,
    #[module]
    pub ff: ResidualAdd<FeedForward<Model, F>>,
    #[module]
    pub norm3: LayerNorm1D<Model>,
}

impl<Model: Dim, NumHeads: Dim, F: Dim> DecoderBlock<Model, NumHeads, F> {
    pub fn new(model: Model, num_heads: NumHeads, f: F) -> Self {
        DecoderBlock {
            self_attn: ResidualAdd(MultiHeadAttention::new(model, num_heads, model, model)),
            norm1: LayerNorm1D(model),
            mh_attn: MultiHeadAttention::new(model, num_heads, model, model),
            norm2: LayerNorm1D(model),
            ff: ResidualAdd(FeedForward {
                l1: Linear::new(model, f),
                act1: ReLU,
                l2: Linear::new(f, model),
            }),
            norm3: LayerNorm1D(model),
        }
    }
}

impl<M: Dim, H: Dim, F: Dim, E: Dtype, D: Device<E>, Tgt, Mem> basenn::Module<(Tgt, Mem)>
    for DeviceDecoderBlock<M, H, F, E, D>
where
    Tgt: WithEmptyTape + SplitTape + TryAdd<Tgt::NoTape, Output = Tgt> + HasErr<Err = D::Err>,
    Mem: Clone,
    ResidualAdd<DeviceMultiHeadAttention<M, H, M, M, E, D>>:
        basenn::Module<Tgt, Output = Tgt, Error = D::Err>,
    DeviceMultiHeadAttention<M, H, M, M, E, D>:
        basenn::Module<(Tgt, Mem, Mem), Output = Tgt, Error = D::Err>,
    DeviceLayerNorm1D<M, E, D>: basenn::Module<Tgt, Output = Tgt, Error = D::Err>,
    ResidualAdd<DeviceFeedForward<M, F, E, D>>: basenn::Module<Tgt, Output = Tgt, Error = D::Err>,
{
    type Output = Tgt;
    type Error = D::Err;

    fn try_forward(&self, (tgt, mem): (Tgt, Mem)) -> Result<Self::Output, D::Err> {
        let x = self.self_attn.try_forward(tgt)?;
        let x = self.norm1.try_forward(x)?;

        let (x, tape) = x.split_tape();
        let x_residual = x.clone();
        let x = self
            .mh_attn
            .try_forward((x.put_tape(tape), mem.clone(), mem))?;
        let x = x.try_add(x_residual)?;
        let x = self.norm2.try_forward(x)?;
        let x = self.ff.try_forward(x)?;
        self.norm3.try_forward(x)
    }
}

#[derive(Clone, Debug, CustomModule)]
pub struct Transformer<Model: Dim, NumHeads: Dim, F: Dim> {
    #[module]
    pub encoder: Vec<EncoderBlock<Model, NumHeads, F>>,
    #[module]
    pub decoder: Vec<DecoderBlock<Model, NumHeads, F>>,
}

impl<Model: Dim, NumHeads: Dim, F: Dim> Transformer<Model, NumHeads, F> {
    pub fn new(
        model: Model,
        num_heads: NumHeads,
        f: F,
        num_encoder_layers: usize,
        num_decoder_layers: usize,
    ) -> Self {
        let mut encoder = Vec::with_capacity(num_encoder_layers);
        for _ in 0..num_encoder_layers {
            encoder.push(EncoderBlock::new(model, num_heads, f));
        }
        let mut decoder = Vec::with_capacity(num_decoder_layers);
        for _ in 0..num_decoder_layers {
            decoder.push(DecoderBlock::new(model, num_heads, f));
        }
        Self { encoder, decoder }
    }
}

impl<M: Dim, H: Dim, F: Dim, E: Dtype, D: Device<E>, Src: SplitTape, Tgt: PutTape<Src::Tape>>
    basenn::Module<(Src, Tgt)> for DeviceTransformer<M, H, F, E, D>
where
    Vec<DeviceEncoderBlock<M, H, F, E, D>>: basenn::Module<Src, Output = Src, Error = D::Err>,
    DeviceDecoderBlock<M, H, F, E, D>: basenn::Module<
        (<Tgt as PutTape<Src::Tape>>::Output, Src::NoTape),
        Output = <Tgt as PutTape<Src::Tape>>::Output,
        Error = D::Err,
    >,
{
    type Output = <Tgt as PutTape<Src::Tape>>::Output;
    type Error = D::Err;

    fn try_forward(&self, (src, tgt): (Src, Tgt)) -> Result<Self::Output, D::Err> {
        let (mem, tape) = self.encoder.try_forward(src)?.split_tape();
        let mut tgt = tgt.put_tape(tape);
        for block in self.decoder.iter() {
            tgt = block.try_forward((tgt, mem.clone()))?;
        }
        Ok(tgt)
    }
}
