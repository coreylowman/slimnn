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
pub struct EncoderBlock<Model: Dim, NumHeads: Dim, F: Dim> {
    pub self_attn: ResidualAdd<MultiHeadAttention<Model, NumHeads>>,
    pub norm1: LayerNorm1D<Model>,
    pub ff: ResidualAdd<(Linear<Model, F>, ReLU, Linear<F, Model>)>,
    pub norm2: LayerNorm1D<Model>,
}

#[derive(Clone, Debug, CustomModule)]
pub struct DecoderBlock<Model: Dim, NumHeads: Dim, F: Dim> {
    pub self_attn: ResidualAdd<MultiHeadAttention<Model, NumHeads>>,
    pub norm1: LayerNorm1D<Model>,
    pub mh_attn: MultiHeadAttention<Model, NumHeads>,
    pub norm2: LayerNorm1D<Model>,
    pub ff: ResidualAdd<(Linear<Model, F>, ReLU, Linear<F, Model>)>,
    pub norm3: LayerNorm1D<Model>,
}

impl<M: Dim, H: Dim, F: Dim, E: Dtype, D: Device<E>, Tgt, Mem> basenn::Module<(Tgt, Mem)>
    for DeviceDecoderBlock<M, H, F, E, D>
where
    Tgt: SplitTape + TryAdd<Tgt::NoTape> + HasErr<Err = D::Err>,
    Mem: Clone,
    DeviceMultiHeadAttention<M, H, M, M, E, D>: basenn::Module<Tgt, Output = Tgt, Error = D::Err>
        + basenn::Module<(Tgt, Mem, Mem), Output = Tgt, Error = D::Err>,
    DeviceLayerNorm1D<M, E, D>: basenn::Module<Tgt, Output = Tgt, Error = D::Err>,
    FF<M, F, E, D>: basenn::Module<Tgt, Output = Tgt, Error = D::Err>,
{
    type Output = Tgt;
    type Error = D::Err;

    fn try_forward(&self, (tgt, mem): (Tgt, Mem)) -> Result<Self::Output, D::Err> {
        let (tgt, tape) = tgt.split_tape();
        let x = self.self_attn.try_forward(tgt.clone().put_tape(tape))?;
        let x = x.try_add(tgt)?;
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
    pub encoder: Vec<EncoderBlock<Model, NumHeads, F>>,
    pub decoder: Vec<DecoderBlock<Model, NumHeads, F>>,
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
        self.decoder.try_forward((tgt.put_tape(tape), mem))
    }
}
