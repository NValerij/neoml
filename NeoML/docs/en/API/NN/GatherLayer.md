# CGatherLayer Class

<!-- TOC -->

- [CGatherLayer Class](#cgatherlayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

Gather layer is aimed to extract particular elements from given sequence by its index.
It is useful for NLP tasks. For example, you can extract last subtoken features of every word in your BPE-tokenized sentence.

It is possible to add paddings to your index sequence. For example, if you have a batch of different length samples.
See Settings section below for details.

## Settings

```c++
void EnablePaddings( bool enable );
```

Enabling this option allows to use padding index (-1). It is useful for composing sequences of different length.
Paddings are zero-filled vectors which gradient doesn't flow to weights tensor.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The layer has two inputs.

`CGatherLayer::I_Weights` input is a source sequence `float`-tensor of the following dimensions:

- `BatchLength` is a source sequence length
- `BatchWidth` is the number of samples in the batch
- `ListSize` is equal to `1`
- `Height`, `Width`, `Depth` and `Channels` are of any size.

`CGatherLayer::I_Indexes` input is a selected indexes `float`-tensor of the following dimensions:

- `BatchLength` is a target sequence length
- `BatchWidth` equals `BatchWidth` of weights input
- the other dimensions equal `1`

Index values are from range [0, `BatchLength` from `I_Weights` input] (or [-1, `BatchLength` from `I_Weights` input] if paddings are enabled).

## Outputs

The single output returns a blob of the following dimensions:

- `BatchLength` equals `BatchLength` of `I_Indexes`
- `BatchWidth` equals inputs' `BatchWidth`
- `ListSize` equals `1`
- the other dimensions equal corresponding `I_Weights` dimensions

Blob contains chosen `I_Weights`'s object vectors selected by indexes from `I_Indexes`.
