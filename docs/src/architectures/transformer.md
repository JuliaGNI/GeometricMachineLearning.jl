# Standard Transformer

The transformer is a relatively modern neural network architecture [vaswani2017attention](@cite) that has come to dominate the field of natural language processing (NLP, [patwardhan2023transformers](@cite)) and replaced the previously dominant long-short term memory cells (LSTM, [hochreiter1997long](@cite)). Its success is due to a variety of factors: 
- unlike LSTMs it consists of very simple building blocks and hence is easier to interpret mathematically,
- it is very flexible in its application and the data it is fed with do not have to conform to a rigid pattern, 
- transformers utilize modern hardware (especially GPUs) very effectively. 

The transformer architecture is sketched below: 

```@example
Main.include_graphics("../tikz/transformer_encoder") # hide
```

It is nothing more than a combination of a [multihead attention layer](@ref "Multihead Attention") and a residual neural network[^1] (ResNet).

[^1]: A ResNet is nothing more than a neural network to whose output we again add the input, i.e. every ResNet is of the form ``\mathrm{ResNet}(x) = x + \mathcal{NN}(x)``.

## Library Functions 

```@docs; canonical=false
StandardTransformerIntegrator
```