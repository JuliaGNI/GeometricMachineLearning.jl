# Volume-Preserving Feedforward Layer 

**Volume preserving feedforward layers** are a special type of ResNet layer for which we restrict the weight matrices to be of a particular form. I.e. each layer computes: 

```math
x \mapsto x + \sigma(Ax + b),
```
where ``\sigma`` is a nonlinearity, ``A`` is the weight and ``b`` is the bias. The matrix ``A`` is either a lower-triangular matrix ``L`` or an upper-triangular matrix ``U``[^1]. The lower triangular matrix is of the form (the upper-triangular layer is simply the transpose of the lower triangular): 

[^1]: Implemented as `LowerTriangular` and `UpperTriangular` in `GeometricMachineLearning`.

```math 
L = \begin{pmatrix}
     0 & 0 & \cdots & 0      \\
     a_{21} & \ddots &        & \vdots \\
     \vdots & \ddots & \ddots & \vdots \\
     a_{n1} & \cdots & a_{n(n-1)}      & 0 
\end{pmatrix}.
```

The Jacobian of a layer of the above form then is of the form

```math 
J = \begin{pmatrix}
     1 & 0 & \cdots & 0      \\
     b_{21} & \ddots &        & \vdots \\
     \vdots & \ddots & \ddots & \vdots \\
     b_{n1} & \cdots & b_{n(n-1)}      & 1 
\end{pmatrix},
```
and the determinant of ``J`` is 1, i.e. the map is volume-preserving. 

## Neural network architecture

Volume-preserving feedforward neural networks should be used as `Architecture`s in `GeometricMachineLearning`. The constructor for them is: 

```@eval
using GeometricMachineLearning, Markdown
Markdown.parse(description(Val(:VPFconstructor)))
```

The constructor produces the following architecture[^2]:

[^2]: Based on the input arguments `n_linear` and `n_blocks`. In this example `init_upper` is set to false, which means that the first layer is of type *lower* followed by a layer of type *upper*. 

```@example 
import Images, Plots # hide
if Main.output_type == :html # hide
     HTML("""<object type="image/svg+xml" class="display-light-only" data=$(joinpath(Main.buildpath, "../tikz/vp_feedforward.png"))></object>""") # hide
else # hide
     Plots.plot(Images.load("../tikz/vp_feedforward.png"), axis=([], false)) # hide
end # hide
```

```@example
if Main.output_type == :html # hide
     HTML("""<object type="image/svg+xml" class="display-dark-only" data=$(joinpath(Main.buildpath, "../tikz/vp_feedforward_dark.png"))></object>""") # hide
end # hide
```

Here *LinearLowerLayer* performs ``x \mapsto x + Lx`` and *NonLinearLowerLayer* performs ``x \mapsto x + \sigma(Lx + b)``. The activation function ``\sigma`` is the forth input argument to the constructor and `tanh` by default. 

## Note on Sympnets 

As [SympNets](../architectures/sympnet.md) are symplectic maps, they also conserve phase space volume and therefore form a subcategory of volume-preserving feedforward layers. 