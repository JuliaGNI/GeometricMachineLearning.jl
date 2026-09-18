# `batch_over_two_axes` builds the minibatch index set. These two assertions hold it to the two
# properties that keep it cheap: its return type is concrete, and four times the minibatches cost
# about four times the memory. A tuple grown by splatting has neither — it infers as
# `Tuple{Vararg{Vector{Tuple{Int, Int}}}}` and costs the square of the minibatch count. See
# `CHANGELOG.md`.

using GeometricMachineLearning
using GeometricMachineLearning: batch_over_two_axes
using Test
import Random

Random.seed!(123)

@test isconcretetype(Base.infer_return_type(
    batch_over_two_axes, Tuple{Batch, Int, Int, DataLoader}))

function index_set_allocations(n_columns::Int, batch_size::Int = 8)
    dl = DataLoader(rand(Float32, 4, n_columns); autoencoder = false, suppress_info = true)
    batch = Batch(batch_size)
    batch(dl)
    @allocated batch(dl)
end

const allocations_250_batches = index_set_allocations(2000)
const allocations_1000_batches = index_set_allocations(8000)

# Four times the minibatches. Linear growth puts the ratio at four; a quadratic index set puts it
# at 12.25 here. The bound is wide enough for the fixed per-call cost and tight enough to fail on
# quadratic growth.
@test allocations_1000_batches / allocations_250_batches < 6
