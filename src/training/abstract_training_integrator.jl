abstract type  AbstractTrainingIntegrator end

abstract type HnnTrainingIntegrator <: AbstractTrainingIntegrator end
abstract type LnnTrainingIntegrator <: AbstractTrainingIntegrator end
abstract type SympNetTrainingIntegrator <: AbstractTrainingIntegrator end

function loss end
function loss_single end


#Define common strucutre integrator
struct TrainingIntegrator{TIT<:AbstractTrainingIntegrator, TSymbol<:AbstractDataSymbol, TShape<:AbstractDataShape, TD}
    sqdist::TD
end

@inline type(::TrainingIntegrator{T}) where T<: AbstractTrainingIntegrator = T
@inline symbols(::TrainingIntegrator{T,Symbols}) where {T<: AbstractTrainingIntegrator, Symbols<:AbstractDataSymbol} = Symbols
@inline shape(::TrainingIntegrator{T,Symbols, Shape}) where {T<: AbstractTrainingIntegrator, Symbols<:AbstractDataSymbol, Shape<:AbstarctDataShape} = Shape


