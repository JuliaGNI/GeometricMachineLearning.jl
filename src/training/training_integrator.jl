#Define common strucutre integrator
struct TrainingIntegrator{TIT<:AbstractTrainingIntegrator, TSymbol<:AbstractDataSymbol, TShape<:AbstractDataShape, TD}
    sqdist::TD
end

@inline type(::TrainingIntegrator{T}) where T<: AbstractTrainingIntegrator = T
@inline symbols(::TrainingIntegrator{T,Symbols}) where {T<: AbstractTrainingIntegrator, Symbols<:AbstractDataSymbol} = Symbols
@inline shape(::TrainingIntegrator{T,Symbols, Shape}) where {T<: AbstractTrainingIntegrator, Symbols<:AbstractDataSymbol, Shape<:AbstractDataShape} = Shape

