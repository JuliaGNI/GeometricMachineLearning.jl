
abstract type AbstractDataSymbol end

abstract type PositionSymbol <: AbstractDataSymbol end

abstract type PhaseSpaceSymbol <: PositionSymbol end
abstract type DerivativePhaseSpaceSymbol <: PhaseSpaceSymbol end

abstract type PosVeloSymbol <: PositionSymbol end
abstract type PosVeloAccSymbol <: PosVeloSymbol end


struct DataSymbol{T <: AbstractDataSymbol} end

type(::DataSymbol{T}) where T<: AbstractDataSymbol = T


symbols(::DataSymbol{AbstractDataSymbol}) = nothing
symbols(::DataSymbol{PositionSymbol}) = (:q,)
symbols(::DataSymbol{PhaseSpaceSymbol}) = (:q,:p)
symbols(::DataSymbol{DerivativePhaseSpaceSymbol}) = (:q,:p,:q̇,:ṗ)
symbols(::DataSymbol{PosVeloSymbol}) = (:q,:q̇)
symbols(::DataSymbol{PosVeloAccSymbol}) = (:q,:q̇,:q̈)


function can_reduce(::DataSymbol{T}, s₂::DataSymbol{M}) where {T<:AbstractDataSymbol, M  <:AbstractDataSymbol}
    T <: M ? true : false
end

function symboldiff(s₁::DataSymbol, s₂::DataSymbol) 
    _tuplediff(symbols(s₁),symbols(s₂))
end



transform(::DataSymbol{T}, ::DataSymbol{U}) where {T<:AbstractDataSymbol,U<:AbstractDataSymbol} = @error "No method to convert "*string(T)*" in "*string(M)*" !"

can_transform(::DataSymbol{T}, ::DataSymbol{U}) where {T<:AbstractDataSymbol,U<:AbstractDataSymbol} = false


#=
   DataSymbol(keys; kwargs...) give a DataSymbol{T} where T is the smaler (in sens of <:) type of AbstractDataSymbol
   the symbols of which are includes in keys.
=#

function DataSymbol(keys::Tuple; symbol = DataSymbol{AbstractDataSymbol}())
    for subtype in subtypes(type(symbol))
        if symbols(DataSymbol{subtype}()) == keys
            return DataSymbol{subtype}()
        elseif symbols(DataSymbol{subtype}()) ⊆ keys
            return DataSymbol(keys; symbol = DataSymbol{subtype}())
        end
    end
    return symbol
end


#=
    Some exceptions that need to be catch in the matching function.
=#

struct ReductionSymbolError <: Exception
    input_symbol::AbstractDataSymbol
    focus_symbol::AbstractDataSymbol
end

Base.showerror(io::IO, e::ReductionSymbolError) = print(io, String(e.input_symbol), "can not be reduced in", String(e.focus_symbol), " !")

struct TransformationSymbolError <: Exception
    input_symbol::AbstractDataSymbol
    focus_symbol::AbstractDataSymbol
end

Base.showerror(io::IO, e::TransformationSymbolError) = print(io, String(e.input_symbol), "can not be transformed in", String(e.focus_symbol), " !")