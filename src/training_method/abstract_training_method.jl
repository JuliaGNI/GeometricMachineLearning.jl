abstract type  AbstractTrainingMethod end

abstract type HnnTrainingMethod <: AbstractTrainingMethod end
abstract type LnnTrainingMethod <: AbstractTrainingMethod end
abstract type SympNetTrainingMethod <: AbstractTrainingMethod end

function loss end
function loss_single end


min_length_batch(ti::AbstractTrainingMethod) = 1