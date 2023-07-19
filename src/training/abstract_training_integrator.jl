abstract type  AbstractTrainingIntegrator end

abstract type HnnTrainingIntegrator <: AbstractTrainingIntegrator end
abstract type LnnTrainingIntegrator <: AbstractTrainingIntegrator end
abstract type SympNetTrainingIntegrator <: AbstractTrainingIntegrator end

function loss end
function loss_single end


min_length_batch(ti::AbstractTrainingIntegrator) = 1