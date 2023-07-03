abstract type  AbstractTrainingIntegrator end

abstract type HnnTrainingIntegrator <: AbstractTrainingIntegrator end
abstract type LnnTrainingIntegrator <: AbstractTrainingIntegrator end
abstract type SympNetTrainingIntegrator <: AbstractTrainingIntegrator end

function loss end
function loss_single end


#Define common strucutre integrator
struct TrainingIntegrator{TIT,TD}
    type::TIT
    sqdist::TD

    TrainingIntegrator(type;sqdist = sqeuclidean) = new{typeof(type),typeof(sqdist)}(type, sqdist)
end

type(ti::TrainingIntegrator) = ti.type