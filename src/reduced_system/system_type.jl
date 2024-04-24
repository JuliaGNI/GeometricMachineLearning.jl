"""
Can specify a special type of the system, to be used with ReducedSystem. For now the only option is Symplectic (and NoStructure).
"""
abstract type SystemType end
struct Symplectic <: SystemType end
struct NoStructure <: SystemType end