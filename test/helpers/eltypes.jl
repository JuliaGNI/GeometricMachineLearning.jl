# The real element types every numeric test runs in. A test loops over this tuple rather than over a
# literal one, so that one line sets the precisions of the whole suite.
const REAL_ELTYPES = (Float32, Float64)
