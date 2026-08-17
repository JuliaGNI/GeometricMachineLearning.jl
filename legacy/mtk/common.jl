
#layer dimension/width
const ld = 5

#number of inputs/dimension of system
const n_in = 2

#expand model
expand(m) = (vec(m[1].W), m[1].b, vec(m[2].W), m[2].b, vec(m[3].W))
