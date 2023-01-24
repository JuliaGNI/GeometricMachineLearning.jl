using Zygote

#define Hamiltonian
H(x) = x[2]^2 / 2 - cos(x[1])

#compute vector field
∇H(x) = Zygote.gradient(χ -> H(χ), x)[1]
dH(x) = [0 1;-1 0] * ∇H(x)

#get data set (includes dat & target)
function get_data_set(num=10, xymin=-1.2, xymax=+1.2)
	#range in which the data should be in
	rang = range(xymin, stop=xymax, length=num)

	# all combinations of (x,y) points
	data = [[x,y] for x in rang, y in rang]

	#compute the value of the vector field 
	target = ∇H.(data)

	return (data, target)
end
