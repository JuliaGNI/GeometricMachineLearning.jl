using Zygote

#define Hamiltonian
H(x) = sum(x[1]^2 + x[2]^2) / 2

#compute vector field  
dH(x) = [0 1;-1 0] * Zygote.gradient(χ -> H(χ), x)[1]

#get data set (includes dat & target)
function get_data_set(num=10, xymin=-1.2, xymax=+1.2)
	#range in which the data should be in
	rang = range(xymin, stop=xymax, length=num)

	# all combinations of (x,y) points
	data = [[x,y] for x in rang, y in rang]

	#compute the value of the vector field 
	target = dH.(data)

	return (data, target)
end

#select a number of points at random (one b	atch)
function get_batch(data, target, batch_size=10)
	index = rand(eachindex(data), batch_size)
	return (data[index], target[index])
end
