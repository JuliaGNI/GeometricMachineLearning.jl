
# select a number of points at random (one batch)
function get_batch(data, target, batch_size=10)
	index = rand(eachindex(data, target), batch_size)
	return (data[index], target[index])
end
