
# select a number of points at random (one b	atch)
function get_batch(data, target, batch_size=10)
	index = rand(eachindex(data), batch_size)
	return (data[index], target[index])
end
