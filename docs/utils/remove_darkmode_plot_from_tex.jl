file = open("build/GeometricMachineLearning.jl.tex")

collection_of_lines = (line for line in eachline(file))

new_contents = ""

for (i, line) in zip(1:length(collection_of_lines), collection_of_lines)
    if line == raw" \begin{figure}"
        new_contents *= raw"\begin{comment}"
    elseif line == raw"\end{figure}" || collection_of_lines[i + 1] == ""
        new_contents *= raw"\end{comment}"
    else
        new_contents *= line
    end
end