# Standalone check of the two Documenter passes that do not need the pages to be built:
# `@docs` block resolution and `@ref` resolution for code references.
#
# Documenter resolves a `@docs` entry with `DocSystem.getdocs(binding, typesig; modules)` and then
# keeps only the docstrings whose *defining* module (`d.data[:module]`) is in
# `makedocs(modules = …)`; a `@ref` to a code name resolves against the docstrings the manual
# actually includes. Both passes need only the loaded package, so this runs in seconds where
# `makedocs` takes hours. It is the cheap half of the Documentation job, not a replacement for it:
# `@example`/`@setup` blocks and section-title `@ref`s are not covered.
#
#     julia --project=docs docs/check_references.jl

using GeometricMachineLearning
using HDF5 # so that the extension below is loaded, as in make.jl
using AbstractNeuralNetworks # signatures in `@docs` blocks name types from here
using Documenter
using Documenter: DocSystem

# keep in sync with `modules` in make.jl
const MODULES = Module[GeometricMachineLearning,
    Base.get_extension(GeometricMachineLearning, :HDF5Ext)]

const SRC = joinpath(@__DIR__, "src")

"""Resolve `text` the way Documenter resolves a `@docs` entry. Returns `(ok, reason)`.

`mod` is the module the reference is written in: `Main` for a reference on a page, and the
*defining* module of the docstring for a reference inside one -- which is how Documenter resolves
them, and why a docstring GML writes on a `GeometricOptimizers` binding still resolves its own
`@ref`s in GML.
"""
function resolve(text::AbstractString, mod::Module = Main; exact_signature::Bool = true)
    ex = try
        Meta.parse(text)
    catch err
        return (false, "does not parse")
    end
    binding = try
        DocSystem.binding(mod, ex)
    catch err
        return (false, "no such binding")
    end
    DocSystem.iskeyword(binding) && return (true, "")
    DocSystem.defined(binding) || return (false, "undefined binding `$binding`")
    # A `@docs` entry has to match a specific method, but a `@ref` resolves to whatever anchor the
    # manual created for the binding -- `[`accuracy(::Chain, ::Tuple, ::DataLoader)`](@ref)` links to
    # `@docs accuracy` even though no method has that exact signature. Checking a `@ref` against the
    # signature would reject those.
    typesig = if exact_signature
        try
            Core.eval(mod, DocSystem.signature(ex, String(text)))
        catch err
            return (false, "signature does not evaluate")
        end
    else
        Union{}
    end
    docs = DocSystem.getdocs(binding, typesig; modules = MODULES)
    filter!(d -> d.data[:module] in MODULES, docs)
    if isempty(docs)
        elsewhere = DocSystem.getdocs(binding, typesig)
        where_ = isempty(elsewhere) ? "nowhere" :
                 join(sort!(unique(string(d.data[:module]) for d in elsewhere)), ", ")
        return (false, "`$binding` is documented in $where_, not in $(join(MODULES, ", "))")
    end
    (true, "")
end

function _collect_md!(pages::Set{String}, s::String)
    (endswith(s, ".md") && push!(pages, s); nothing)
end
function _collect_md!(pages::Set{String}, e::Expr)
    (foreach(a -> _collect_md!(pages, a), e.args); nothing)
end
_collect_md!(::Set{String}, ::Any) = nothing

"""
The pages `make.jl` builds, as absolute paths.

`walkdir(SRC)` is the wrong set. `docs/src` also holds pages that no `pages =` entry names, and
Documenter never reads those: their headings would excuse a section `@ref` the build rejects, and
their own references would be checked against a manual that does not contain them.

`make.jl` assembles `_html_pages` and `_latex_pages` from nested variables, so the list is read off
its syntax tree -- every page is a string literal ending in `.md` -- rather than duplicated here and
left to drift. Taking every such literal is the union of the HTML and LaTeX page sets, which is what
the job builds.
"""
function markdown_files()
    pages = Set{String}()
    _collect_md!(pages, Meta.parseall(read(joinpath(@__DIR__, "make.jl"), String)))
    isempty(pages) &&
        error("no `.md` literals in make.jl: the page list moved, and this check would pass blindly")
    files = String[]
    for page in sort!(collect(pages))
        path = joinpath(SRC, page)
        isfile(path) || error("make.jl lists `$page`, which does not exist under $SRC")
        push!(files, path)
    end
    files
end

"""
The lines of `path` outside fenced code blocks, as `(lineno, line)`.

The fence marker is matched with leading whitespace allowed. A fence indented inside a list or an
admonition is still a fence, and reading it as ordinary text fails in two ways: a matched indented
pair leaks the block's contents into the scan, and a pair with only one side indented leaves the
toggle inverted, which drops every remaining line of the file.
"""
function prose_lines(path::AbstractString)
    lines = Tuple{Int, String}[]
    infence = false
    for (i, line) in enumerate(eachline(path))
        if occursin(r"^\s*```", line)
            infence = !infence
            continue
        end
        infence || push!(lines, (i, line))
    end
    lines
end

"""Every entry of every `@docs` block in the manual, as `(file, line, text)`."""
function docs_entries()
    entries = Tuple{String, Int, String}[]
    for path in markdown_files()
        inblock = false
        for (i, line) in enumerate(eachline(path))
            if startswith(line, "```@docs")
                inblock = true
            elseif inblock && startswith(line, "```")
                inblock = false
            elseif inblock
                text = strip(line)
                (isempty(text) || startswith(text, "#")) && continue
                push!(entries, (relpath(path, SRC), i, String(text)))
            end
        end
    end
    entries
end

"""
Every ``[`x`](@ref)`` target in the manual that names code, as `(file, line, target)`.

An explicit target may itself contain parentheses -- ``[`geodesic`](@ref geodesic(::A, ::B) where T)``
-- so the closing `)` is found by counting depth rather than by a regular expression.
"""
function ref_targets()
    refs = Tuple{String, Int, String}[]
    opening = r"\[`([^`]+)`\]\(@ref"
    for path in markdown_files()
        for (i, line) in enumerate(eachline(path))
            for m in eachmatch(opening, line)
                rest = @view line[nextind(line, last(m.offset + ncodeunits(m.match) - 1)):end]
                depth, stop = 1, 0
                for (j, c) in pairs(rest)
                    c == '(' && (depth += 1)
                    c == ')' && (depth -= 1)
                    depth == 0 && (stop = j; break)
                end
                stop == 0 && continue # unbalanced; leave it to Documenter
                explicit = strip(rest[firstindex(rest):prevind(rest, stop)])
                target = isempty(explicit) ? strip(m.captures[1]) : explicit
                push!(refs, (relpath(path, SRC), i, String(target)))
            end
        end
    end
    refs
end

"""Every markdown heading in the manual, as the text Documenter anchors a section `@ref` by."""
function section_titles()
    titles = Set{String}()
    for path in markdown_files()
        for (_, line) in prose_lines(path)
            m = match(r"^#+\s+(.*?)\s*$", line)
            m === nothing || push!(titles, String(m.captures[1]))
        end
    end
    titles
end

"""
Every `[text](@ref)` target in the manual, outside fenced code blocks, whose link text is *not* in
backticks, as `(file, line, text)`.

Unbackticked is legitimate: `[Some Section](@ref)` is how one links to a section by its title. It
is a defect only when the text names no section and *does* name a documented binding, which means
the backticks were forgotten -- and that is what Documenter reports as an unresolved
`cross_references` error. Nothing else here sees it, because every other pass in this file matches
backticked references only.
"""
function unbackticked_ref_targets()
    refs = Tuple{String, Int, String}[]
    opening = r"\[([^\]`]+)\]\(@ref\s*\)"
    for path in markdown_files()
        for (i, line) in prose_lines(path)
            for m in eachmatch(opening, line)
                push!(refs, (relpath(path, SRC), i, String(strip(m.captures[1]))))
            end
        end
    end
    refs
end

"""
Docstrings in `MODULES` that no `@docs` block includes -- Documenter's `missing_docs` check.

Mirrors `Documenter.missingbindings`: collect every binding in each module's doc metadata with its
set of signatures, then strike out the ones a `@docs` entry covers.
"""
signatures_of(x::Base.Docs.MultiDoc) = x.order
signatures_of(::Any) = Type[Union{}]

function missing_docs()
    bindings = Dict{Docs.Binding, Set{Type}}()
    for mod in MODULES, (binding, doc) in DocSystem.getmeta(mod)

        isa(binding, Docs.Binding) || continue
        bindings[binding] = Set(signatures_of(doc))
    end
    for (_, _, text) in docs_entries()
        ex = try
            Meta.parse(text)
        catch
            continue
        end
        binding = try
            DocSystem.binding(Main, ex)
        catch
            continue
        end
        signature = try
            Core.eval(Main, DocSystem.signature(ex, String(text)))
        catch
            continue
        end
        haskey(bindings, binding) || continue
        signatures = bindings[binding]
        if signature === Union{} || length(signatures) == 1
            delete!(bindings, binding)
        elseif signature in signatures
            delete!(signatures, signature)
        end
    end
    [(binding, sig) for (binding, sigs) in bindings for sig in sigs]
end

"""
The ``[`x`](@ref)`` targets inside the docstrings the manual includes, with the module each
docstring was written in.

Documenter resolves these too, and they are easy to miss: they live in `src/`, not in the manual.
"""
function docstring_refs()
    refs = Tuple{String, String, Module}[]
    opening = r"\[`([^`]+)`\]\(@ref\s*([^)]*)\)"
    for (_, _, text) in docs_entries()
        ex = try
            Meta.parse(text)
        catch
            continue
        end
        binding = try
            DocSystem.binding(Main, ex)
        catch
            continue
        end
        typesig = try
            Core.eval(Main, DocSystem.signature(ex, String(text)))
        catch
            continue
        end
        docs = DocSystem.getdocs(binding, typesig; modules = MODULES)
        filter!(d -> d.data[:module] in MODULES, docs)
        for d in docs
            body = join(d.text, "")
            for m in eachmatch(opening, body)
                target = strip(m.captures[2])
                isempty(target) && (target = strip(m.captures[1]))
                startswith(target, "\"") && continue
                push!(refs, (String(text), String(target), d.data[:module]))
            end
        end
    end
    unique!(refs)
    refs
end

failures = Tuple{String, Int, String, String}[]

for (file, line, text) in docs_entries()
    ok, why = resolve(text)
    ok || push!(failures, (file, line, "@docs  $text", why))
end

for (owner, target, mod) in docstring_refs()
    ok, why = resolve(target, mod; exact_signature = false)
    ok || push!(failures, ("<docstring of $owner>", 0, "@ref   [`$target`]", why))
end

for (binding, sig) in missing_docs()
    push!(failures,
        ("<missing_docs>", 0,
            "$binding$(sig === Union{} ? "" : " :: $sig")",
            "documented in the package but not included in any `@docs` block"))
end

for (file, line, target) in ref_targets()
    # a `@ref` whose target is a section title rather than a code name is out of scope here
    startswith(target, "\"") && continue
    ok, why = resolve(target; exact_signature = false)
    ok || push!(failures, (file, line, "@ref   [`$target`]", why))
end

let titles = section_titles()
    for (file, line, target) in unbackticked_ref_targets()
        target in titles && continue
        ok, _ = resolve(target; exact_signature = false)
        ok && push!(failures,
            (file, line, "@ref   [$target]",
                "names no section, but `$target` is a documented binding -- the backticks are missing"))
    end
end

sort!(failures)
for (file, line, what, why) in failures
    println("FAIL $file:$line\n     $what\n     $why")
end
println("\n$(length(failures)) unresolved reference(s)")
exit(isempty(failures) ? 0 : 1)
