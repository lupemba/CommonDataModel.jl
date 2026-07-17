import DataStructures.OrderedCollections as OC

# A variable of raw integer codes (`data`), together with the pool of category
# labels and the raw-code -> ref lookup needed to present it as categorical
# values. Both are computed once, here, from `mapping` — not on every
# readblock!/getindex call, since readblock! in particular is called once per
# chunk during a single array read.
struct CategoricalVariable{V, N, R, TV <: AbstractVariable{R, N}} <: AbstractVariable{CategoricalValue{V, UInt32}, N}
    data::TV
    pool::CategoricalPool{V, UInt32}
    coderefs::OC.OrderedDict{R, UInt32}
end

function CategoricalVariable(data::AbstractVariable{R, N}, mapping::AbstractDict{R, V}) where {R, N, V}
    sorted_codes = sort(collect(keys(mapping)))
    labels = V[mapping[c] for c in sorted_codes]
    coderefs = OC.OrderedDict{R, UInt32}(c => i for (i, c) in enumerate(sorted_codes)) |> OC.freeze
    pool = CategoricalPool{V, UInt32}(labels, false)
    return CategoricalVariable{V, N, R, typeof(data)}(data, pool, coderefs)
end

# raw code -> category label, e.g. for looking up the label of a fill/missing value
_label(a::CategoricalVariable, code) = a.pool.levels[a.coderefs[code]]

# forward CommonDataModel.API
name(v_category::CategoricalVariable) = name(v_category.data)
dimnames(v_category::CategoricalVariable) = dimnames(v_category.data)
dataset(v_category::CategoricalVariable) = dataset(v_category.data)
attribnames(v_category::CategoricalVariable) = attribnames(v_category.data)
attrib(v_category::CategoricalVariable, name::SymbolOrString) = attrib(v_category.data, name)

# forward other basic method
Base.size(a::CategoricalVariable) = size(a.data)
DiskArrays.haschunks(a::CategoricalVariable) = DiskArrays.haschunks(a.data)
DiskArrays.eachchunk(a::CategoricalVariable) = DiskArrays.eachchunk(a.data)
Base.getindex(a::CategoricalVariable, name::SymbolOrString) = getindex(a.data, name)
Base.getindex(a::CategoricalVariable, name::CFStdName) = getindex(a.data, name)

# ---- readblock! ---------------------------------------------------------

function DiskArrays.readblock!(
    a::CategoricalVariable{V, N, R}, aout, r::AbstractUnitRange...
) where {V, N, R}
    broadcast!(aout, a.data[r...]) do x
        CategoricalValue{V, UInt32}(a.pool, a.coderefs[x])
    end
    return aout
end

function DiskArrays.writeblock!(
    ::CategoricalVariable, ::Any, r::AbstractUnitRange...
)
    throw(ArgumentError(
        "Writing to a categorical CF variable is not supported yet."
    ))
end

# ---- getindex -----------------------------------------------------------------
#
# Two methods via multiple dispatch — no runtime type check needed.

# Scalar indices (all Integer or CartesianIndex) → CategoricalValue
function Base.getindex(
    a::CategoricalVariable{V, N, R}, inds::Union{Integer, CartesianIndex}...
) where {V, N, R}
    @boundscheck checkbounds(a, inds...)
    DiskArrays.checkscalar(a, inds)
    code = a.data[inds...]
    return CategoricalValue{V, UInt32}(a.pool, a.coderefs[code])
end

# Array indices (ranges, colons, vectors, …) → CategoricalArray
function Base.getindex(a::CategoricalVariable{V, N, R}, inds...) where {V, N, R}
    @boundscheck checkbounds(a, inds...)
    raw = a.data[inds...]
    refs = UInt32[a.coderefs[c] for c in raw]
    return CategoricalArray{V, ndims(raw)}(refs, a.pool)
end

# ---- CFVariable -----------------------------------------------------------------
function _add_missing(data, parent_var::CategoricalVariable, f_m_vals)
    pairs = [_label(parent_var, code) => missing for code in f_m_vals]
    return isempty(pairs) ? data : replace(data, pairs...)
end

# A custom `maskingvalue` (e.g. NaN, used for numeric CFVariables) does not
# make sense for categorical data, so masked entries always become `missing`
# here regardless of `maskingvalue(v)`.
function DiskArrays.readblock!(
    v::CFVariable{T,N,TV}, aout, r::AbstractUnitRange...
    ) where {T,N,TV<:CategoricalVariable}

    parent_var = parent(v)
    data = similar(aout, eltype(parent_var))
    DiskArrays.readblock!(parent_var, data, r...)

    aout .= _add_missing(data, parent_var, fill_and_missing_values(v))

    return nothing
end


function Base.getindex(v::CFVariable{T,N,TV}, inds::Union{Integer, CartesianIndex}...
    ) where {T,N,TV<:CategoricalVariable}

    parent_var = parent(v)
    cat_val = Base.getindex(parent_var, inds...)
    is_missing = cat_val in (_label(parent_var, code) for code in fill_and_missing_values(v))
    return is_missing ? missing : cat_val
end

function Base.getindex(v::CFVariable{T,N,TV}, inds...
    ) where {T,N,TV<:CategoricalVariable}

    parent_var = parent(v)
    data = Base.getindex(parent_var, inds...)
    return _add_missing(data, parent_var, fill_and_missing_values(v))
end


function Base.getindex(v::CFVariable{T,N,TV}, name::SymbolOrString
    ) where {T,N,TV<:CategoricalVariable}

    parent_var = parent(v)
    return getindex(parent_var.data, name)
end

function Base.getindex(v::CFVariable{T,N,TV}, name::CFStdName
    ) where {T,N,TV<:CategoricalVariable}

    parent_var = parent(v)
    return getindex(parent_var.data, name)
end
