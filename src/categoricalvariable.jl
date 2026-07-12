

abstract type AbstractCategoricalVariable{V, N, R} <: AbstractVariable{CategoricalValue{V, UInt32}, N} end

# special methods for AbstractCategoricalVariable
getvaluearray(a::AbstractCategoricalVariable)::AbstractVariable = throw(ArgumentError(
    "getvaluearray is not implemented for $(typeof(a))."
))
getmapping(a::AbstractCategoricalVariable)::AbstractDict = throw(ArgumentError(
    "getmapping is not implemented for $(typeof(a))."
))

# forward CommonDataModel.API
name(v_category::AbstractCategoricalVariable) = name(getvaluearray(v_category))
dimnames(v_category::AbstractCategoricalVariable) = dimnames(getvaluearray(v_category))
dataset(v_category::AbstractCategoricalVariable) = dataset(getvaluearray(v_category))
attribnames(v_category::AbstractCategoricalVariable) = attribnames(getvaluearray(v_category))
attrib(v_category::AbstractCategoricalVariable, name::SymbolOrString) = attrib(getvaluearray(v_category),name)

# forward other basic method
Base.size(a::AbstractCategoricalVariable) = size(getvaluearray(a))
DiskArrays.haschunks(a::AbstractCategoricalVariable) = DiskArrays.haschunks(getvaluearray(a))
DiskArrays.eachchunk(a::AbstractCategoricalVariable) = DiskArrays.eachchunk(getvaluearray(a))
Base.getindex(a::AbstractCategoricalVariable, name::SymbolOrString)  = getindex(getvaluearray(a),name)
Base.getindex(a::AbstractCategoricalVariable, name::CFStdName)  = getindex(getvaluearray(a),name)

# ---- internal helpers ---------------------------------------------------------

function _sorted_labels(mapping::AbstractDict{R, V}) where {R, V}
    sorted_codes = sort(collect(keys(mapping)))
    return V[mapping[c] for c in sorted_codes]
end

function _build_categorical_array(
    raw::AbstractArray{R, N}, mapping::AbstractDict{R, V}
) where {V, N, R}
    label_values = V[mapping[c] for c in raw]
    return CategoricalArray{V, N, UInt32}(label_values; levels=_sorted_labels(mapping))
end

function _build_cat_value(code::R, mapping::AbstractDict{R, V}) where {R, V}
    ca = CategoricalArray{V, 1, UInt32}(V[mapping[code]]; levels=_sorted_labels(mapping))
    return ca[1]
end


# ---- readblock! ---------------------------------------------------------

function DiskArrays.readblock!(
    a::AbstractCategoricalVariable{V, N, R}, aout, r::AbstractUnitRange...
) where {V, N, R}
    raw = Array{R}(undef, length.(r)...)
    DiskArrays.readblock!(getvaluearray(a), raw, r...)
    ca = _build_categorical_array(raw, getmapping(a))
    for i in eachindex(aout, ca)
        aout[i] = ca[i]
    end
    return ca
end

function DiskArrays.writeblock!(
    ::AbstractCategoricalVariable, ::Any, r::AbstractUnitRange...
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
    a::AbstractCategoricalVariable{V, N, R}, inds::Union{Integer, CartesianIndex}...
) where {V, N, R}
    @boundscheck checkbounds(a, inds...)
    DiskArrays.checkscalar(a, inds)
    raw = getvaluearray(a)[inds...]
    return _build_cat_value(raw, getmapping(a))
end

# Array indices (ranges, colons, vectors, …) → CategoricalArray
function Base.getindex(a::AbstractCategoricalVariable{V, N, R}, inds...) where {V, N, R}
    @boundscheck checkbounds(a, inds...)
    raw = getvaluearray(a)[inds...]
    return _build_categorical_array(raw, getmapping(a))
end

# ---- CFVariable -----------------------------------------------------------------
function _add_missing(data, mapping, f_m_vals)
    pairs = [mapping[code] => missing for code in f_m_vals]
    return isempty(pairs) ? data : replace(data, pairs...)
end
 
# A custom `maskingvalue` (e.g. NaN, used for numeric CFVariables) does not
# make sense for categorical data, so masked entries always become `missing`
# here regardless of `maskingvalue(v)`.
function DiskArrays.readblock!(
    v::CFVariable{T,N,TV}, aout, r::AbstractUnitRange...
    ) where {T,N,TV<:AbstractCategoricalVariable}

    parent_var = parent(v) ## 
    data = similar(aout, eltype(parent_var))
    DiskArrays.readblock!(parent_var, data, r...)

    aout .= _add_missing(data, 
            getmapping(parent_var),
        fill_and_missing_values(v))
    
    return nothing
end


function Base.getindex(v::CFVariable{T,N,TV}, inds::Union{Integer, CartesianIndex}...
    ) where {T,N,TV<:AbstractCategoricalVariable}
    
    parent_var = parent(v)
    cat_val = Base.getindex(parent_var, inds...)
    mapping = getmapping(parent_var)
    is_missing = cat_val in (mapping[code] for code in fill_and_missing_values(v)) 
    return is_missing ? missing : cat_val
end

function Base.getindex(v::CFVariable{T,N,TV}, inds...
    ) where {T,N,TV<:AbstractCategoricalVariable}
    
    parent_var = parent(v)
    data = Base.getindex(parent_var, inds...)
    return _add_missing(data, 
        getmapping(parent_var),
        fill_and_missing_values(v))
end


function Base.getindex(v::CFVariable{T,N,TV}, name::SymbolOrString
    ) where {T,N,TV<:AbstractCategoricalVariable}

    parent_var = parent(v)
    return getindex(getvaluearray(parent_var),name)
end

function Base.getindex(v::CFVariable{T,N,TV}, name::CFStdName
    ) where {T,N,TV<:AbstractCategoricalVariable}

    parent_var = parent(v)
    return getindex(getvaluearray(parent_var),name)
end
