using Test
using CommonDataModel: 
    AbstractCategoricalVariable,
    AbstractVariable,
    MemoryDataset,
    defVar,
    cfvariable,
    dataset

import CommonDataModel as CDM
import DiskArrays
import CategoricalArrays: CategoricalValue, CategoricalArray, unwrap, levels


struct CategoricalVariable{V, N, R} <: AbstractCategoricalVariable{V, N, R}
    data::AbstractVariable{R,N}
    mapping::Dict{R,V}
end

CDM.getvaluearray(a::CategoricalVariable) = a.data
CDM.getmapping(a::CategoricalVariable) = a.mapping

const CLOUD_MAPPING = Dict{Int8, String}(
    Int8(0) => "Not processed",
    Int8(1) => "Cloud free",
    Int8(2) => "Cloud contaminated",
    Int8(3) => "Cloud filled",
    Int8(4) => "Dust contaminated",
)

# 3×4 grid of cloud codes, chunked 2×2
const RAW_CODES = Int8[
    0 1 2 3;
    1 2 3 4;
    0 0 1 1
]

function make_mock()
    ds = MemoryDataset(tempname(), "c")
    v = defVar(ds,"cloud",RAW_CODES,("lon","lat"), attrib = [
        "standard_name"             => "cloud_mask",
        "long_name"                 => "Cloud mask",
        "_FillValue"                => Int8(0)
        ])

    mock = CategoricalVariable(parent(v), CLOUD_MAPPING)
    return mock
end


@testset "AbstractCategoricalVariable — eltype" begin
    mock = make_mock()
    @test eltype(mock) == CategoricalValue{String, UInt32}
end

@testset "AbstractCategoricalVariable — collect" begin
    mock = make_mock()
    ca = collect(mock)

    @test ca isa CategoricalArray{String, 2, UInt32}
    @test size(ca) == size(RAW_CODES)

    # Level ordering follows sorted raw codes (0,1,2,3,4)
    expected_levels = [CLOUD_MAPPING[k] for k in sort(collect(keys(CLOUD_MAPPING)))]
    @test levels(ca) == expected_levels

    # Values match the mapping
    for i in eachindex(RAW_CODES)
        @test unwrap(ca[i]) == CLOUD_MAPPING[RAW_CODES[i]]
    end
end


@testset "AbstractCategoricalVariable — array getindex" begin
    mock = make_mock()

    slice = mock[1:2, :]
    @test slice isa CategoricalArray{String, 2, UInt32}
    @test size(slice) == (2, 4)
    @test slice == collect(mock)[1:2, :]

    row = mock[1, :]
    @test row isa CategoricalArray{String, 1, UInt32}
    @test size(row) == (4,)
    @test row == collect(mock)[1, :]

    col = mock[:, 2]
    @test col isa CategoricalArray{String, 1, UInt32}
    @test unwrap.(col) == getindex.(Ref(CLOUD_MAPPING), RAW_CODES[:, 2])
end


@testset "AbstractCategoricalVariable — scalar getindex" begin
    mock = make_mock()

    val = mock[1, 1]
    @test val isa CategoricalValue{String, UInt32}
    @test unwrap(val) == CLOUD_MAPPING[RAW_CODES[1, 1]]

    val2 = mock[2, 4]
    @test unwrap(val2) == CLOUD_MAPPING[RAW_CODES[2, 4]]
end

@testset "AbstractCategoricalVariable — broadcasting" begin
    mock = make_mock()

    # Broadcast a function element-wise: unwrap over all elements
    broad_cast_var = mock .== "Cloud free"
    @test broad_cast_var isa DiskArrays.BroadcastDiskArray
    @test sum(broad_cast_var) == 4
end


@testset "AbstractCategoricalVariable — in" begin
    mock = make_mock()

    @test "Cloud free" in mock
    @test "Not processed" in mock
    @test !("Snowy" in mock)
end


@testset "categorical CFVariable" begin
    mock = make_mock()
    mock_cf = cfvariable(dataset(mock), "cloud_type";_v = mock)

    @test eltype(mock_cf) == Union{eltype(mock),Missing}
    @test ismissing(mock_cf[1, 1])
    @test mock_cf[1, 2] == mock[1, 2]
    @test mock_cf[2, 1] == mock[2, 1] 
    @test ismissing(mock_cf[3, 2])

    date_read = mock_cf[:,:]
    @test date_read isa CategoricalArray
    @test count(ismissing, date_read) == 3

    broad_cast_var = mock_cf .== "Cloud free"
    @test broad_cast_var isa DiskArrays.BroadcastDiskArray
    @test sum(skipmissing(broad_cast_var)) == 4
    
end
