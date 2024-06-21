# 检查fastq: 将文件分为4行一组，对每组的第二行和第四行比较字数
using DelimitedFiles
using Tables: table

function my_count_lines(filePath::String)::Int64## The result may differ to `wc -l filePath`
    num_l::Int64 = 0
    ioin = open(filePath, "r")
    num_l = countlines(ioin)
    close(ioin)
    return num_l
end


function util_my_readline(filePath::String, nth::Int64, maxLen::Int64=8192*16, eol::AbstractChar='\n', windowSize::Int64=8192)::String
    aeol = UInt8(eol)
    a = Vector{UInt8}(undef, windowSize)
    nl = nb = 0
    nthd = nth - 1
    numElem = maxLen + 1
    outStrU = Vector{UInt8}(undef, numElem)
    UFilled = 0
    io = open(filePath, "r", lock=false)
    while !eof(io)
        nb = readbytes!(io, a)
        @views for i=1:nb#@simd
            @inbounds nl += a[i] == aeol
            if nl == nthd
                UFilled += 1
                outStrU[UFilled] = a[i]
            end
            if nl == nth || UFilled == numElem
                break
            end
        end
        if nl == nth || UFilled == numElem
            break
        end
    end
    close(io)
    outStr = ""
    if UFilled > 1
        if nth > 1
            outStr = join(Char.(outStrU[2:UFilled]))
        else
            outStr = join(Char.(outStrU[1:UFilled]))
        end
    else
        if nth < 2
            outStr = Char(outStrU[1]) * ""
        end
    end
    return outStr
end

function my_readline(filePath::String, nth::Int64; maxLen::Int64=8192*16, delim::Char='\t', isSplit::Bool=true, nElem::Int64=0, eol::AbstractChar='\n', windowSize::Int64=8192)::Union{String, Vector{String}}
    outStr = util_my_readline(filePath, nth, maxLen, eol, windowSize)
    if isSplit && length(outStr) > 1
        o_split::Vector{String} = split(outStr, delim)
        if nElem > 0
            return o_split[1:nElem]
        else
            return o_split
        end
    else
        return outStr
    end
end


function my_read_table(XPath::String, type::DataType=Float32, delim::Char=',', isTranspose::Bool=false)::Matrix
    txt = open(XPath) do file; read(file, String); end;
    out = readdlm(IOBuffer(txt), delim, type)
    if isTranspose; out = transpose(out) |> Matrix; end;
    return out
end

function my_write_table(tableO::Union{AbstractVecOrMat, AbstractDataFrame}, XPath::String; delim::Char='\t', isAppend::Bool=false, toTable::Bool=false)
    if isAppend
        isapd = "a"
    else
        isapd = "w"
    end
    if toTable
        open(XPath, isapd) do io
            writedlm(io, table(tableO), delim)
        end
    else
        open(XPath, isapd) do io
            writedlm(io, tableO, delim)
        end
    end
    return nothing
end
