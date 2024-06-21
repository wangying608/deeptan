# tmapreduce
__precompile__(true)

module TMapReduce


using Base.Threads: nthreads, @spawn, @threads
export tmapreduce

function tmapreduce(f, op, itr; tasks_per_thread::Int = 8, kwargs...)
    chunk_size = max(1, length(itr) ÷ (tasks_per_thread * nthreads()))
    tasks = map(Iterators.partition(itr, chunk_size)) do chunk
        @spawn mapreduce(f, op, chunk; kwargs...)
    end
    mapreduce(fetch, op, tasks; kwargs...)
end


end
