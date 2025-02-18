using LinearAlgebra
using Utilities
using Distributions
using Random
using IterTools

function single_transitions!( neighbor_dependencies, single_transition, live, stationary_distribution )
    alphabet_size = size( neighbor_dependencies, 1 )
    alphabet = 1:alphabet_size
    
    for a in alphabet
        for b in alphabet
            single_transition[a, b] = 0.0
            denominator = 0.0
            for c in alphabet
                single_transition[a, b] += neighbor_dependencies[c, a, b] * stationary_distribution[live, c, a]
                denominator += stationary_distribution[live, c, a]
            end
            single_transition[a, b] /= denominator
        end
    end
    return single_transition
end

function calculate_single_transition( neighbor_dependencies, stationary_distribution )
    alphabet_size = size( neighbor_dependencies, 1 )
    single_transition = zeros( alphabet_size, alphabet_size )
    stationary_distribution = reshape( stationary_distribution, ( 1, alphabet_size, alphabet_size ) )
    single_transitions!( neighbor_dependencies, single_transition, 1, stationary_distribution )
    return single_transition
end

function stationary_distributions!( neighbor_dependencies, single_transition, live, stationary_distribution )
    alphabet_size = size( neighbor_dependencies, 1 )
    alphabet = 1:alphabet_size
    
    for a in alphabet
        for b in alphabet
            stationary_distribution[3-live, a, b] = 0.0
            for c in alphabet
                for d in alphabet
                    stationary_distribution[3-live, a, b] +=
                        stationary_distribution[live, c, d] * single_transition[c, a] * neighbor_dependencies[c, d, b]
                end
            end
        end
    end
    return stationary_distribution[3-live, :, :]
end

const alphabet_size = 2
const alphabet = 1:alphabet_size

neighbor_dependencies = zeros( alphabet_size, alphabet_size, alphabet_size )

neighbor_dependencies[1, 1, 2] = 0.01
neighbor_dependencies[2, 2, 1] = 0.02
neighbor_dependencies[1, 2, 1] = 0.1
neighbor_dependencies[2, 1, 2] = 0.15
for a in alphabet
    for b in alphabet
        neighbor_dependencies[a, b, b] = 1 - sum(neighbor_dependencies[a, b, :])
    end
end

stationary_distribution = zeros( alphabet_size, alphabet_size )
stationary_distribution[1, 1] = 0.35
stationary_distribution[1, 2] = 0.05
stationary_distribution[2, 1] = 0.1
stationary_distribution[2, 2] = 0.5

single_transition = zeros( alphabet_size, alphabet_size )

function contraction( neighbor_dependencies, single_transition, stationary_distribution )
    alphabet_size = size( neighbor_dependencies, 1 )
    scratch = zeros( 2, alphabet_size, alphabet_size )
    scratch[1, :, :] = stationary_distribution[:, :]
    live = 1
    diff = true
    while diff
        single_transitions!( neighbor_dependencies, single_transition, live, scratch )
        stationary_distributions!( neighbor_dependencies, single_transition, live, scratch )
        live = 3 - live
        diff = maximum(abs.(scratch[live, :, :] - scratch[3-live, :, :])) > 0
    end
    return (single_transition, scratch[live, :, :])
end

result = contraction( neighbor_dependencies, single_transition, stationary_distribution )
@time contraction( neighbor_dependencies, single_transition, stationary_distribution );
sum(result[1], dims=2)
sum(result[2])

# order of variables:
# LHS: p_0->0, p_0->1, p_1->0, p_1->1
# RHS: p00, p01, p10, p11
# I should really change this to use higher order tensors
index( alphabet_size, a, b ) = alphabet_size * (a - 1) + b

function f( neighbor_dependencies )
    alphabet_size = size( neighbor_dependencies, 1 )
    alphabet = 1:alphabet_size
    a2 = alphabet_size^2

    return (x; y = zeros( 2 * alphabet_size^2 )) -> begin
        for a in alphabet
            for b in alphabet
                numerator = 0.0
                denominator = 0.0
                for c in alphabet
                    j = a2 + index( alphabet_size, c, a )
                    numerator -= neighbor_dependencies[c, a, b] * x[j]
                    denominator += x[j]
                end
                i = index( alphabet_size, a, b )
                y[i] = x[i] + numerator/denominator
            end
        end
    
        for a in alphabet
            for b in alphabet
                i = a2 + index( alphabet_size, a, b )
                last = i == 2 * a2
                y[i] = last ? 1.0 : x[i]
                for c in alphabet
                    j = index( alphabet_size, c, a )
                    for d in alphabet
                        k = a2 + index( alphabet_size, c, d )
                        # if this is the last equation, replace it with unity sum
                        y[i] -= last ? x[k] : x[k] * x[j] * neighbor_dependencies[c, d, b]
                    end
                end
            end
        end

            
        return y
    end
end

function Df( neighbor_dependencies )
    alphabet_size = size( neighbor_dependencies, 1 )
    alphabet = 1:alphabet_size
    a2 = alphabet_size^2

    return (x; dy = zeros( 2 * a2, 2 * a2 )) -> begin
        for a in alphabet
            for b in alphabet
                numerator = 0.0
                denominator = 0.0
                for c in alphabet
                    j = a2 + index( alphabet_size, c, a )
                    numerator -= neighbor_dependencies[c, a, b] * x[j]
                    denominator += x[j]
                end
                i = index( alphabet_size, a, b )
                for c in alphabet
                    j = a2 + index( alphabet_size, c, a )
                    dy[i,j] = (-denominator * neighbor_dependencies[c, a, b] - numerator)/denominator^2
                end
                dy[i,i] += 1
            end
        end
    
        for a in alphabet
            for b in alphabet
                i = a2 + index( alphabet_size, a, b )
                last = i == 2 * a2
                dy[i,i] = last ? 0 : 1
                for c in alphabet
                    j = index( alphabet_size, c, a )
                    for d in alphabet
                        k = a2 + index( alphabet_size, c, d )
                        if last
                            dy[i,k] -= 1.0
                        else
                            dy[i,k] -= x[j] * neighbor_dependencies[c, d, b]
                            dy[i,j] -= x[k] * neighbor_dependencies[c, d, b]
                        end
                    end
                end
            end
        end
        return dy
    end
end

function Newton( neighbor_dependencies, x; epsilon = 1e-15 )
    g = f( neighbor_dependencies )
    dg = Df( neighbor_dependencies )

    delta = inv(dg(x))*g(x)
    while norm(delta) >= epsilon
        x = x - delta
        delta = inv(dg(x))*g(x)
    end
    return x
end

vst = vec(result[1]')
vsd = vec(result[2]')

single_transition = [0.9 0.1; 0.05 0.95]
x = [vec(single_transition'); vec(stationary_distribution')]
Newton( neighbor_dependencies, x )
@time Newton( neighbor_dependencies, x );

f( neighbor_dependencies )( [vst; vsd] )

g = f( neighbor_dependencies )
g( x )
dy = Df( neighbor_dependencies )( x )

a2 = alphabet_size^2
fidi = hcat((g.([x] .+ getindex.([1e-8 * I[1:2*a2,1:2*a2]], 1:2*a2, [:] )) .- [g(x)])./1e-8...)
@assert( maximum(abs.(dy - fidi)) .< 1e-6 )

separate( a ) =
    let sizes = size(a), p = length(sizes)-1
        if p == 0
            a
        else
            [separate( getindex( a, i, fill( :, p )... ) ) for i in 1:sizes[1]]
        end
    end

recursive_index( v::Vector{Float64}, s, i, j )::Vector{Float64} = v

recursive_index( a, s, i, j )::Vector{Float64} = recursive_index( a[s[i, j]], s, i, j+1 )

simulate( neighbor_dependencies, m, n; seed=1 ) =
    let sizes = size( neighbor_dependencies )
        simulate( separate( cumsum( neighbor_dependencies, dims=length(sizes) ) ), sizes, m, n, seed=seed )
    end

function simulate( cums::Vector, sizes, m, n; seed=1 )
    Random.seed!( seed )

    alphabet_size = sizes[1]
    p = length(sizes)-1
    
    s = zeros( Int, n, m )
    rand!( view( s, 1, : ), 1:alphabet_size )
    
    for i = 2:n
        for j = 1:m-(p-1)*(i-1)
            cum = recursive_index( cums, s, i-1, j )
            r = rand()
            l = 1
            while r > cum[l]
                l += 1
            end
            s[i, j] = l
        end
    end
    return s
end

s = simulate( neighbor_dependencies, 1_000_000, 100 );
s = simulate( neighbor_dependencies, 1_000_000, 100 );
@time s = simulate( neighbor_dependencies, 1_000_000, 100 );

recat( x ) = if isbitstype(eltype(x))
    x
else
    let y = recat.( x )
        stack( y, dims=1 )
    end
end

neighbor_dependencies = recat( [[[0.99 0.01; 0.98 0.02], [0.03 0.97; 0.04 0.96]], [[0.95 0.05; 0.94 0.06], [0.07 0.93; 0.08 0.92]]] )

@time s = simulate( neighbor_dependencies, 10_000_000, 100 );

find_pattern( s, pat ) = let n = length(pat)
    (s[1:end-n+1] .== pat[1]) .& if n > 1; find_pattern( s[2:end], pat[2:end] ); else true; end
end

mean(find_pattern( s[100,1:end-198], [1] ))
mean(find_pattern( s[100,1:end-198], [2] ))
mean(find_pattern( s[100,1:end-198], [1,1] ))
mean(find_pattern( s[100,1:end-198], [1,2] ))
mean(find_pattern( s[100,1:end-198], [2,1] ))
mean(find_pattern( s[100,1:end-198], [2,2] ))

function find_patterns( s, n, p, from, to; offset = 0 )
    m = size(s,2)
    
    priorstart = max(2 + offset, 1)
    posteriorstart = max(-offset, 1)
    priorend = m - (p-1)*(n-2)
    posteriorend = m - (p-1)*(n-1)

    priorlen = priorend - priorstart - length(from) + 1
    posteriorlen = posteriorend - posteriorstart - length(to) + 1
    len = min( priorlen, posteriorlen )

    priorend = priorstart + len + length(from) - 1
    posteriorend = posteriorstart + len + length(to) - 1
    
    x = find_pattern( s[n-1, priorstart:priorend], from );
    y = find_pattern( s[n, posteriorstart:posteriorend], to );
    return (x,y)
end

function arrow( s, n, p, from, to; offset = 0 )
    (x, y) = find_patterns( s, n, p, from, to, offset = offset )
    return sum( x .& y )/sum( x )
end

error = similar( neighbor_dependencies )
for t in reduce( vcat, product( fill( 1:2, 4 )... ) )
    error[t...] = arrow( s, 100, 3, t[1:end-1], t[end:end], offset=-1 )/neighbor_dependencies[t...] - 1
end
maximum(abs.(error))

function semicolon( s, n, p, from, to; offset=0 )
    (x, y) = find_patterns( s, n, p, from, to, offset = offset )
    return mean( x .& y )
end

semicolon.( [s], [n], [p], reduce( vcat, product( fill( 1:2, 3 )... ) ), [1] )

function circular_transition_probabilities( neighbor_dependencies, m )
    tuples = reduce( vcat, product( fill( 1:2, m )... ) )
    neighborhoods = collect.(partition.( map( t -> (t[end], t..., t[1]), tuples ), 3, 1 ))
    indices = vec( map( x -> ((y,z) -> (y...,z)).(x[1], x[2]), product( neighborhoods, tuples ) ) )

    sizes = size( neighbor_dependencies )
    ps = separate( neighbor_dependencies )

    N = 1 << m
    return (tuples, reshape((i -> prod( reduce.( getindex, i, init=ps ) )).( indices ), (N,N)))
end

function circular_stationary_distribution( neighbor_dependencies, m )
    N = 1 << m
    (tuples, P) = circular_transition_probabilities( neighbor_dependencies, m )
    return (tuples, vec([fill(0,N-1);1]' * inv([(I - P)[:,1:N-1] fill(1,N)])))
end

pi3 = circular_stationary_distribution( neighbor_dependencies, 3 )
pi4 = circular_stationary_distribution( neighbor_dependencies, 4 )

pi4[1][9]
sum(pi4[2][1:2])
pi4[2][1]+pi4[2][9]
pi3[2][1]

neighbor_dependencies = zeros( alphabet_size, alphabet_size, alphabet_size )

neighbor_dependencies[1, 1, 2] = 0.01
neighbor_dependencies[2, 2, 1] = 0.02
neighbor_dependencies[1, 2, 1] = 0.1
neighbor_dependencies[2, 1, 2] = 0.15
for a in alphabet
    for b in alphabet
        neighbor_dependencies[a, b, b] = 1 - sum(neighbor_dependencies[a, b, :])
    end
end

@time s = simulate( neighbor_dependencies, 10_000_000, 100 );

neighbor_error = similar( neighbor_dependencies )
equation_error = zeros( fill( 2, 4 )... )
theoretical_error = similar( neighbor_dependencies )
for t in reduce( vcat, product( fill( 1:2, 4 )... ) )
#    t = first( reduce( vcat, product( fill( 1:2, 4 )... ) ) )
    pbb = arrow( s, 100, 2, t[1:2], t[3:4], offset=0 )
    pbl = arrow( s, 100, 2, t[1:2], t[3:3], offset=0 )
    pbr = arrow( s, 100, 2, t[1:2], t[4:4], offset=-1 )
    pll = arrow( s, 100, 2, t[1:1], t[3:3], offset=0 )
    neighbor_error[t[1], t[2], t[4]] = pbr/neighbor_dependencies[t[1], t[2], t[4]] - 1
    equation_error[t...] = pbl*pbr/pbb - 1
    theoretical_error[t[1:3]...] = pll/pbl - 1
end

maximum(abs.(neighbor_error))
maximum(abs.(equation_error))
maximum(abs.(theoretical_error))
    
arrow( s, 100, 2, [1,1], [1,1], offset=0 )
arrow( s, 100, 2, [1,1], [1], offset=0 ) * arrow( s, 100, 2, [1,1], [1], offset=-1 )

arrow( s, 100, 2, [1], [1], offset=0 )

