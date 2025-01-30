function single_transitions!( neighbor_dependencies, single_transition, live, stationary_distribution )
    alphabet_size = size( neighbor_dependencies, 1 )
    alphabest = 1:alphabet_size
    
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

function stationary_distributions!( neighbor_dependencies, single_transition, live, stationary_distribution )
    alphabet_size = size( neighbor_dependencies, 1 )
    alphabest = 1:alphabet_size
    
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
neighbor_dependencies[2, 2, 1] = 0.01
neighbor_dependencies[1, 2, 1] = 0.1
neighbor_dependencies[2, 1, 2] = 0.1
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
    return scratch[live, :, :]
end

result = contraction( neighbor_dependencies, single_transition, stationary_distribution )
@time result = contraction( neighbor_dependencies, single_transition, stationary_distribution )
sum(result)

# order of variables:
# LHS: p_0->0, p_0->1, p_1->0, p_1->1
# RHS: p00, p01, p10, p11
index( alphabet_size, a, b ) = alphabet_size * (a - 1) + b

function single_transition_tensors( neighbor_dependencies )
    alphabet_size = size( neighbor_dependencies, 1 )
    alphabet = 1:alphabet_size

    as2 = alphabet_size^2

    A = zeros( as2, 2*as2, as2 )
    b = zeros( 2*as2, 2*as2 )
    
    single_transition_A = view( A, 1:as2, 1:as2, 1:as2 )
    single_transition_b = view( B, 1:as2, 1:as2 )
    
    for a in alphabet
        for b in alphabet
            # p_a->b equation
            i = index( alphabet_size, a, b )
            for c in alphabet
                # p_ca
                j = index( alphabet_size, c, a )

                # coefficient for p_a->b * p_ca
                single_transition_A[i, i, j] = 1.0
                # coefficient for p_ca
                single_transition_b[i, j] += - neighbor_dependencies[c, a, b]
            end
        end
    end

    stationary_distribution_indices = as2 .+ (1:as2)
    stationary_distribution_A = view( A, 1:as2, stationary_distribution_indices, 1:as2 )
    stationary_distribution_b = zeros( stationary_distribution_indices, stationary_distribution_indices )
    for a in alphabet
        for b in alphabet
            # p_ab equation
            i = index( alphabet_size, a, b )
            stationary_distribution_b[i, i] = 1.0
            for c in alphabet
                # p_c->a coefficients
                j = index( alphabet_size, c, a )
                for d in alphabet
                    # p_cd
                    k = index( alphabet_size, c, d )
                    # coefficient for p_c->A * p_cd
                    stationary_distribution_A[j, i, k] = - neighbor_dependencies[c, d, b]
                end
            end
        end
    end

    return (A, b)
end
