import LinearAlgebra: I, Hermitian, tr, inv
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision, VariateType,
				  collectNaiveVariationalNodeInbounds, assembleClamp!, ultimatePartner
include("util.jl")

export ruleVariationalNARXOutNPPPPP,
       ruleVariationalNARXIn1PNPPPP,
       ruleVariationalNARXIn2PPNPPP,
       ruleVariationalNARXIn3PPPNPP,
	   ruleVariationalNARXIn4PPPPNP,
	   ruleVariationalNARXIn5PPPPPN


function ruleVariationalNARXOutNPPPPP(g :: Function,
									  marg_y :: Nothing,
									  marg_θ :: ProbabilityDistribution{Multivariate},
									  marg_x :: ProbabilityDistribution{Multivariate},
									  marg_u :: ProbabilityDistribution{Univariate},
                                      marg_z :: ProbabilityDistribution{Multivariate},
                                      marg_τ :: ProbabilityDistribution{Univariate})

	# Extract moments of beliefs
	mθ = unsafeMean(marg_θ)
	mx = unsafeMean(marg_x)
	mz = unsafeMean(marg_z)
	mu = unsafeMean(marg_u)
	mτ = unsafeMean(marg_τ)

	# Evaluate f at mθ
	fθ = mθ'*g([mx; mu; mz])

	# Set outgoing message
	return Message(Univariate, GaussianMeanPrecision, m=fθ, w=mτ)
end

function ruleVariationalNARXIn1PNPPPP(g :: Function,
									  marg_y :: ProbabilityDistribution{Univariate},
                                      marg_θ :: Nothing,
									  marg_x :: ProbabilityDistribution{Multivariate},
									  marg_u :: ProbabilityDistribution{Univariate},
                                      marg_z :: ProbabilityDistribution{Multivariate},
                                      marg_τ :: ProbabilityDistribution{Univariate})

    # Extract moments of beliefs
	my = unsafeMean(marg_y)
	mx = unsafeMean(marg_x)
	mu = unsafeMean(marg_u)
	mz = unsafeMean(marg_z)
	mτ = unsafeMean(marg_τ)

	# Jacobian of f w.r.t. θ
	Jθ = g([mx; mu; mz])

	# Update parameters
	Φ = mτ*Jθ*Jθ'
	ϕ = mτ*my*Jθ

	# Set message
    return Message(Multivariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalNARXIn2PPNPPP(g :: Function,
									  marg_y :: ProbabilityDistribution{Univariate},
									  marg_θ :: ProbabilityDistribution{Multivariate},
									  marg_x :: Nothing,
									  marg_u :: ProbabilityDistribution{Univariate},
							  	      marg_z :: ProbabilityDistribution{Multivariate},
                                      marg_τ :: ProbabilityDistribution{Univariate})

	error("Output history vector should be observed.")
    return Message(vague(GaussianWeightedMeanPrecision, 2))
end

function ruleVariationalNARXIn3PPPNPP(g :: Function,
									  marg_y :: ProbabilityDistribution{Univariate},
									  marg_θ :: ProbabilityDistribution{Multivariate},
									  marg_x :: ProbabilityDistribution{Multivariate},
									  marg_u :: Nothing,
							  	      marg_z :: ProbabilityDistribution{Multivariate},
                                      marg_τ :: ProbabilityDistribution{Univariate})

	error("Current input should be observed.")
    return Message(vague(GaussianWeightedMeanPrecision, 2))
end

function ruleVariationalNARXIn4PPPPNP(g :: Function,
									  marg_y :: ProbabilityDistribution{Univariate},
									  marg_θ :: ProbabilityDistribution{Multivariate},
									  marg_x :: ProbabilityDistribution{Multivariate},
									  marg_u :: ProbabilityDistribution{Univariate},
							  	      marg_z :: Nothing,
                                      marg_τ :: ProbabilityDistribution{Univariate})

	error("Input history vector should be observed.")
    return Message(vague(GaussianWeightedMeanPrecision))
end

function ruleVariationalNARXIn5PPPPPN(g :: Function,
									  marg_y :: ProbabilityDistribution{Univariate},
									  marg_θ :: ProbabilityDistribution{Multivariate},
									  marg_x :: ProbabilityDistribution{Multivariate},
									  marg_u :: ProbabilityDistribution{Univariate},
							  	      marg_z :: ProbabilityDistribution{Multivariate},
                                      marg_τ :: Nothing)

    # Extract moments of beliefs
	my = unsafeMean(marg_y)
	mθ = unsafeMean(marg_θ)
	mx = unsafeMean(marg_x)
	mz = unsafeMean(marg_z)
	mu = unsafeMean(marg_u)
	Vθ = unsafeCov(marg_θ)

	# Jacobian of f w.r.t. θ
	Jθ = g([mx; mu; mz])

	# Auto-regression function
	fθ = mθ'*Jθ

	# Update parameters
	a = 3/2.
	b = (my^2 - 2*my*fθ + fθ^2 + Jθ'*Vθ*Jθ)/2.

	# Set message
    return Message(Univariate, Gamma, a=a, b=b)
end


function collectNaiveVariationalNodeInbounds(node::NAutoregressiveX, entry::ScheduleEntry)
	inbounds = Any[]

	# Push function to calling signature (g needs to be defined in user scope)
	push!(inbounds, Dict{Symbol, Any}(:g => node.g, :keyword => false))

    target_to_marginal_entry = currentInferenceAlgorithm().target_to_marginal_entry

    for node_interface in entry.interface.node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        if node_interface === entry.interface
            # Ignore marginal of outbound edge
            push!(inbounds, nothing)
        elseif (inbound_interface != nothing) && isa(inbound_interface.node, Clamp)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, assembleClamp!(inbound_interface.node, ProbabilityDistribution))
        else
            # Collect entry from marginal schedule
            push!(inbounds, target_to_marginal_entry[node_interface.edge.variable])
        end
    end

    return inbounds
end
