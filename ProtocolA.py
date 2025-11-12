import numpy as np
from math import sqrt, log2, log, pi, cos, sin
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
from ncpol2sdpa.nc_utils import ncdegree, get_monomials
from ncpol2sdpa.solver_common import get_xmat_value
import mosek
import chaospy
import qutip as qtp
# from joblib import Parallel, delayed, parallel_backend
from glob import glob
from scipy.optimize import minimize
import sympy
import csv
import pandas


sympy.var('pDC, v, eta, etad, etae, a,b')

p1 = 2*(1-pDC)/(2+(-1+v)*eta*etad)
p2 = 4*(-1+pDC)**2/((2+(-1+v)*eta*etad)**2)
p = p1-p2

Omega2 = sympy.Matrix([
    [0,1],
    [-1,0]
])
Omega4 = sympy.Matrix([
    [0,1,0,0],
    [-1,0,0,0],
    [0,0,0,1],
    [0,0,-1,0]
])

I2 = sympy.Matrix([
    [1,0],
    [0,1]
])
I4 = sympy.Matrix([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
])

gamma1 = sympy.Matrix([
    [1+(-1+v)*(4+(-3+v)*eta*etad)*etae/(4+2*(-1+v)*eta*etad), 0, (-1+v**2)*eta*etad*etae/(4+2*(-1+v)*eta*etad), 0],
    [0, 1+(-1+v)*(4+(-3+v)*eta*etad)*etae/(4+2*(-1+v)*eta*etad), 0, (-1+v**2)*eta*etad*etae/(4+2*(-1+v)*eta*etad)],
    [(-1+v**2)*eta*etad*etae/(4+2*(-1+v)*eta*etad), 0, 1+(-1+v)*(4+(-3+v)*eta*etad)*etae/(4+2*(-1+v)*eta*etad), 0],
    [0, (-1+v**2)*eta*etad*etae/(4+2*(-1+v)*eta*etad), 0, 1+(-1+v)*(4+(-3+v)*eta*etad)*etae/(4+2*(-1+v)*eta*etad)]
])

gamma2 = sympy.Matrix([
    [1-2*(-1+v)*(-1+eta*etad)*etae/(2+(-1+v)*eta*etad), 0, 0, 0],
    [0, 1-2*(-1+v)*(-1+eta*etad)*etae/(2+(-1+v)*eta*etad), 0, 0],
    [0, 0, 1-2*(-1+v)*(-1+eta*etad)*etae/(2+(-1+v)*eta*etad), 0],
    [0, 0, 0, 1-2*(-1+v)*(-1+eta*etad)*etae/(2+(-1+v)*eta*etad)]
])

gamma1A = sympy.Matrix([
    [1+(-1+v)*(4+(-3+v)*eta*etad)*etae/(4+2*(-1+v)*eta*etad), 0],
    [0, 1+(-1+v)*(4+(-3+v)*eta*etad)*etae/(4+2*(-1+v)*eta*etad)]
])
gamma1B = sympy.Matrix([
    [1+(-1+v)*(4+(-3+v)*eta*etad)*etae/(4+2*(-1+v)*eta*etad), 0],
    [0, 1+(-1+v)*(4+(-3+v)*eta*etad)*etae/(4+2*(-1+v)*eta*etad)]
])
gamma2A = sympy.Matrix([
    [1-2*(-1+v)*(-1+eta*etad)*etae/(2+(-1+v)*eta*etad), 0],
    [0, 1-2*(-1+v)*(-1+eta*etad)*etae/(2+(-1+v)*eta*etad)]
])
gamma2B = sympy.Matrix([
    [1-2*(-1+v)*(-1+eta*etad)*etae/(2+(-1+v)*eta*etad), 0],
    [0, 1-2*(-1+v)*(-1+eta*etad)*etae/(2+(-1+v)*eta*etad)]
])


d1 = 2*sympy.Matrix([
    [a],
    [0],
    [b],
    [0]
])
d2 = 2*sympy.Matrix([
    [a],
    [0],
    [b],
    [0]
])
d1A = 2*sympy.Matrix([
    [a],
    [0]
])
d1B = 2*sympy.Matrix([
    [b],
    [0]
])
d2A = 2*sympy.Matrix([
    [a],
    [0]
])
d2B = 2*sympy.Matrix([
    [b],
    [0]
])

p0A = (1-pDC)/p*(2*p1/sympy.sqrt((Omega2*(gamma1A+I2)*(Omega2.transpose())).det())*sympy.exp(-0.5*(Omega2*d1A).transpose()*(Omega2*(gamma1A+I2)*(Omega2.transpose())).inv()*(Omega2*d1A))\
                -2*p2/sympy.sqrt((Omega2*(gamma2A+I2)*(Omega2.transpose())).det())*sympy.exp(-0.5*(Omega2*d2A).transpose()*(Omega2*(gamma2A+I2)*(Omega2.transpose())).inv()*(Omega2*d2A)))[0]
p0B = (1-pDC)/p*(2*p1/sympy.sqrt((Omega2*(gamma1B+I2)*(Omega2.transpose())).det())*sympy.exp(-0.5*(Omega2*d1B).transpose()*(Omega2*(gamma1B+I2)*(Omega2.transpose())).inv()*(Omega2*d1B))\
                -2*p2/sympy.sqrt((Omega2*(gamma2B+I2)*(Omega2.transpose())).det())*sympy.exp(-0.5*(Omega2*d2B).transpose()*(Omega2*(gamma2B+I2)*(Omega2.transpose())).inv()*(Omega2*d2B)))[0]
p00 = (1-pDC)**2/p*(4*p1/sympy.sqrt((Omega4*(gamma1+I4)*(Omega4.transpose())).det())*sympy.exp(-0.5*(Omega4*d1).transpose()*(Omega4*(gamma1+I4)*(Omega4.transpose())).inv()*(Omega4*d1))\
                   -4*p2/sympy.sqrt((Omega4*(gamma2+I4)*(Omega4.transpose())).det())*sympy.exp(-0.5*(Omega4*d2).transpose()*(Omega4*(gamma2+I4)*(Omega4.transpose())).inv()*(Omega4*d2)))[0]
p01 = p0A-p00
p10 = p0B-p00
p11 = 1-p0A-p0B+p00

P0A = sympy.lambdify([v, eta, etae, etad, pDC, a, b], p0A)
P0B = sympy.lambdify([v, eta, etae, etad, pDC, a, b], p0B)
P00 = sympy.lambdify([v, eta, etae, etad, pDC, a, b], p00)
P01 = sympy.lambdify([v, eta, etae, etad, pDC, a, b], p01)
P10 = sympy.lambdify([v, eta, etae, etad, pDC, a, b], p10)
P11 = sympy.lambdify([v, eta, etae, etad, pDC, a, b], p11)
P   = sympy.lambdify([v, eta, etae, etad, pDC, a, b], p)

def ProbDist(v, eta, etae, etad, pDC, a, b):
  probdist = []
  q0A = P0A(v, eta, etae, etad, pDC, a, b)
  q0B = P0B(v, eta, etae, etad, pDC, a, b)
  q00 = P00(v, eta, etae, etad, pDC, a, b)
  q01 = P01(v, eta, etae, etad, pDC, a, b)
  q10 = P10(v, eta, etae, etad, pDC, a, b)
  q11 = P11(v, eta, etae, etad, pDC, a, b)
  qsucc = P(v, eta, etae, etad, pDC, a, b)

  probdist = [q0A, q0B, q00, q01, q10, q11, qsucc]
  return probdist

def cond_ent(joint, marg):
    """
    Returns H(A|B) = H(AB) - H(B)

    Inputs:
        joint    --     joint distribution on AB
        marg     --     marginal distribution on B
    """

    hab, hb = 0.0, 0.0

    for prob in joint:
        if 0.0 < prob < 1.0:
            hab += -prob*log2(prob)

    for prob in marg:
        if 0.0 < prob < 1.0:
            hb += -prob*log2(prob)

    return hab - hb


def objective(ti, pn):
    """
    Returns the objective function for the faster computations.
        Key generation on X=0
        Only two outcomes for Alice

        ti     --    i-th node
        q      --    bit flip probability
    """
    obj = 0.0
    F = [A[0][0], 1 - A[0][0]]                # POVM for Alices key gen measurement
    for a in range(A_config[0]):
        b = (a + 1) % 2                     # (a + 1 mod 2)
        M = (1-pn) * F[a] + pn * F[b]         # Noisy preprocessing povm element
        obj += M * (Z[a] + Dagger(Z[a]) + (1-ti)*Dagger(Z[a])*Z[a]) + ti*Z[a]*Dagger(Z[a])

    return obj

def compute_entropy(SDP, sys, eta, etae, pn):
    """
    Computes lower bound on H(A|X=0,E) using the fast (but less tight) method

        SDP   --   sdp relaxation object
        q     --   probability of bitflip
    """
    ck = 0.0        # kth coefficient
    ent = 0.0        # lower bound on H(A|X=0,E)

    # We can also decide whether to perform the final optimization in the sequence
    # or bound it trivially. Best to keep it unless running into numerical problems
    # with it. Added a nontrivial bound when removing the final term
    # (WARNING: proof is not yet in the associated paper).
    if KEEP_M:
        num_opt = len(T)
    else:
        num_opt = len(T) - 1
        ent = 2 * pn * (1-pn) * W[-1] / log(2)
        # ent = 0

    for k in range(num_opt):
        ck = W[k]/(T[k] * log(2))

        # Get the k-th objective function
        new_objective = objective(T[k], pn)

        SDP.set_objective(new_objective)
        SDP.solve('mosek')

        if SDP.status == 'optimal':
            # 1 contributes to the constant term
            ent += ck * (1 + SDP.dual)
        else:
            # If we didn't solve the SDP well enough then just bound the entropy
            # trivially
            ent = 0
            if VERBOSE:
                print('Bad solve: ', k, SDP.status)
            break

    return ent

def HAgB(sys, eta, etae, pn):
    """
    Computes the error correction term in the key rate for a given system,
    a fixed detection efficiency and noisy preprocessing. Computes the relevant
    components of the distribution and then evaluates the conditional entropy.

        sys    --    parameters of system
        eta    --    detection efficiency
        q      --    bitflip probability
    """

    # Computes H(A|B) required for rate
    [id, sx, sy, sz] = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
    [a0, a1, b0, b1, b2] = sys[:]

    # joint distribution
    p00 = ProbDist(v, eta, etae, etad, pDC, a0, b2)[2]
    p01 = ProbDist(v, eta, etae, etad, pDC, a0, b2)[3]
    p10 = ProbDist(v, eta, etae, etad, pDC, a0, b2)[4]
    p11 = ProbDist(v, eta, etae, etad, pDC, a0, b2)[5]

    pp00 = (1-pn) * p00 + pn * p10
    pp01 = (1-pn) * p01 + pn * p11
    pp10 = (1-pn) * p10 + pn * p00
    pp11 = (1-pn) * p11 + pn * p01

    pb0 = ProbDist(v, eta, etae, etad, pDC, a0, b2)[1]
    pb1 = 1-pb0

    qjoint = [pp00, pp01, pp10, pp11]
    qmarg = [pb0, pb1]

    return cond_ent(qjoint, qmarg)



def compute_rate(SDP, sys, eta, etae, pn):
    """
    Computes a lower bound on the rate H(A|X=0,E) - H(A|X=0,Y=2,B) using the fast
    method

        SDP       --     sdp relaxation object
        sys       --     parameters of the system
        eta       --     detection efficiency
        q         --     bitflip probability
    """
    score_cons = score_constraints(sys[:], eta, etae)
    SDP.process_constraints(equalities = op_eqs,
                        inequalities = op_ineqs,
                        momentequalities = moment_eqs[:] + score_cons[:],
                        momentinequalities = moment_ineqs)
    ent = compute_entropy(SDP, sys, eta, etae, pn)
    err = HAgB(sys, eta, etae, pn)
    return ent - err

def compute_dual_vector(SDP, sys, eta, etae, pn):
    """
    Extracts the vector from the dual problem(s) that builds into the affine function
    of the constraints that lower bounds H(A|X=0,E)

        SDP    --     sdp relaxation object
        q      --     probability of bitflip
    """

    dual_vec = np.zeros(8)    # dual vector
    ck = 0.0                # kth coefficient
    ent = 0.0                # lower bound on H(A|X=0,E)

    if KEEP_M:
        num_opt = len(T)
    else:
        num_opt = len(T) - 1
        ent = 2 * pn * (1-pn) * W[-1] / log(2)
        # ent = 0

    # Compute entropy and build dual vector from each sdp solved
    for k in range(num_opt):
        ck = W[k]/(T[k] * log(2))

        # Get the k-th objective function
        new_objective = objective(T[k], pn)

        # Set the objective and solve
        SDP.set_objective(new_objective)
        SDP.solve('mosek')

        # Check solution status
        if SDP.status == 'optimal':
            ent += ck * (1 + SDP.dual)
            # Extract the dual vector from the solved sdp
            d = sdp_dual_vec(SDP)
            # Add the dual vector to the total dual vector
            dual_vec = dual_vec + ck * d
        else:
            ent = 0
            dual_vec = np.zeros(8)
            break

    return dual_vec, ent

def score_constraints(sys, eta, etae):
    """
    Returns moment equality constraints generated by the two-qubit system specified by sys.
    In particular implements the constraints p(00|xy) and p(0|x), p(0|y) for each x,y.
    """

    [a0, a1, b0, b1, b2] = sys[:]

    a_list = [a0, a1]
    b_list = [b0, b1]


    # Now collect the constraints subject to the inefficient detection distribution
    constraints = []
    for x in range(2):
        for y in range(2):
               constraints += [A[x][0]*B[y][0] - ProbDist(v, eta, etae, etad, pDC, a_list[x], b_list[y])[2]]

    # Marginal constraints
    constraints += [A[0][0] - ProbDist(v, eta, etae, etad, pDC, a_list[0], 0)[0]]
    constraints += [B[0][0] - ProbDist(v, eta, etae, etad, pDC, 0, b_list[0])[1]]
    constraints += [A[1][0] - ProbDist(v, eta, etae, etad, pDC, a_list[1], 0)[0]]
    constraints += [B[1][0] - ProbDist(v, eta, etae, etad, pDC, 0, b_list[1])[1]]

    return constraints[:]

def sys2vec(sys, eta, etae):
    """
    Returns a vector of probabilities determined from the system in the same order as specified
    in the function score_constraints()

        sys    --     system parameters
        eta    --     detection efficiency
    """
    # Get the system from the parameters

    [a0, a1, b0, b1, b2] = sys[:]

    a_list = [a0, a1]
    b_list = [b0, b1]


    # Now collect the constraints subject to the inefficient detection distribution
    vec = []
    for x in range(2):
        for y in range(2):
               vec += [ProbDist(v, eta, etae, etad, pDC, a_list[x], b_list[y])[2]]

    # Marginal constraints
    vec += [ProbDist(v, eta, etae, etad, pDC, a_list[0], 0)[0]]
    vec += [ProbDist(v, eta, etae, etad, pDC, 0, b_list[0])[1]]
    vec += [ProbDist(v, eta, etae, etad, pDC, a_list[1], 0)[0]]
    vec += [ProbDist(v, eta, etae, etad, pDC, 0, b_list[1])[1]]

    return vec

def sdp_dual_vec(SDP):
    """
    Extracts the dual vector from the solved sdp by ncpol2sdpa

        SDP -- sdp relaxation object

    Would need to be modified if the number of moment constraints or their
    nature (equalities vs inequalities) changes.
    """
    raw_vec = SDP.y_mat[-16:]
    vec = [0 for _ in range(8)]
    for k in range(8):
        vec[k] = raw_vec[2*k][0][0] - raw_vec[2*k + 1][0][0]
    return np.array(vec[:])

def optimise_sys(SDP, sys, eta, etae, pn):
    """
    Optimizes the rate using the iterative method via the dual vectors.

        SDP    --    sdp relaxation object
        sys    --    parameters of system that are optimized
        eta    --    detection efficiency
        q      --    bitflip probability
    """

    NEEDS_IMPROVING = True            # Flag to check if needs optimizing still
    FIRST_PASS = True                # Checks if first time through loop
    improved_sys = sys[:]            # Improved choice of system
    best_sys = sys[:]                # Best system found
    dual_vec = np.zeros(8)            # Dual vector same length as num constraints

    [a0, a1, b0, b1, b2] = sys[:]
    psucc = ProbDist(v, eta, etae, etad, pDC, 0, 0)[6]

    # Loop until we converge on something
    while(NEEDS_IMPROVING):
        # On the first loop we just solve and extract the dual vector
        if not FIRST_PASS:
            # Here we optimize the dual vector
            # The distribution associated with the improved system
            pstar = sys2vec(improved_sys[:], eta, etae)

            # function to optimize parameters over
            def f0(x):
                #x is sys that we are optimizing
                p = sys2vec(x, eta, etae)
                return -np.dot(p, dual_vec) + HAgB(x, eta, etae, pn)

            # Bounds on the parameters of sys
            bounds = [[-5, 5]] + [[-5,5] for _ in range(len(sys) - 1)]
            # Optmize qubit system (maximizing due to negation in f0)
            res = minimize(f0, improved_sys[:], bounds = bounds)
            improved_sys = res.x.tolist()[:]    # Extract optimizer

        # Apply the new system to the sdp
        score_cons = score_constraints(improved_sys[:], eta, etae)
        SDP.process_constraints(equalities = op_eqs,
                            inequalities = op_ineqs,
                            momentequalities = moment_eqs[:] + score_cons[:],
                            momentinequalities = moment_ineqs)

        # Compute new dual vector and the rate
        dual_vec, new_ent = compute_dual_vector(SDP, improved_sys[:], eta, etae, pn)
        new_rate = new_ent - HAgB(improved_sys[:], eta, etae, pn)

        if not FIRST_PASS:
            if new_rate < best_rate + best_rate*EPS_M or new_rate < best_rate + EPS_A :
                NEEDS_IMPROVING = False
        else:
            # If first run through then this is the initial entropy
            starting_rate = new_rate
            best_rate = new_rate
            FIRST_PASS = False

        if new_rate > best_rate:
            psucc = ProbDist(v, eta, etae, etad, pDC, 0, 0)[6]
            print('Optimizing sys (eta, etae, pn) =', (eta, etae, pn), ' ... ', starting_rate, '->', new_rate)
            best_rate = new_rate
            best_sys = improved_sys[:]


    return best_rate, best_sys[:], psucc

def optimise_pn(SDP, sys, eta, etae, pn):
    """
    Optimizes the choice of q.

        SDP    --    sdp relaxation object
        sys    --    parameters of system that are optimized
        eta --     detection efficiency
        q     --     bitflip probability

    This function can probably be improved to make the search a bit more efficient and fine grained.
    """
    pn_eps = 0.005    # Can be tuned
    pn_eps_min = 0.001

    opt_pn = pn
    rate = compute_rate(SDP, sys, eta, etae, pn) # Computes rate for given q
    starting_rate = rate

    # We check if we improve going left
    if pn - pn_eps < 0:
        LEFT = 0
    else:
        new_rate = compute_rate(SDP, sys, eta, etae, opt_pn - pn_eps)
        if new_rate > rate:
            opt_pn = opt_pn - pn_eps
            rate = new_rate
            print('Optimizing pn (eta,etae,pn) =', (eta, etae, opt_pn), ' ... ', starting_rate, '->', rate)
            LEFT = 1
        else:
            LEFT = 0


    def next_pn(pn0, step_size):
        pn1 = pn0 + ((-1)**LEFT) * step_size
        if pn1 >= 0 and pn1 <= 0.5:
            return pn1
        elif step_size/2 >= pn_eps_min:
            return next_pn(pn0, step_size/2)
        else:
            return -1


    STILL_OPTIMIZING = 1

    while STILL_OPTIMIZING:
        # define the next q
        new_pn = next_pn(opt_pn, pn_eps)
        if new_pn < 0:
            break

        #compute the rate
        new_rate = compute_rate(SDP, sys, eta, etae, new_pn)

        if new_rate > rate:
            opt_pn = new_pn
            rate = new_rate
            print('Optimizing pn (eta,etae,pn) =', (eta, etae, opt_pn), ' ... ', starting_rate, '->', rate)
        else:
            # If we didn't improve try shortening the distance
            pn_eps = pn_eps / 2
            if pn_eps < pn_eps_min:
                STILL_OPTIMIZING = 0

    return rate, opt_pn



def optimise_rate(SDP, sys, eta, etae, pn):
    """
    Iterates between optimizing sys and optimizing q in order to optimize overall rate.
    """

    STILL_OPTIMIZING = 1

    best_rate = compute_rate(SDP, sys, eta, etae, pn)
    best_sys = sys[:]
    best_pn = pn

    while STILL_OPTIMIZING:
        _, new_sys, pSucc = optimise_sys(SDP, best_sys[:], eta, etae, best_pn)
        new_rate, new_pn = optimise_pn(SDP, new_sys[:], eta, etae, best_pn)
        # new_rate, new_sys = optimise_sys(SDP, best_sys[:], eta, etae, best_pn, best_pr)


        if (new_rate < best_rate + best_rate*EPS_M) or (new_rate < best_rate + EPS_A):
            STILL_OPTIMIZING = 0

        if new_rate > best_rate:
            best_rate = new_rate
            best_sys = new_sys[:]
            best_pn = new_pn

    return best_rate, best_sys, best_pn, pSucc

def generate_quadrature(m):
    """
    Generates the Gaussian quadrature nodes t and weights w. Due to the way the
    package works it generates 2*M nodes and weights. Maybe consider finding a
    better package if want to compute for odd values of M.

         m    --    number of nodes in quadrature / 2
    """
    t, w = chaospy.quadrature.radau(m, chaospy.Uniform(0, 1), 1)
    t = t[0]
    return t, w

def get_subs():
    """
    Returns any substitution rules to use with ncpol2sdpa. E.g. projections and
    commutation relations.
    """
    subs = {}
    # Get Alice and Bob's projective measurement constraints
    subs.update(ncp.projective_measurement_constraints(A,B))

    # Finally we note that Alice and Bob's operators should All commute with Eve's ops
    for a in ncp.flatten([A,B]):
        for z in Z:
            subs.update({z*a : a*z, Dagger(z)*a : a*Dagger(z)})

    return subs

def get_extra_monomials():
    """
    Returns additional monomials to add to sdp relaxation.
    """

    monos = []

    # Add ABZ
    ZZ = Z + [Dagger(z) for z in Z]
    Aflat = ncp.flatten(A)
    Bflat = ncp.flatten(B)
    for a in Aflat:
        for b in Bflat:
            for z in ZZ:
                monos += [a*b*z]

    # Add monos appearing in objective function
    for z in Z:
        monos += [A[0][0]*Dagger(z)*z]

    return monos[:]

"""
Now we start with setting up the ncpol2sdpa computations
"""
LEVEL = 2                          # NPA relaxation level
M = 4                              # Number of nodes / 2 in gaussian quadrature
T, W = generate_quadrature(M)      # Nodes, weights of quadrature
KEEP_M = 0                         # Optimizing mth objective function?
VERBOSE = 0                        # If > 1 then ncpol2sdpa will also be verbose
EPS_M, EPS_A = 1e-4, 1e-4          # Multiplicative/Additive epsilon in iterative optimization

# number of outputs for each inputs of Alice / Bobs devices
A_config = [2,2]
B_config = [2,2]

# Operators in problem
A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
Z = ncp.generate_operators('Z', 2, hermitian=0)

substitutions = get_subs()             # substitutions used in ncpol2sdpa
moment_ineqs = []                      # moment inequalities
moment_eqs = []                        # moment equalities
op_eqs = []                            # operator equalities
op_ineqs = []                          # operator inequalities
extra_monos = get_extra_monomials()    # extra monomials

v = 1.03
eta = 1
etad = 0.95
etae = 0.95
pDC = 10**(-6)


# Defining the test sys
test_sys = [0, -0.6, 0.3, -0.470277, 0]
score_cons = score_constraints(test_sys,eta, etae)
test_pn = 0

ops = ncp.flatten([A,B,Z])        # Base monomials involved in problem
obj = objective(1, test_pn)    # Placeholder objective function

sdp = ncp.SdpRelaxation(ops, verbose = VERBOSE-1, normalized=True, parallel=0)
sdp.get_relaxation(level = LEVEL,
                    equalities = op_eqs[:],
                    inequalities = op_ineqs[:],
                    momentequalities = moment_eqs[:] + score_cons,
                    momentinequalities = moment_ineqs[:],
                    objective = obj,
                    substitutions = substitutions,
                    extramonomials = extra_monos)



etas = np.arange(1,0.99,-0.01)
entropy = []
successprob = []
keyrates = []
qs = []
syss = []
for eta in etas:
    sdp.process_constraints(equalities = op_eqs[:],
						inequalities = op_ineqs[:],
						momentequalities = moment_eqs[:]+ score_constraints(test_sys, eta, etae),
						momentinequalities = moment_ineqs[:] )
    opt_rate, opt_sys, opt_pn, psucc = optimise_rate(sdp, test_sys[:], eta, etae, test_pn)
    opt_keyrate = opt_rate * psucc
    entropy += [opt_rate]
    test_sys = opt_sys
    test_pn = opt_pn
    successprob += [psucc]
    keyrates += [opt_keyrate]
    qs += [opt_pn]
    syss += [opt_sys]
    print(eta, etae, opt_rate)
    print(eta, etae, opt_sys)
    print(eta, etae, opt_keyrate)
    print(eta, etae, opt_pn)
    print(eta, etae, psucc)

np.savetxt(f"TMSV_ent_etae_{etae}.csv", np.array([entropy]).T, delimiter = ',')
np.savetxt(f"TMSV_succ_etae_{etae}.csv", np.array([successprob]).T, delimiter = ',')  
np.savetxt(f"TMSV_key_etae_{etae}.csv", np.array([keyrates]).T, delimiter = ',')  
np.savetxt(f"TMSV_q_etae_{etae}.csv", np.array([qs]).T, delimiter = ',') 
np.savetxt(f"TMSV_sys_etae_{etae}.csv", syss, delimiter = ',')   

csv_files = [f"TMSV_ent_etae_{etae}.csv", f"TMSV_succ_etae_{etae}.csv", f"TMSV_key_etae_{etae}.csv", f"TMSV_q_etae_{etae}.csv", f"TMSV_sys_etae_{etae}.csv"]
df_list = [pandas.read_csv(file) for file in csv_files]
merged_df = pandas.concat(df_list, axis=1)
merged_df.to_csv(f"All_Herald_ent_etae_{etae}.csv", index=False)