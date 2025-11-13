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

sympy.var('pd, pDC, vA, vB, tauB, eta, etad, etae, a,b')

p1 = -((4*(-1+pd))/sympy.sqrt((4+eta*etad*(-2+vA+vB+tauB-vB*tauB))**2))
p2 = (4*(-1+pd)**2)/sympy.sqrt((2+(-1+vA)*eta*etad)**2*(-2+(-1+vB)*eta*etad*(-1+tauB))**2)
p3 = (8*(-1+pd)**2)/sympy.sqrt((8+(-1+vB)*eta*etad**2*(-3+vA+2*tauB)+2*etad*(-2+eta*(-2+vA+tauB)+vB*(2+eta-eta*tauB)))**2)
p4 = -((4*(-1+pd)**3)/((2+(-1+vA)*eta*etad)*sympy.sqrt((2+(-1+vB)*(etad+eta*(-1+etad)*etad*(-1+tauB)))**2)))
p = p1-p2-p3+p4

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
    [1+(-1+vA)*etae-((-1+vA**2)*eta*etad*etae)/(4+eta*etad*(-2+vA+vB+tauB-vB*tauB)), 0, -(((-1+vB)*eta*etad*etae*sympy.sqrt(-((-1+vA**2)*(-1+tauB)*tauB)))/(4+eta*etad*(-2+vA+vB+tauB-vB*tauB))), 0],
    [0, 1+(-1+vA)*etae-((-1+vA**2)*eta*etad*etae)/(4+eta*etad*(-2+vA+vB+tauB-vB*tauB)), 0, (((-1+vB)*eta*etad*etae*sympy.sqrt(-((-1+vA**2)*(-1+tauB)*tauB)))/(4+eta*etad*(-2+vA+vB+tauB-vB*tauB)))],
    [-(((-1+vB)*eta*etad*etae*sympy.sqrt(-((-1+vA**2)*(-1+tauB)*tauB)))/(4+eta*etad*(-2+vA+vB+tauB-vB*tauB))), 0, 1+(-1+vB)*etae*tauB+((-1+vB)**2*eta*etad*etae*(-1+tauB)*tauB)/(4+eta*etad*(-2+vA+vB+tauB-vB*tauB)), 0],
    [0, (((-1+vB)*eta*etad*etae*sympy.sqrt(-((-1+vA**2)*(-1+tauB)*tauB)))/(4+eta*etad*(-2+vA+vB+tauB-vB*tauB))), 0, 1+(-1+vB)*etae*tauB+((-1+vB)**2*eta*etad*etae*(-1+tauB)*tauB)/(4+eta*etad*(-2+vA+vB+tauB-vB*tauB))]
])

gamma3 = sympy.Matrix([
    [(8+8*(-1+vA)*etae+(-1+vB)*eta*etad**2*(-3+vA+4*etae+2*vA*etae*(-2+tauB)+2*tauB-2*etae*tauB)+2*etad*(-2+(-2+vA)*eta+2*etae-2*vA*etae-vB*(1+(-1+vA)*etae)*(-2+eta*(-1+tauB))+eta*((-1+vA)*etae*(-3+tauB)+tauB)))/(8+(-1+vB)*eta*etad**2*(-3+vA+2*tauB)+2*etad*(-2+eta*(-2+vA+tauB)+vB*(2+eta-eta*tauB))), 0, (2*(-1+vB)*sympy.sqrt((-1+vA**2)*eta*etad*etae)*(-sympy.sqrt(-eta*etad*etae*(-1+tauB)*tauB)+sympy.sqrt(-eta*etad**3*etae*(-1+tauB)*tauB)))/(8+(-1+vB)*eta*etad**2*(-3+vA+2*tauB)+2*etad*(-2+eta*(-2+vA+tauB)+vB*(2+eta-eta*tauB))), 0],
    [0, (8+8*(-1+vA)*etae+(-1+vB)*eta*etad**2*(-3+vA+4*etae+2*vA*etae*(-2+tauB)+2*tauB-2*etae*tauB)+2*etad*(-2+(-2+vA)*eta+2*etae-2*vA*etae-vB*(1+(-1+vA)*etae)*(-2+eta*(-1+tauB))+eta*((-1+vA)*etae*(-3+tauB)+tauB)))/(8+(-1+vB)*eta*etad**2*(-3+vA+2*tauB)+2*etad*(-2+eta*(-2+vA+tauB)+vB*(2+eta-eta*tauB))), 0, -(2*(-1+vB)*sympy.sqrt((-1+vA**2)*eta*etad*etae)*(-sympy.sqrt(-eta*etad*etae*(-1+tauB)*tauB)+sympy.sqrt(-eta*etad**3*etae*(-1+tauB)*tauB)))/(8+(-1+vB)*eta*etad**2*(-3+vA+2*tauB)+2*etad*(-2+eta*(-2+vA+tauB)+vB*(2+eta-eta*tauB)))],
    [(2*(-1+vB)*sympy.sqrt((-1+vA**2)*eta*etad*etae)*(-sympy.sqrt(-eta*etad*etae*(-1+tauB)*tauB)+sympy.sqrt(-eta*etad**3*etae*(-1+tauB)*tauB)))/(8+(-1+vB)*eta*etad**2*(-3+vA+2*tauB)+2*etad*(-2+eta*(-2+vA+tauB)+vB*(2+eta-eta*tauB))), 0, (8+etad*(4*(-1+vB)+2*(-2+vA+vB)*eta+(-3+vA)*(-1+vB)*eta*etad)-2*(-1+vB)*(-1+etad)*(4*etae+eta*etad*(-1+(-1+vA)*etae))*tauB)/(8+(-1+vB)*eta*etad**2*(-3+vA+2*tauB)+2*etad*(-2+eta*(-2+vA+tauB)+vB*(2+eta-eta*tauB))), 0],
    [0, -(2*(-1+vB)*sympy.sqrt((-1+vA**2)*eta*etad*etae)*(-sympy.sqrt(-eta*etad*etae*(-1+tauB)*tauB)+sympy.sqrt(-eta*etad**3*etae*(-1+tauB)*tauB)))/(8+(-1+vB)*eta*etad**2*(-3+vA+2*tauB)+2*etad*(-2+eta*(-2+vA+tauB)+vB*(2+eta-eta*tauB))), 0, (8+etad*(4*(-1+vB)+2*(-2+vA+vB)*eta+(-3+vA)*(-1+vB)*eta*etad)-2*(-1+vB)*(-1+etad)*(4*etae+eta*etad*(-1+(-1+vA)*etae))*tauB)/(8+(-1+vB)*eta*etad**2*(-3+vA+2*tauB)+2*etad*(-2+eta*(-2+vA+tauB)+vB*(2+eta-eta*tauB)))]
])

gamma2 = sympy.Matrix([
    [(2+2*(-1+vA)*etae+eta*etad*(-1+vA+2*etae-2*vA*etae))/(2+(-1+vA)*eta*etad), 0, 0, 0],
    [0, (2+2*(-1+vA)*etae+eta*etad*(-1+vA+2*etae-2*vA*etae))/(2+(-1+vA)*eta*etad), 0, 0],
    [0, 0, (-2+(-1+vB)*eta*etad*(-1+tauB)-2*(-1+vB)*etae*tauB)/(-2+(-1+vB)*eta*etad*(-1+tauB)), 0],
    [0, 0, 0, (-2+(-1+vB)*eta*etad*(-1+tauB)-2*(-1+vB)*etae*tauB)/(-2+(-1+vB)*eta*etad*(-1+tauB))]
])

gamma4 = sympy.Matrix([
    [(2+2*(-1+vA)*etae+eta*etad*(-1+vA+2*etae-2*vA*etae))/(2+(-1+vA)*eta*etad), 0, 0, 0],
    [0, (2+2*(-1+vA)*etae+eta*etad*(-1+vA+2*etae-2*vA*etae))/(2+(-1+vA)*eta*etad), 0, 0],
    [0, 0, (2+(-1+vB)*eta*etad**2*(-1+tauB)+2*(-1+vB)*etae*tauB-(-1+vB)*etad*(-1+eta*(-1+tauB)+2*etae*tauB))/(2-(-1+vB)*etad*(-1+eta*(-1+tauB))+(-1+vB)*eta*etad**2*(-1+tauB)), 0],
    [0, 0, 0, (2+(-1+vB)*eta*etad**2*(-1+tauB)+2*(-1+vB)*etae*tauB-(-1+vB)*etad*(-1+eta*(-1+tauB)+2*etae*tauB))/(2-(-1+vB)*etad*(-1+eta*(-1+tauB))+(-1+vB)*eta*etad**2*(-1+tauB))]
])

gamma1A = sympy.Matrix([
    [1+(-1+vA)*etae-((-1+vA**2)*eta*etad*etae)/(4+eta*etad*(-2+vA+vB+tauB-vB*tauB)), 0],
    [0, 1+(-1+vA)*etae-((-1+vA**2)*eta*etad*etae)/(4+eta*etad*(-2+vA+vB+tauB-vB*tauB))]
])
gamma1B = sympy.Matrix([
    [1+(-1+vB)*etae*tauB+((-1+vB)**2*eta*etad*etae*(-1+tauB)*tauB)/(4+eta*etad*(-2+vA+vB+tauB-vB*tauB)), 0],
    [0, 1+(-1+vB)*etae*tauB+((-1+vB)**2*eta*etad*etae*(-1+tauB)*tauB)/(4+eta*etad*(-2+vA+vB+tauB-vB*tauB))]
])
gamma3A = sympy.Matrix([
    [(8+8*(-1+vA)*etae+(-1+vB)*eta*etad**2*(-3+vA+4*etae+2*vA*etae*(-2+tauB)+2*tauB-2*etae*tauB)+2*etad*(-2+(-2+vA)*eta+2*etae-2*vA*etae-vB*(1+(-1+vA)*etae)*(-2+eta*(-1+tauB))+eta*((-1+vA)*etae*(-3+tauB)+tauB)))/(8+(-1+vB)*eta*etad**2*(-3+vA+2*tauB)+2*etad*(-2+eta*(-2+vA+tauB)+vB*(2+eta-eta*tauB))), 0],
    [0, (8+8*(-1+vA)*etae+(-1+vB)*eta*etad**2*(-3+vA+4*etae+2*vA*etae*(-2+tauB)+2*tauB-2*etae*tauB)+2*etad*(-2+(-2+vA)*eta+2*etae-2*vA*etae-vB*(1+(-1+vA)*etae)*(-2+eta*(-1+tauB))+eta*((-1+vA)*etae*(-3+tauB)+tauB)))/(8+(-1+vB)*eta*etad**2*(-3+vA+2*tauB)+2*etad*(-2+eta*(-2+vA+tauB)+vB*(2+eta-eta*tauB)))]
])
gamma3B = sympy.Matrix([
    [(8+etad*(4*(-1+vB)+2*(-2+vA+vB)*eta+(-3+vA)*(-1+vB)*eta*etad)-2*(-1+vB)*(-1+etad)*(4*etae+eta*etad*(-1+(-1+vA)*etae))*tauB)/(8+(-1+vB)*eta*etad**2*(-3+vA+2*tauB)+2*etad*(-2+eta*(-2+vA+tauB)+vB*(2+eta-eta*tauB))), 0],
    [0, (8+etad*(4*(-1+vB)+2*(-2+vA+vB)*eta+(-3+vA)*(-1+vB)*eta*etad)-2*(-1+vB)*(-1+etad)*(4*etae+eta*etad*(-1+(-1+vA)*etae))*tauB)/(8+(-1+vB)*eta*etad**2*(-3+vA+2*tauB)+2*etad*(-2+eta*(-2+vA+tauB)+vB*(2+eta-eta*tauB)))]
])
gamma2A = sympy.Matrix([
    [(2+2*(-1+vA)*etae+eta*etad*(-1+vA+2*etae-2*vA*etae))/(2+(-1+vA)*eta*etad), 0],
    [0, (2+2*(-1+vA)*etae+eta*etad*(-1+vA+2*etae-2*vA*etae))/(2+(-1+vA)*eta*etad)]
])
gamma2B = sympy.Matrix([
    [(-2+(-1+vB)*eta*etad*(-1+tauB)-2*(-1+vB)*etae*tauB)/(-2+(-1+vB)*eta*etad*(-1+tauB)), 0],
    [0, (-2+(-1+vB)*eta*etad*(-1+tauB)-2*(-1+vB)*etae*tauB)/(-2+(-1+vB)*eta*etad*(-1+tauB))]
])
gamma4A = sympy.Matrix([
    [(2+2*(-1+vA)*etae+eta*etad*(-1+vA+2*etae-2*vA*etae))/(2+(-1+vA)*eta*etad), 0],
    [0, (2+2*(-1+vA)*etae+eta*etad*(-1+vA+2*etae-2*vA*etae))/(2+(-1+vA)*eta*etad)]
])
gamma4B = sympy.Matrix([
    [(2+(-1+vB)*eta*etad**2*(-1+tauB)+2*(-1+vB)*etae*tauB-(-1+vB)*etad*(-1+eta*(-1+tauB)+2*etae*tauB))/(2-(-1+vB)*etad*(-1+eta*(-1+tauB))+(-1+vB)*eta*etad**2*(-1+tauB)), 0],
    [0, (2+(-1+vB)*eta*etad**2*(-1+tauB)+2*(-1+vB)*etae*tauB-(-1+vB)*etad*(-1+eta*(-1+tauB)+2*etae*tauB))/(2-(-1+vB)*etad*(-1+eta*(-1+tauB))+(-1+vB)*eta*etad**2*(-1+tauB))]
])


d = 2*sympy.Matrix([
    [a],
    [0],
    [b],
    [0]
])

dA = 2*sympy.Matrix([
    [a],
    [0]
])
dB = 2*sympy.Matrix([
    [b],
    [0]
])

P1 = sympy.lambdify([vA, vB, tauB, eta, etae, etad, pd, pDC, a, b], p1)
P2 = sympy.lambdify([vA, vB, tauB, eta, etae, etad, pd, pDC, a, b], p2)
P3 = sympy.lambdify([vA, vB, tauB, eta, etae, etad, pd, pDC, a, b], p3)
P4 = sympy.lambdify([vA, vB, tauB, eta, etae, etad, pd, pDC, a, b], p4)
P = sympy.lambdify([vA, vB, tauB, eta, etae, etad, pd, pDC, a, b], p)
G1 = sympy.lambdify([vA, vB, tauB, eta, etae, etad, pd, pDC, a, b], gamma1)
G2 = sympy.lambdify([vA, vB, tauB, eta, etae, etad, pd, pDC, a, b], gamma2)
G3 = sympy.lambdify([vA, vB, tauB, eta, etae, etad, pd, pDC, a, b], gamma3)
G4 = sympy.lambdify([vA, vB, tauB, eta, etae, etad, pd, pDC, a, b], gamma4)
G1A = sympy.lambdify([vA, vB, tauB, eta, etae, etad, pd, pDC, a, b], gamma1A)
G2A = sympy.lambdify([vA, vB, tauB, eta, etae, etad, pd, pDC, a, b], gamma2A)
G3A = sympy.lambdify([vA, vB, tauB, eta, etae, etad, pd, pDC, a, b], gamma3A)
G4A = sympy.lambdify([vA, vB, tauB, eta, etae, etad, pd, pDC, a, b], gamma4A)
G1B = sympy.lambdify([vA, vB, tauB, eta, etae, etad, pd, pDC, a, b], gamma1B)
G2B = sympy.lambdify([vA, vB, tauB, eta, etae, etad, pd, pDC, a, b], gamma2B)
G3B = sympy.lambdify([vA, vB, tauB, eta, etae, etad, pd, pDC, a, b], gamma3B)
G4B = sympy.lambdify([vA, vB, tauB, eta, etae, etad, pd, pDC, a, b], gamma4B)
D = sympy.lambdify([vA, vB, tauB, eta, etae, etad, pd, pDC, a, b], d)
DA = sympy.lambdify([vA, vB, tauB, eta, etae, etad, pd, pDC, a, b], dA)
DB = sympy.lambdify([vA, vB, tauB, eta, etae, etad, pd, pDC, a, b], dB)


def ProbDist(vA, vB, tauB, eta, etae, etad, pd, pDC, a, b):
  probdist = []
  Q1 = P1(vA, vB, tauB, eta, etae, etad, pd, pDC, a, b)
  Q2 = P2(vA, vB, tauB, eta, etae, etad, pd, pDC, a, b)
  Q3 = P3(vA, vB, tauB, eta, etae, etad, pd, pDC, a, b)
  Q4 = P4(vA, vB, tauB, eta, etae, etad, pd, pDC, a, b)
  Q = Q1-Q2-Q3+Q4

  g1 = G1(vA, vB, tauB, eta, etae, etad, pd, pDC, a, b)
  g2 = G2(vA, vB, tauB, eta, etae, etad, pd, pDC, a, b)
  g3 = G3(vA, vB, tauB, eta, etae, etad, pd, pDC, a, b)
  g4 = G4(vA, vB, tauB, eta, etae, etad, pd, pDC, a, b)
  g1A = G1A(vA, vB, tauB, eta, etae, etad, pd, pDC, a, b)
  g2A = G2A(vA, vB, tauB, eta, etae, etad, pd, pDC, a, b)
  g3A = G3A(vA, vB, tauB, eta, etae, etad, pd, pDC, a, b)
  g4A = G4A(vA, vB, tauB, eta, etae, etad, pd, pDC, a, b)
  g1B = G1B(vA, vB, tauB, eta, etae, etad, pd, pDC, a, b)
  g2B = G2B(vA, vB, tauB, eta, etae, etad, pd, pDC, a, b)
  g3B = G3B(vA, vB, tauB, eta, etae, etad, pd, pDC, a, b)
  g4B = G4B(vA, vB, tauB, eta, etae, etad, pd, pDC, a, b)

  dab = D(vA, vB, tauB, eta, etae, etad, pd, pDC, a, b)
  da = DA(vA, vB, tauB, eta, etae, etad, pd, pDC, a, b)
  db = DB(vA, vB, tauB, eta, etae, etad, pd, pDC, a, b)

  q0A = (1-pDC)/Q*(2*Q1/sympy.sqrt((Omega2*(g1A+I2)*(Omega2.transpose())).det())*sympy.exp(-0.5*(Omega2*da).transpose()*(Omega2*(g1A+I2)*(Omega2.transpose())).inv()*(Omega2*da))\
                  -2*Q2/sympy.sqrt((Omega2*(g2A+I2)*(Omega2.transpose())).det())*sympy.exp(-0.5*(Omega2*da).transpose()*(Omega2*(g2A+I2)*(Omega2.transpose())).inv()*(Omega2*da))\
                  -2*Q3/sympy.sqrt((Omega2*(g3A+I2)*(Omega2.transpose())).det())*sympy.exp(-0.5*(Omega2*da).transpose()*(Omega2*(g3A+I2)*(Omega2.transpose())).inv()*(Omega2*da))\
                  +2*Q4/sympy.sqrt((Omega2*(g4A+I2)*(Omega2.transpose())).det())*sympy.exp(-0.5*(Omega2*da).transpose()*(Omega2*(g4A+I2)*(Omega2.transpose())).inv()*(Omega2*da)))[0]
  q0B = (1-pDC)/Q*(2*Q1/sympy.sqrt((Omega2*(g1B+I2)*(Omega2.transpose())).det())*sympy.exp(-0.5*(Omega2*db).transpose()*(Omega2*(g1B+I2)*(Omega2.transpose())).inv()*(Omega2*db))\
                  -2*Q2/sympy.sqrt((Omega2*(g2B+I2)*(Omega2.transpose())).det())*sympy.exp(-0.5*(Omega2*db).transpose()*(Omega2*(g2B+I2)*(Omega2.transpose())).inv()*(Omega2*db))\
                  -2*Q3/sympy.sqrt((Omega2*(g3B+I2)*(Omega2.transpose())).det())*sympy.exp(-0.5*(Omega2*db).transpose()*(Omega2*(g3B+I2)*(Omega2.transpose())).inv()*(Omega2*db))\
                  +2*Q4/sympy.sqrt((Omega2*(g4B+I2)*(Omega2.transpose())).det())*sympy.exp(-0.5*(Omega2*db).transpose()*(Omega2*(g4B+I2)*(Omega2.transpose())).inv()*(Omega2*db)))[0]
  q00 = (1-pDC)**2/Q*(4*Q1/sympy.sqrt((Omega4*(g1+I4)*(Omega4.transpose())).det())*sympy.exp(-0.5*(Omega4*dab).transpose()*(Omega4*(g1+I4)*(Omega4.transpose())).inv()*(Omega4*dab))\
                     -4*Q2/sympy.sqrt((Omega4*(g2+I4)*(Omega4.transpose())).det())*sympy.exp(-0.5*(Omega4*dab).transpose()*(Omega4*(g2+I4)*(Omega4.transpose())).inv()*(Omega4*dab))\
                     -4*Q3/sympy.sqrt((Omega4*(g3+I4)*(Omega4.transpose())).det())*sympy.exp(-0.5*(Omega4*dab).transpose()*(Omega4*(g3+I4)*(Omega4.transpose())).inv()*(Omega4*dab))\
                     +4*Q4/sympy.sqrt((Omega4*(g4+I4)*(Omega4.transpose())).det())*sympy.exp(-0.5*(Omega4*dab).transpose()*(Omega4*(g4+I4)*(Omega4.transpose())).inv()*(Omega4*dab)))[0]
  q01 = q0A-q00
  q10 = q0B-q00
  q11 = 1-q0A-q0B+q00

  probdist = [q0A, q0B, q00, q01, q10, q11, Q]
  return probdist

def objective(ti, q):
    obj = 0.0
    F = [A[0][0], 1 - A[0][0]]                # POVM for Alices key gen measurement
    for a in range(A_config[0]):
        b = (a + 1) % 2                     # (a + 1 mod 2)
        M = (1-q) * F[a] + q * F[b]         # Noisy preprocessing povm element
        obj += M * (Z[a] + Dagger(Z[a]) + (1-ti)*Dagger(Z[a])*Z[a]) + ti*Z[a]*Dagger(Z[a])

    return obj

def compute_entropy(SDP, q):
    ck = 0.0        # kth coefficient
    ent = 0.0        # lower bound on H(A|X=0,E)

    if KEEP_M:
        num_opt = len(T)
    else:
        num_opt = len(T) - 1
        ent = 2 * q * (1-q) * W[-1] / log(2)

    for k in range(num_opt):
        ck = W[k]/(T[k] * log(2))

        # Get the k-th objective function
        new_objective = objective(T[k], q)

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

def cond_ent(joint, marg):
    hab, hb = 0.0, 0.0

    for prob in joint:
        if 0.0 < prob < 1.0:
            hab += -prob*log2(prob)

    for prob in marg:
        if 0.0 < prob < 1.0:
            hb += -prob*log2(prob)

    return hab - hb

def HAgB(sys, eta, etae, q):
    # Computes H(A|B) required for rate
    [id, sx, sy, sz] = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
    [tauB, a0, a1, b0, b1, b2] = sys[:]

    # joint distribution
    p00 = ProbDist(vA, vB, tauB, eta, etae, etad, pd, pDC, a0, b2)[2]
    p01 = ProbDist(vA, vB, tauB, eta, etae, etad, pd, pDC, a0, b2)[3]
    p10 = ProbDist(vA, vB, tauB, eta, etae, etad, pd, pDC, a0, b2)[4]
    p11 = ProbDist(vA, vB, tauB, eta, etae, etad, pd, pDC, a0, b2)[5]

    pp00 = (1-q) * p00 + q * p10
    pp01 = (1-q) * p01 + q * p11
    pp10 = (1-q) * p10 + q * p00
    pp11 = (1-q) * p11 + q * p01

    pb0 = ProbDist(vA, vB, tauB, eta, etae, etad, pd, pDC, a0, b2)[1]
    pb1 = 1-pb0

    qjoint = [pp00, pp01, pp10, pp11]
    qmarg = [pb0, pb1]

    return cond_ent(qjoint, qmarg)

def compute_rate(SDP, sys, eta, etae, q):
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
    ent = compute_entropy(SDP, q)
    err = HAgB(sys, eta, etae, q)
    return ent - err

def compute_dual_vector(SDP, q):
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
        ent = 2 * q * (1-q) * W[-1] / log(2)

    # Compute entropy and build dual vector from each sdp solved
    for k in range(num_opt):
        ck = W[k]/(T[k] * log(2))

        # Get the k-th objective function
        new_objective = objective(T[k], q)

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

def score_constraints(sys,eta, etae):
    """
    Returns moment equality constraints generated by the two-qubit system specified by sys.
    In particular implements the constraints p(00|xy) and p(0|x), p(0|y) for each x,y.
    """

    [tauB, a0, a1, b0, b1, b2] = sys[:]

    a_list = [a0, a1]
    b_list = [b0, b1]


    # Now collect the constraints subject to the inefficient detection distribution
    constraints = []
    for x in range(2):
        for y in range(2):
               constraints += [A[x][0]*B[y][0] - ProbDist(vA, vB, tauB, eta, etae, etad, pd, pDC, a_list[x], b_list[y])[2]]

    # Marginal constraints
    constraints += [A[0][0] - ProbDist(vA, vB, tauB, eta, etae, etad, pd, pDC, a_list[0], 0)[0]]
    constraints += [B[0][0] - ProbDist(vA, vB, tauB, eta, etae, etad, pd, pDC, 0, b_list[0])[1]]
    constraints += [A[1][0] - ProbDist(vA, vB, tauB, eta, etae, etad, pd, pDC, a_list[1], 0)[0]]
    constraints += [B[1][0] - ProbDist(vA, vB, tauB, eta, etae, etad, pd, pDC, 0, b_list[1])[1]]

    return constraints[:]

def sys2vec(sys, eta = 1.0):
    """
    Returns a vector of probabilities determined from the system in the same order as specified
    in the function score_constraints()

        sys    --     system parameters
        eta    --     detection efficiency
    """
    # Get the system from the parameters

    [tauB, a0, a1, b0, b1, b2] = sys[:]

    a_list = [a0, a1]
    b_list = [b0, b1]


    # Now collect the constraints subject to the inefficient detection distribution
    vec = []
    for x in range(2):
        for y in range(2):
               vec += [ProbDist(vA, vB, tauB, eta, etae, etad, pd, pDC, a_list[x], b_list[y])[2]]

    # Marginal constraints
    vec += [ProbDist(vA, vB, tauB, eta, etae, etad, pd, pDC, a_list[0], 0)[0]]
    vec += [ProbDist(vA, vB, tauB, eta, etae, etad, pd, pDC, 0, b_list[0])[1]]
    vec += [ProbDist(vA, vB, tauB, eta, etae, etad, pd, pDC, a_list[1], 0)[0]]
    vec += [ProbDist(vA, vB, tauB, eta, etae, etad, pd, pDC, 0, b_list[1])[1]]

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

def optimise_sys(SDP, sys, eta, etae, q):
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

    [tauB, a0, a1, b0, b1, b2] = sys[:]
    psucc = ProbDist(vA, vB, tauB, eta, etae, etad, pd, pDC, 0, 0)[6]

    # Loop until we converge on something
    while(NEEDS_IMPROVING):
        # On the first loop we just solve and extract the dual vector
        if not FIRST_PASS:
            # Here we optimize the dual vector
            # The distribution associated with the improved system
            pstar = sys2vec(improved_sys[:], eta)

            # function to optimize parameters over
            def f0(x):
                #x is sys that we are optimizing
                p = sys2vec(x, eta)
                return (-np.dot(p, dual_vec) + HAgB(x, eta, etae, q))*psucc

            # Bounds on the parameters of sys
            bounds = [[0,1]]+ [[-5,5]]+ [[-5,5]]+ [[-5,5]]+ [[-5,5]]+ [[-5,5]]
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
        dual_vec, new_ent = compute_dual_vector(SDP, q)
        new_rate = new_ent - HAgB(improved_sys[:], eta, etae, q)

        if not FIRST_PASS:
            if new_rate < best_rate + best_rate*EPS_M or new_rate < best_rate + EPS_A :
                NEEDS_IMPROVING = False
        else:
            # If first run through then this is the initial entropy
            starting_rate = new_rate
            best_rate = new_rate
            FIRST_PASS = False

        if new_rate > best_rate:
            print('Optimizing sys (eta, q) =', (eta,q), ' ... ', starting_rate, '->', new_rate)
            print(improved_sys)
            best_rate = new_rate
            best_sys = improved_sys[:]


    return best_rate, best_sys[:]

def optimise_q(SDP, sys, eta, etae, q):
    """
    Optimizes the choice of q.

        SDP    --    sdp relaxation object
        sys    --    parameters of system that are optimized
        eta --     detection efficiency
        q     --     bitflip probability

    This function can probably be improved to make the search a bit more efficient and fine grained.
    """
    q_eps = 0.005    # Can be tuned
    q_eps_min = 0.001

    opt_q = q
    rate = compute_rate(SDP, sys, eta, etae, q) # Computes rate for given q
    starting_rate = rate

    # We check if we improve going left
    if q - q_eps < 0:
        LEFT = 0
    else:
        new_rate = compute_rate(SDP, sys, eta, etae, opt_q - q_eps)
        if new_rate > rate:
            opt_q = opt_q - q_eps
            rate = new_rate
            print('Optimizing q (eta,q) =', (eta, opt_q), ' ... ', starting_rate, '->', rate)
            LEFT = 1
        else:
            LEFT = 0


    def next_q(q0, step_size):
        q1 = q0 + ((-1)**LEFT) * step_size
        if q1 >= 0 and q1 <= 0.5:
            return q1
        elif step_size/2 >= q_eps_min:
            return next_q(q0, step_size/2)
        else:
            return -1


    STILL_OPTIMIZING = 1

    while STILL_OPTIMIZING:
        # define the next q
        new_q = next_q(opt_q, q_eps)
        if new_q < 0:
            break

        #compute the rate
        new_rate = compute_rate(SDP, sys, eta, etae, new_q)

        if new_rate > rate:
            opt_q = new_q
            rate = new_rate
            print('Optimizing q (eta,q) =', (eta, opt_q), ' ... ', starting_rate, '->', rate)
        else:
            # If we didn't improve try shortening the distance
            q_eps = q_eps / 2
            if q_eps < q_eps_min:
                STILL_OPTIMIZING = 0

    return rate, opt_q

def optimise_rate(SDP, sys, eta, etae, q):
    """
    Iterates between optimizing sys and optimizing q in order to optimize overall rate.
    """

    STILL_OPTIMIZING = 1

    best_rate = compute_rate(SDP, sys, eta, etae, q)
    best_sys = sys[:]
    best_q = q

    while STILL_OPTIMIZING:
        _, new_sys = optimise_sys(SDP, best_sys[:], eta, etae, best_q)
        new_rate, new_q = optimise_q(SDP, new_sys[:], eta, etae, best_q)


        if (new_rate < best_rate + best_rate*EPS_M) or (new_rate < best_rate + EPS_A):
            STILL_OPTIMIZING = 0

        if new_rate > best_rate:
            best_rate = new_rate
            best_sys = new_sys[:]
            best_q = new_q

    return best_rate, best_sys, best_q

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

vA = 1.04
vB = 1.04
tauB = 0.98
eta = 1
etad = 1
etae = 1
pd = 0
pDC = 0


# Defining the test sys
test_sys = [tauB, 0.0018415210676123004, -0.3244847812422259, 0.20834559167471495, -0.07596149444947721, 0.0018407476014714234]
score_cons = score_constraints(test_sys,eta, etae)
test_q = 0

ops = ncp.flatten([A,B,Z])        # Base monomials involved in problem
obj = objective(1,test_q)    # Placeholder objective function

sdp = ncp.SdpRelaxation(ops, verbose = VERBOSE-1, normalized=True, parallel=0)
sdp.get_relaxation(level = LEVEL,
                    equalities = op_eqs[:],
                    inequalities = op_ineqs[:],
                    momentequalities = moment_eqs[:] + score_cons,
                    momentinequalities = moment_ineqs[:],
                    objective = obj,
                    substitutions = substitutions,
                    extramonomials = extra_monos)

new_rate, new_sys, new_q = optimise_rate(sdp, test_sys, eta, etae, test_q)
print(new_rate)
print(new_sys)
print(new_q)