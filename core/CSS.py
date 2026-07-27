import numpy as np
import galois
import stim
from ldpc import BpOsdDecoder
from itertools import product
import re
from docplex.mp.model import Model
from docplex.mp.progress import ProgressClock
import cplex
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from sympy import Poly, symbols

def solve_gf2(A: galois.FieldArray, b: galois.FieldArray):
    """
    Ax = b over GF(2)
    Returns: (particular_solution, null_space_basis)
             particular_solution이 None이면 해 없음
    """
    GF2 = type(A)
    m, n = A.shape

    # augmented matrix [A | b]
    Ab = GF2(np.hstack([A, b.reshape(-1, 1)]))
    rref = Ab.row_reduce()

    # pivot columns 찾기
    pivots = {}
    for i in range(m):
        for j in range(n):
            if rref[i, j] == 1:
                pivots[i] = j
                break

    # consistency check: non-pivot row에서 b 열이 1이면 해 없음
    for i in range(m):
        if i not in pivots and rref[i, -1] == 1:
            return None, None

    # particular solution: free variable = 0
    x_particular = GF2(np.zeros(n, dtype=int))
    for i, j in pivots.items():
        x_particular[j] = rref[i, -1]

    # null space basis (homogeneous solution)
    x_null = A.null_space()

    return x_particular, x_null

class CSS:
    def __init__(self, Gx, Gz):
        self.n = Gx.shape[1]
        self.Gx = Gx
        self.Gz = Gz
        self.F = galois.GF(2) # Field
        self.get_center()
        self.k = self.calculate_k()
        self.logical_x_repr = [None] * self.k    # X logical representatives
        self.logical_z_repr = [None] * self.k    # Z logical representatives
        self.d = None
        self.dx = None
        self.dz = None


    def get_center(self,):
        s = (self.Gz @ self.Gx.T).null_space()
        self.Hx = s.T @ self.Gx
        s = (self.Gx @ self.Gz.T).null_space()
        self.Hz = s.T @ self.Gz
    def get_density(self,):
        # check의 max weight와, 각 qubit이 속한 check의 max 개수
        max_check_weight_x = max([np.sum(np.asarray(row)) for row in self.Hx])
        max_check_weight_z = max([np.sum(np.asarray(row)) for row in self.Hz])
        max_qubit_degree_x = max([np.sum(np.asarray(self.Hx[:,i])) for i in range(self.n)])
        max_qubit_degree_z = max([np.sum(np.asarray(self.Hz[:,i])) for i in range(self.n)])
        print("Max check weight X:", max_check_weight_x)
        print("Max check weight Z:", max_check_weight_z)
        print("Max qubit degree X:", max_qubit_degree_x)
        print("Max qubit degree Z:", max_qubit_degree_z)
        return max(max_check_weight_x, max_check_weight_z), max(max_qubit_degree_x, max_qubit_degree_z)
    def calculate_k(self,):
        rank_x = np.linalg.matrix_rank(self.Hx)
        rank_z = np.linalg.matrix_rank(self.Hz)
        self.k = int(self.n - (rank_x + rank_z))
        return int(self.n - (rank_x + rank_z))

    def _extract_quotient_basis(self, G_opp, G_self):
        """
        Ker(G_opp) / Rowspace(G_self) 의 기저를 추출하는 헬퍼 함수
        """
        # G_opp와 commute하는 space의 basis
        ker_opp = G_opp.null_space()
        # print("ker rank:",np.linalg.matrix_rank(ker_opp))
        # basis of G_self
        basis_self = G_self.row_space()
        # print("rank:",np.linalg.matrix_rank(basis_self))
        logicals = []
        current_span = basis_self
        # ker 원소 중 Gauge 기저와 독립적인 것들만 선별 (Stacking Trick)
        for vec in ker_opp:
            # 현재까지의 span(Gauge + 찾은 Logicals)에 새로운 벡터를 추가해봄
            if len(current_span) == 0:
                test_matrix = vec.reshape(1, -1)
            else:
                test_matrix = np.vstack([current_span, vec])

            # 랭크가 증가한다면 새로운 Logical 기저임
            if np.linalg.matrix_rank(test_matrix) > np.linalg.matrix_rank(current_span):
                logicals.append(vec)
                current_span = self.F(test_matrix)

        return self.F(logicals) if logicals else self.F.Zeros((0, self.n))
    
    def commutativity_check(self,):
        # Hx @ Hz^T == 0 이어야 함.
        if not np.all(np.dot(self.Hx, self.Hz.T) == 0):
            return False
        else:
            return True

    def set_logical_representative(self, type: str, idx: int, repr_vec: list[int]):
        repr_vec = self.F(repr_vec)
        if type == "X":
            # repr_vec가 Gz와 commute하는지, Gx의 row space에 없는지 확인
            assert np.all(np.dot(repr_vec, self.Gz.T) == 0), "Logical representative should commute with all Z-checks."
            assert np.linalg.matrix_rank(self.F(np.vstack((self.Gx, repr_vec)))) > np.linalg.matrix_rank(self.F(self.Gx)), "Logical representative should not be in the row space of Gx."
            # Z-logic representatives와의 anti-commute, commute 조건 체크 (None이 아닌 것들에 한해서)
            for j in range(self.k):
                if self.logical_z_repr[j] is not None:
                    comm = np.dot(repr_vec, self.logical_z_repr[j])
                    if idx == j:
                        assert comm == 1, f"Logical X_{idx} and Z_{j} should anti-commute."
                    else:
                        assert comm == 0, f"Logical X_{idx} and Z_{j} should commute."
            self.logical_x_repr[idx] = repr_vec
        elif type == "Z":
            # repr_vec가 Gx와 commute하는지, Gz의 row space에 없는지 확인
            assert np.all(np.dot(repr_vec, self.Gx.T) == 0), "Logical representative should commute with all X-checks."
            assert np.linalg.matrix_rank(self.F(np.vstack((self.Gz, repr_vec)))) > np.linalg.matrix_rank(self.F(self.Gz)), "Logical representative should not be in the row space of Gz."
            # X-logic representatives와의 anti-commute, commute 조건 체크 (None이 아닌 것들에 한해서)
            for j in range(self.k):
                if self.logical_x_repr[j] is not None:
                    comm = np.dot(repr_vec, self.logical_x_repr[j])
                    if idx == j:
                        assert comm == 1, f"Logical Z_{idx} and X_{j} should anti-commute."
                    else:
                        assert comm == 0, f"Logical Z_{idx} and X_{j} should commute."
            self.logical_z_repr[idx] = repr_vec
        else:
            raise ValueError("Invalid logical operator type. Expected 'X' or 'Z'.")
    
    def is_complete_logical_set(self,):
        # logical representatives가 모두 채워져 있는지 확인
        for i in range(self.k):
            if self.logical_x_repr[i] is None or self.logical_z_repr[i] is None:
                return False
        return True

    def set_missing_logicals(self,):
        # logical representatives 중에서 None인 것들 찾아서 채워넣음.
        # dual logical operator가 None이 아닌 것 우선으로!
        for i in range(self.k):
            # dual이 None이 아닌 경우 우선 처리
            if self.logical_x_repr[i] is None and self.logical_z_repr[i] is not None:
                self._set_missing_logical("X", i)
            if self.logical_z_repr[i] is None and self.logical_x_repr[i] is not None:
                self._set_missing_logical("Z", i)
        # dual도 None인 경우 처리
        for i in range(self.k):
            if self.logical_x_repr[i] is None:
                self._set_missing_logical("X", i)
            if self.logical_z_repr[i] is None:
                self._set_missing_logical("Z", i)

    def _set_missing_logical(self, typ:str, idx:int):
        # dual logical operator가 None이라면 dual check와 commute하고 primal check의 span에 안들어가는 것 찾음.
        # print(f"set logical {typ} {idx}")
        if typ == "X":
            G_opp = self.Gz
            G_self = self.Gx
            primal_logic_ops = self.logical_x_repr
            dual_logic_ops = self.logical_z_repr    # 적어도 하나의 Z-logic op과 anti-commute해야 non-trivial logical op.임.
        elif typ == "Z":
            G_opp = self.Gx
            G_self = self.Gz
            primal_logic_ops = self.logical_z_repr
            dual_logic_ops = self.logical_x_repr
        A = G_opp
        b = self.F.Zeros((G_opp.shape[0],), dtype=int)
        # for primal_op in primal_logic_ops:    # 제약조건 있어야할 이유 없음.
        #     if primal_op is not None:
        #         A = np.vstack((A, primal_op))
        #         b = np.hstack((b, self.F(0)))
        for i, dual_op in enumerate(dual_logic_ops):
            if dual_op is not None:
                A = np.vstack((A, dual_op))
                if i == idx:
                    b = np.hstack((b, self.F(1)))
                else:
                    b = np.hstack((b, self.F(0)))
        
        # Ax=b 의 nullspace의 basis중에 G_self의 row space에 없는 것 찾기 (반드시 존재함)
        x_part, x_null = solve_gf2(A, b)
        # print("x_part: ", x_part)
        # print("x_null: ", x_null)
        # x_part는 row vector
        # x_null의 행이 해
        for row in x_null:
            x = x_part + row
            # G_self의 row space에 없는지 확인
            if np.linalg.matrix_rank(self.F(np.vstack((G_self, x)))) == np.linalg.matrix_rank(self.F(G_self)) + 1:
                # print("Found missing logical operator:", x)
                if typ == "X":
                    self.logical_x_repr[idx] = self.F(x)
                elif typ == "Z":
                    self.logical_z_repr[idx] = self.F(x)
                return
            


    def set_logical_representatives(self, logical_x_repr: list[list], logical_z_repr: list[list]):
        

        assert len(logical_x_repr) == self.k, "Number of logical X representatives must be equal to k."
        assert len(logical_z_repr) == self.k, "Number of logical Z representatives must be equal to k."
        self.logical_x_repr = [self.F(op) for op in logical_x_repr]
        self.logical_z_repr = [self.F(op) for op in logical_z_repr]
        # check 만족하는지 확인
        for i in range(self.k):
            assert np.all(np.dot(self.logical_x_repr[i], self.Gz.T) == 0), f"Logical X_{i} should commute with all Z-checks."
            assert np.all(np.dot(self.logical_z_repr[i], self.Gx.T) == 0), f"Logical Z_{i} should commute with all X-checks."
        
        # logic op간 anti-commute, commute 조건 체크
        for i in range(self.k):
            for j in range(self.k):
                comm = np.dot(self.logical_x_repr[i], self.logical_z_repr[j])
                if i == j:
                    assert comm == 1, f"Logical X_{i} and Z_{j} should anti-commute."
                else:
                    assert comm == 0, f"Logical X_{i} and Z_{j} should commute."

    def calc_logical_representatives(self,):
        # logical X, Z commutation relation 만족하지 않을 수 있음
        # distance 계산에만 유효
        if self.logical_x_repr is None:
            # Lx = Ker(Gz) / Rowspace(Gx)
            self.logical_x_repr = self._extract_quotient_basis(self.Gz, self.Gx)
        if self.logical_z_repr is None:
            # Lz = Ker(Gx) / Rowspace(Gz)
            self.logical_z_repr = self._extract_quotient_basis(self.Gx, self.Gz)
    
    
    def _minimize_logical_op(self, l_type, support: galois.FieldArray):
        # l_type, support를 가지는 logical operator를 같은 type check곱해서 weight 최소화
        if l_type == "X":
            G_self = self.Gx
        elif l_type == "Z":
            G_self = self.Gz
        
        G = np.array(G_self, dtype=int)
        s = np.array(support, dtype=int)
        m, n = G.shape  # m: stabilizer 수, n: 큐비트 수

        model = Model(name="minimize_logical_op")

        # 변수
        r = model.binary_var_list(m, name="r")   # 어떤 stabilizer를 XOR할지
        y = model.binary_var_list(n, name="y")   # 결과 비트
        z = model.integer_var_list(n, lb=0, name="z")  # mod 2 slack

        # 제약: y[j] ≡ s[j] + sum_i G[i,j]*r[i]  (mod 2)
        for j in range(n):
            lhs = s[j] + model.sum(int(G[i, j]) * r[i] for i in range(m))
            model.add_constraint(lhs == y[j] + 2 * z[j])

        # 목적함수
        model.minimize(model.sum(y))

        sol = model.solve()
        print(sol.solve_details.status)
        if sol:
            result = np.array([int(sol.get_value(y[j])) for j in range(n)])
            return galois.GF(2)(result)
        else:
            return None



    def _find_min_op_weight(self, l_type):
        # l_type의 모든 logical operator 중 (up to multiplication with checks) 최소 weight
        if l_type == "X":
            G_opp = self.Gz
            G_self = self.Gx
            anti_logic_ops = self.logical_z_repr    # 적어도 하나의 Z-logic op과 anti-commute해야 non-trivial logical op.임.
        elif l_type == "Z":
            G_opp = self.Gx
            G_self = self.Gz
            anti_logic_ops = self.logical_x_repr

        model = Model(name="css_distance")
        e = model.binary_var_list(self.n, name='e')
        s = model.binary_var_list(self.k, name='s') # s_i: Z_i와 anti-commute하는지
        c = model.binary_var_list(self.n, name='c')

        z_comm = [model.integer_var(lb=0, ub=int(np.sum(np.asarray(G_opp[j])) // 2), name=f'z_comm_{j}') for j in range(G_opp.shape[0])]
        z_coset = [model.integer_var(lb=0, ub=int(np.sum(np.asarray(anti_logic_ops[j])) // 2), name=f'z_coset_{j}') for j in range(self.k)]
        model.minimize(model.sum(e))

        # Commute with G_opp
        # G_opp @ e = 0 (mod 2)
        for j in range(G_opp.shape[0]):
            model.add_constraint(model.sum(int(G_opp[j][i])*e[i] for i in range(self.n)) == 2*z_comm[j])

        # Logical syndrome (anti_logic_ops[j]랑 anti-commute하는지?)
        for j in range(self.k):
            lhs = model.sum(int(anti_logic_ops[j][i])*e[i] for i in range(self.n))
            model.add_constraint(lhs == 2*z_coset[j] + s[j])

        # Non-triviality:
        model.add_constraint(model.sum(s)>=1)
        sol = model.solve(log_output=False)
        status = model.solve_status
        # print(f"statue: {status}")
        solve_time = sol.solve_details.time
        # print(f"⏱️ CPLEX 순수 계산 시간: {solve_time:.4f} 초")
        # print(sol.get_values(e))

        if sol:
            return int(sol.objective_value)
        else:
            return None



    def calculate_distance(self,):
        self.calc_logical_representatives()
        self.calculate_k()
        # print(self.logical_x_repr)
        # print(self.logical_z_repr)
        self.dx = self._find_min_op_weight("X")
        self.dz = self._find_min_op_weight("Z")
        self.d = min(self.dx,self.dz)
        return self.d

    def get_distance(self,):
        if self.dx is None or self.dz is None:
            self.calculate_distance()
        return self.dx, self.dz

def _v(x):
    return "".join(str(int(b)) for b in np.asarray(x, dtype=int))
 
# Classical Linear Code
class LinearCode:
    def __init__(self, H):
        self.F = galois.GF(2)
        self.H = self.F(np.array(H))
        self.n = self.H.shape[1]
        self.k = self.calculate_k()
        self.G = self.calculate_generator_matrix()
 
    # ------------------------------------------------------------ 기본
    def calculate_k(self):
        return int(self.n - np.linalg.matrix_rank(self.H))
 
    def calculate_generator_matrix(self):
        self.G = self.H.null_space().row_reduce()
        return self.G
 
    def get_distance(self):
        k = self.k
        if k == 0:
            return float('inf')
        G = np.asarray(self.G, dtype=int)
        rows = [sum(1 << j for j in range(self.n) if G[r, j]) for r in range(k)]
        d, cur = self.n, 0
        for i in range(1, 1 << k):
            cur ^= rows[(i & -i).bit_length() - 1]
            w = cur.bit_count()
            if w < d:
                d = w
        return d
 
    @classmethod
    def from_generator(cls, G):
        F = galois.GF(2)
        return cls(np.asarray(F(np.asarray(G, dtype=int)).null_space(), dtype=int))
 
    # ------------------------------------------------------------ 구조
    @property
    def params(self):
        return (self.n, self.k, self.get_distance())
 
    def pivots(self):
        """G(RREF)의 pivot 열."""
        Gi = np.asarray(self.G, dtype=int)
        return [int(np.nonzero(r)[0][0]) for r in Gi if r.any()]
 
    def complement(self):
        """pivot 이 아닌 열 (logical 의 여집합 basis 좌표)."""
        piv = set(self.pivots())
        return [j for j in range(self.n) if j not in piv]
 
    def logicals(self):
        """logical basis 벡터 리스트 (G 의 행)."""
        return [self.G[i] for i in range(self.k)]
 
    def codewords(self):
        """(label, vector) 전체 codeword."""
        out = []
        for m in range(1 << self.k):
            w = self.F(np.zeros(self.n, dtype=int))
            lab = []
            for i in range(self.k):
                if m >> i & 1:
                    w = w + self.G[i]
                    lab.append(f"L{i+1}")
            out.append(("+".join(lab) if lab else "0", w))
        return out
 
    def decompose(self, vec):
        """codeword 를 logical basis 계수로 분해. codeword 가 아니면 None."""
        vec = self.F(np.asarray(vec, dtype=int))
        if np.any(self.H @ vec):
            return None
        aug = self.F(np.hstack((np.asarray(self.G.T, dtype=int),
                                np.asarray(vec, dtype=int).reshape(-1, 1))))
        R = np.asarray(aug.row_reduce(), dtype=int)
        c = np.zeros(self.k, dtype=int)
        for r in R:
            nz = np.nonzero(r[:self.k])[0]
            if len(nz):
                c[nz[0]] = r[self.k]
        return c
 
    def label(self, vec):
        c = self.decompose(vec)
        if c is None:
            return "(not a codeword)"
        t = " + ".join(f"L{i+1}" for i in range(self.k) if c[i])
        return t or "0"
 
    def report(self, name="code", show_codewords=True):
        print("=" * 60)
        print(f"{name} = [{self.n},{self.k},{self.get_distance()}]")
        print("=" * 60)
        print("H:")
        print(np.asarray(self.H, dtype=int))
        print(f"\nlogical basis ({self.k}개):")
        for i, L in enumerate(self.logicals()):
            print(f"   L{i+1} = {_v(L)}   (wt {int(np.asarray(L, dtype=int).sum())})")
        print(f"pivots {self.pivots()}, complement {self.complement()}")
        if show_codewords and self.k <= 6:
            print(f"\ncodeword {2**self.k}개:")
            for lab, w in self.codewords():
                print(f"   {lab:<12} = {_v(w)}  (wt {int(np.asarray(w, dtype=int).sum())})")
        print()


class HGPCode(CSS):
    """
    HGP(Ca, Cb).  1-dim chain: index1 = bit, index0 = check,  d = H.
 
        X check :  Ca1 (x) Cb1   (na*nb)   idx (i,j) -> i*nb + j
        qubit   :  Ca0 (x) Cb1   (ra*nb)   idx (b,j) -> b*nb + j            [block1]
                   Ca1 (x) Cb0   (na*rb)   idx (i,s) -> ra*nb + i*rb + s    [block2]
        Z check :  Ca0 (x) Cb0   (ra*rb)   idx (b,s) -> b*rb + s
 
        H_X = [ Ha^T (x) I_nb | I_na (x) Hb^T ]
        H_Z = [ I_ra (x) Hb   | Ha   (x) I_rb ]
 
        ker d1 = ker(Ha) (x) ker(Hb)                 dim = ka*kb
        k      = kaT*kb + ka*kbT
 
    X-logical families:
        F-I  :  ker(Ha) (x) e^{Cb0}   -> block2,  ka  * kbT
        F-II :  e^{Ca0} (x) ker(Hb)   -> block1,  kaT * kb
    logical index order: F-I 먼저 (m = p*kbT + q), 그 다음 F-II (m = ka*kbT + b*kb + t)
    """
 
    def __init__(self, Ca: LinearCode, Cb: LinearCode):
        self.Ca, self.Cb = Ca, Cb
        self.F = galois.GF(2)
 
        Ha = np.asarray(Ca.H, dtype=int)
        Hb = np.asarray(Cb.H, dtype=int)
        self.na, self.ra = Ca.n, Ha.shape[0]
        self.nb, self.rb = Cb.n, Hb.shape[0]
        na, ra, nb, rb = self.na, self.ra, self.nb, self.rb
 
        Hx = np.hstack([np.kron(Ha.T, np.eye(nb, dtype=int)),
                        np.kron(np.eye(na, dtype=int), Hb.T)])
        Hz = np.hstack([np.kron(np.eye(ra, dtype=int), Hb),
                        np.kron(Ha, np.eye(rb, dtype=int))])
 
        super().__init__(self.F(Hx % 2), self.F(Hz % 2))
        self.init_logical_representatives()
 
    # ---- index helpers (surgery 에서 raw 산술 안 하도록) -------------------
    def x_check_idx(self, i, j):  return i * self.nb + j            # Ca1 x Cb1
    def z_check_idx(self, b, s):  return b * self.rb + s            # Ca0 x Cb0
    def qubit1_idx(self, b, j):   return b * self.nb + j            # Ca0 x Cb1
    def qubit2_idx(self, i, s):   return self.ra * self.nb + i * self.rb + s
 
    @staticmethod
    def _e(length, idx):
        return np.eye(1, length, idx, dtype=int)
 
    def init_logical_representatives(self):
        self.CaT = LinearCode(np.asarray(self.Ca.H, dtype=int).T)
        self.CbT = LinearCode(np.asarray(self.Cb.H, dtype=int).T)
 
        na, ra, nb, rb = self.na, self.ra, self.nb, self.rb
        ka, kb = self.Ca.k, self.Cb.k
        kaT, kbT = self.CaT.k, self.CbT.k
 
        Ga = np.asarray(self.Ca.G, dtype=int)
        Gb = np.asarray(self.Cb.G, dtype=int)
        GaT = np.asarray(self.CaT.G, dtype=int)
        GbT = np.asarray(self.CbT.G, dtype=int)
 
        z1 = np.zeros((1, ra * nb), dtype=int)   # block1 zero
        z2 = np.zeros((1, na * rb), dtype=int)   # block2 zero
 
        # information bit = G(RREF) 의 pivot 열. 앞 k개라고 가정하면 안 됨.
        pa, pb   = self.Ca.pivots(),  self.Cb.pivots()
        paT, pbT = self.CaT.pivots(), self.CbT.pivots()
 
        # F-I : block2 (Ca1 x Cb0).  X = l_Ca (x) e_q ,  Z = e_p (x) l_CbT
        for p in range(ka):
            for q in range(kbT):
                m = p * kbT + q
                x = np.hstack([z1, np.kron(Ga[p:p+1], self._e(rb, pbT[q]))])
                z = np.hstack([z1, np.kron(self._e(na, pa[p]), GbT[q:q+1])])
                self.set_logical_representative("X", m, x.flatten())
                self.set_logical_representative("Z", m, z.flatten())
 
        # F-II : block1 (Ca0 x Cb1).  X = e_b (x) l_Cb ,  Z = l_CaT (x) e_t
        for b in range(kaT):
            for t in range(kb):
                m = ka * kbT + b * kb + t
                x = np.hstack([np.kron(self._e(ra, paT[b]), Gb[t:t+1]), z2])
                z = np.hstack([np.kron(GaT[b:b+1], self._e(nb, pb[t])), z2])
                self.set_logical_representative("X", m, x.flatten())
                self.set_logical_representative("Z", m, z.flatten())
 
class LPCode(CSS):
    """
        H_XR = [H_a\otimes_R I | I \otimes_R H_b]
        H_ZR = [I \otimes_R H_b* | H_a* \otimes_R I]

        A1 --(Ha)--> A0
        B1 --(Hb)--> B0

    """
    def __init__(self, Ha: list[list[Poly]], Hb: list[list[Poly]], R: Poly):
        self.Ha = Ha
        self.Hb = Hb
        a0 = len(Ha)
        a1 = len(Ha[0])
        b0 = len(Hb)
        b1 = len(Hb[0])

        self.R = R
        self.x = R.gen
        self.l = R.degree()
        self.F = galois.GF(2) # Field

        H_XR = self.hstack_ring(self.kron_ring(Ha, self.eye_ring(b0)), self.kron_ring(self.eye_ring(a0), Hb))
        H_ZR = self.hstack_ring(self.kron_ring(self.eye_ring(a1), self.conj(Hb)), self.kron_ring(self.conj(Ha), self.eye_ring(b1)))    
        # LPCode.print_ring_matrix(H_XR)
        # LPCode.print_ring_matrix(H_ZR)
        self.Hx = self.to_binary(H_XR)
        self.Hz = self.to_binary(H_ZR)

        # Call parent constructor after Hx and Hz are defined
        super().__init__(galois.GF(2)(self.Hx), galois.GF(2)(self.Hz))
    def print_ring_matrix(M):
        for row in M:
            print([str(p.as_expr()) for p in row])
    def poly_mod(self, p: Poly) -> Poly:
        """p mod R over F2"""
        return p.rem(self.R)

    def poly_mul(self, a: Poly, b: Poly) -> Poly:
        return self.poly_mod(Poly(a * b, self.x, modulus=2))

    # def poly_conj(self, a: Poly) -> Poly:
    #     """conjugate: x^i -> x^{l-i} mod R"""
    #     result = Poly(0, self.x, modulus=2)
    #     for monom, coeff in zip(a.monoms(), a.coeffs()):
    #         i = monom[0]
    #         result = Poly(result + coeff * self.x**(self.l - i), self.x, modulus=2)
    #     return self.poly_mod(result)
    def poly_conj(self, a: Poly) -> Poly:
        """conjugate: x^i -> x^{p-i mod p}"""
        coeffs = {}
        for monom, coeff in zip(a.monoms(), a.coeffs()):
            i = monom[0]
            new_exp = (self.l - i) % self.l
            coeffs[new_exp] = (coeffs.get(new_exp, 0) + int(coeff)) % 2
        expr = sum(c * self.x**e for e, c in coeffs.items() if c)
        if not expr:
            return Poly(0, self.x, modulus=2)
        return self.poly_mod(Poly(expr, self.x, modulus=2))

    def conj(self, M: list[list[Poly]]) -> list[list[Poly]]:
        """conjugate transpose of ring matrix"""
        nr, nc = len(M), len(M[0])
        return [[self.poly_conj(M[i][j]) for i in range(nr)] for j in range(nc)]
    
    def eye_ring(self, n: int) -> list[list[Poly]]:
        one = Poly(1, self.x, modulus=2)
        zero = Poly(0, self.x, modulus=2)
        return [[one if i == j else zero for j in range(n)] for i in range(n)]

    def kron_ring(self, A: list[list[Poly]], B: list[list[Poly]]) -> list[list[Poly]]:

        """A ⊗_R B: Kronecker product at ring level"""
        ra, ca = len(A), len(A[0])
        rb, cb = len(B), len(B[0])
        result = [[Poly(0, self.x, modulus=2)] * (ca * cb) for _ in range(ra * rb)]
        for i in range(ra):
            for j in range(ca):
                for k in range(rb):
                    for l in range(cb):
                        result[i*rb + k][j*cb + l] = self.poly_mul(A[i][j], B[k][l])
        return result

    def hstack_ring(self, A: list[list[Poly]], B: list[list[Poly]]) -> list[list[Poly]]:
        """horizontal concatenation at ring level"""
        return [row_a + row_b for row_a, row_b in zip(A, B)]

    def poly_to_matrix(self, p: Poly) -> np.ndarray:
        l = self.l
        mat = np.zeros((l, l), dtype=np.uint8)
        for monom, coeff in zip(p.monoms(), p.coeffs()):
            k = monom[0]
            for j in range(l):
                mat[(j + k) % l, j] ^= int(coeff) % 2
        return mat
        
    def to_binary(self, M: list[list[Poly]]) -> np.ndarray:
        n_r = len(M)
        n_c = len(M[0])
        l = self.l
        result = np.zeros((n_r * l, n_c * l), dtype=np.uint8)
        for i in range(n_r):
            for j in range(n_c):
                result[i*l:(i+1)*l, j*l:(j+1)*l] = self.poly_to_matrix(M[i][j])
        return result
    
class RotatedSurfaceCode(CSS):

    def __init__(self, d: int):
        assert d%2==1
        self.n = d**2
        self.d = d
        Hx = []
        Hz = []
        # Z-check
        # upper boundary
        for x in range(0,d-1,2):
            chk = [0]*self.n
            chk[self._coord2idx(x,0)] = 1
            chk[self._coord2idx(x+1,0)] = 1
            Hz.append(chk)
        # inside
        for y in range(0,d-1):
            if y%2==0:
                start=1
            else:
                start=0
            for x in range(start,d-1,2):
                chk = [0]*self.n
                chk[self._coord2idx(x,y)] = 1
                chk[self._coord2idx(x+1,y)] = 1
                chk[self._coord2idx(x,y+1)] = 1
                chk[self._coord2idx(x+1,y+1)] = 1
                Hz.append(chk)
        # lower boundary
        for x in range(1,d-1,2):
            chk = [0]*self.n
            chk[self._coord2idx(x,d-1)] = 1
            chk[self._coord2idx(x+1,d-1)] = 1
            Hz.append(chk)
        # X-check
        for y in range(0,d-1):
            if y%2 == 0:
                for x in range(0,d-1,2):
                    chk = [0]*self.n
                    chk[self._coord2idx(x,y)] = 1
                    chk[self._coord2idx(x+1,y)] = 1
                    chk[self._coord2idx(x,y+1)] = 1
                    chk[self._coord2idx(x+1,y+1)] = 1
                    Hx.append(chk)
                # right boundary
                chk = [0]*self.n
                chk[self._coord2idx(d-1,y)] = 1
                chk[self._coord2idx(d-1,y+1)] = 1
                Hx.append(chk)
            else:
                # left boundary
                chk = [0]*self.n
                chk[self._coord2idx(0,y)] = 1
                chk[self._coord2idx(0,y+1)] = 1
                Hx.append(chk)

                for x in range(1,d-1,2):
                    chk = [0]*self.n
                    chk[self._coord2idx(x,y)] = 1
                    chk[self._coord2idx(x+1,y)] = 1
                    chk[self._coord2idx(x,y+1)] = 1
                    chk[self._coord2idx(x+1,y+1)] = 1
                    Hx.append(chk)
        
        super().__init__(galois.GF(2)(np.array(Hx)),galois.GF(2)(np.array(Hz)))


    def _coord2idx(self, x: int, y: int) -> int:
        # x,y = 0,1,...,d-1
        return self.d*y+x

#####################

class BBCode(CSS):
    def __init__(self, l, m, a_poly, b_poly):

        """
        l, m: 그리드 크기
        a_poly, b_poly: a(x,y), b(x,y)
        e.g. a_poly = [(x1, y1), (x2, y2), ... ] => a(x,y) = x**x1*y**y1 + x**x2*y**y2 + ...

        """
        self.l = l
        self.m = m
        self.n = 2*l*m
        self.F = galois.GF(2) # Field
        self.a_poly = a_poly
        self.b_poly = b_poly

        self.A, self.Ai = self._make_cyclic_matrix(a_poly)
        self.B, self.Bi = self._make_cyclic_matrix(b_poly)

        # Hx = [A | B], Hz = [B^T | A^T]
        self.Hx = self.F(np.hstack((self.A, self.B)))
        self.Hz = self.F(np.hstack((self.B.T, self.A.T)))

        # Call parent constructor after Hx and Hz are defined
        super().__init__(self.Hx, self.Hz)

    def _S(self, size, shift):
        mat = self.F.Zeros((size, size), dtype=int)
        for i in range(size):
            mat[i, (i+shift)%size] = 1
        return mat

    def _x(self, pow):
        # return x^pow, where x = S_l \otimes I_m
        return self.F(np.kron(self._S(self.l, pow), np.eye(self.m, dtype=int)))

    def _y(self, pow):
        # return y^pow, where y = I_l \otimes S_m
        return self.F(np.kron(np.eye(self.l, dtype=int), self._S(self.m, pow)))

    def _make_cyclic_matrix(self, poly):
        size = self.l * self.m
        mat = self.F.Zeros((size, size), dtype=int)
        res = []
        for i, j in poly:
            val = self.F(self._x(i) @ self._y(j))
            res.append(val)
            mat += val
        return mat, res

    def get_col_idx(self, mat_str, row_idx):
        # e.g. mat_str=AT1이라면, A_1^T의 row_idx행의 1인 column을 반환.
        assert 0<=row_idx<self.l*self.m
        assert re.match(r"^[AB]T?[1-3]$", mat_str), "Invalid mat format. Expected 'A' or 'B' followed by optional 'T' and a digit from 1 to 3 (e.g., 'A1', 'AT1', 'B2')."
        poly = self.a_poly if mat_str[0] == 'A' else self.b_poly
        x_pow, y_pow = poly[int(mat_str[-1])-1]
        if mat_str[1] == 'T':
            x_pow = - x_pow
            y_pow = - y_pow

        idx = (row_idx + x_pow*self.m)%(self.l*self.m)
        idx = (idx//self.m)*self.m + (idx + y_pow)%self.m
        return idx



def cheeger_constant(G):
    """
    작은 그래프용 정확한 Cheeger constant 계산
    h(G) = min |cut(S, V-S)| / |S|
    """
    n = len(G.nodes)
    if n < 2: return 0

    nodes = list(G.nodes)
    min_h = float('inf')

    # 노드 수의 절반까지 모든 조합 탐색
    for r in range(1, n // 2 + 1):
        for S in combinations(nodes, r):
            # nx.cut_size는 S와 V-S 사이의 간선 수를 반환 (|dS|)
            cut_val = nx.cut_size(G, S)
            h_S = cut_val / r
            if h_S < min_h:
                min_h = h_S

    return min_h

class MergedCode(CSS):
    def __init__(self, C1: BBCode, C2: BBCode):
        self.C1 = C1
        self.C2 = C2
        self.F = galois.GF(2) # Field
        self.n = self.C1.n + self.C2.n
        # Hx = [ H1_x  0   ]
        #      [ 0     H2_x]
        Hx_top = np.hstack((self.C1.Hx, np.zeros((self.C1.Hx.shape[0], self.C2.n), dtype=int)))
        Hx_bottom = np.hstack((np.zeros((self.C2.Hx.shape[0], self.C1.n), dtype=int), self.C2.Hx))
        self.Hx = self.F(np.vstack((Hx_top, Hx_bottom)))
        Hz_top = np.hstack((self.C1.Hz, np.zeros((self.C1.Hz.shape[0], self.C2.n), dtype=int)))
        Hz_bottom = np.hstack((np.zeros((self.C2.Hz.shape[0], self.C1.n), dtype=int), self.C2.Hz))
        self.Hz = self.F(np.vstack((Hz_top, Hz_bottom)))
        # CSS class 상속
        super().__init__(self.Hx, self.Hz)
    def add_edge_qubits(self, num_edge: int):
        # edge qubit들은 |+> 상태로 초기화

        self.Hx = np.hstack((self.Hx, np.zeros((self.Hx.shape[0], num_edge), dtype=int)))
        Hedge = np.zeros((num_edge, self.n + num_edge), dtype=int)
        for i in range(num_edge):
            Hedge[i, self.n+i] = 1
        self.Hx = np.vstack((self.Hx, Hedge))
        self.Hz = np.hstack((self.Hz, np.zeros((self.Hz.shape[0], num_edge), dtype=int)))
        self.n += num_edge
        self.Gx = self.Hx
        self.Gz = self.Gz
    def measure_operator(self, support, typ):
        
        op = self.F(np.zeros((1,self.n), dtype=int))
        
        for i in support:
            op[0, i] = 1
        print("Measure: ")
        print(op)
        # op와 anti-commute하는 기존 check들 모으자.
        # 그런 check없다면 그냥 추가.
        # 있다면 그 중 하나 없애고, (존재한다면) 다른 anti-commute하는 check들에 곱한다.
        # 근데, check의 weight 작게 표현하고 싶으니까, edge qubit들 위주로 없애자.
        if typ == "X":
            G_opp = self.Hz
            G_self = self.Hx
        elif typ == "Z":
            G_opp = self.Hx
            G_self = self.Hz
        anti_commuting_idx = []
        for i, stab_gen in enumerate(G_opp):
            # if stab_gen anti-commutes with op
            if np.dot(op, stab_gen):
                anti_commuting_idx.append(i)
        if len(anti_commuting_idx) == 0:
            # op가 G_self의 row space에 있다면 놔두고, 없다면 새로운 row 추가
            if np.linalg.matrix_rank(self.F(np.vstack((G_self,op)))) > np.linalg.matrix_rank(self.F(G_self)):
                G_self = self.F(np.vstack((G_self, op)))
        else:
            anti_commuting_idx = anti_commuting_idx[::-1]   # 뒤에서부터 해야 edge qubit부터 없앰
            temp = G_opp[anti_commuting_idx[0]]
            G_opp = self.F(np.delete(G_opp, anti_commuting_idx[0], axis=0))
            for i in range(1, len(anti_commuting_idx)):
                G_opp[anti_commuting_idx[i]] = self.F(np.bitwise_xor(temp, G_opp[anti_commuting_idx[i]]))
            G_self = self.F(np.vstack((G_self, op)))

        if typ == "X":
            self.Hz = G_opp
            self.Hx = G_self
        elif typ == "Z":
            self.Hx = G_opp
            self.Hz = G_self
        print("self.Hx:")
        for s in self.Hx:
            print(s,sum([int(ss) for ss in s]), s[[[21,24,30,33]+[57,60,66,69] + [72,73,74,75,76,77,78,79,80,81]]])
            # s의 1의 개수
        print("self.Hz:")
        for s in self.Hz:
            print(s,sum([int(ss) for ss in s]), s[[[21,24,30,33]+[57,60,66,69] + [72,73,74,75,76,77,78,79,80,81]]])
        self.Gx = self.Hx
        self.Gz = self.Hz

    def merge_with_graph(self, A_indices: list[int], B_indices: list[int]):
        vertices_to_qubit_num = A_indices + [i + self.C1.n for i in B_indices]
        # 일단 새로운 vertice는 없다고 하자.
        G = nx.Graph()
        nodes = [f"A{i}" for i in range(len(A_indices))] + [f"B{i}" for i in range(len(B_indices))]
        G.add_nodes_from(nodes)
        if len(nodes) > 1:
            nx.add_cycle(G, nodes)
        G.add_edge(nodes[0],nodes[4])
        G.add_edge(nodes[2],nodes[6])
        print("cheeger: ",cheeger_constant(G))
        edge_list = list(G.edges)
        print("edge:",edge_list)
        edge_to_idx = {tuple(sorted(e)): i for i, e in enumerate(edge_list)}
        self.add_edge_qubits(len(G.edges))

        # measure A_v
        for node in nodes:
            if node[0] == "A":
                q = A_indices[int(node[1:])]
            elif node[0] == "B":
                q = B_indices[int(node[1:])] + self.C1.n
            # edges containing node
            edges = [self.C1.n+self.C2.n+edge_to_idx[e] for e in G.edges if node in e]
            idxs = [q] + edges
            self.measure_operator(idxs, typ="Z")

        # cycle_basis에 대해 B_c measure.
        cycle_basis = nx.cycle_basis(G)
        for cb in cycle_basis:
            support = []
            for node in cb:
                if node[0] == "A":
                    q = A_indices[int(node[1:])]
                elif node[0] == "B":
                    q = B_indices[int(node[1:])]
                support.append(q)
            self.measure_operator(support, typ="X")

        # 이거 measure할 때 stab gen. weight 비정상적으로 (더 줄일수 있는데 안줄이는) 커지는거 고쳐보자.

        cycles = []
        for i, hx in enumerate(self.Hx):
            if np.count_nonzero(hx[:self.C1.n+self.C2.n]) == 0:
                cycles.append(i)
        for i, hx in enumerate(self.Hx):
            print("Before weight reduction:", i, np.count_nonzero(hx), hx)
            if i in cycles:
                continue
            # cycles의 행들을 xor해서 Hx의 weight 줄이기
            # Hx의 각 행의 [self.C1.n+self.C2.n:] 부분의 weight 줄이자.
            # cycles의 각 index에 해당하는 행들을 포함할지 말지 2^len(cycles) 경우 다 해보기
            # brute-force
            min_weight = np.count_nonzero(hx)
            best_combination = []
            for select in product([0, 1], repeat=len(cycles)):
                test_hx = hx.copy()
                for j, sel in enumerate(select):
                    if sel == 1:
                        test_hx = self.F(np.bitwise_xor(test_hx, self.Hx[cycles[j]]))
                test_weight = np.count_nonzero(test_hx)
                if test_weight < min_weight:
                    min_weight = test_weight
                    best_combination = select
            if best_combination:
                new_hx = hx.copy()
                for j, sel in enumerate(best_combination):
                    if sel == 1:
                        new_hx = self.F(np.bitwise_xor(new_hx, self.Hx[cycles[j]]))
                self.Hx[i] = new_hx
            print("After weight reduction:", i, np.count_nonzero(self.Hx[i]), self.Hx[i])
        self.Gx = self.Hx
        self.Gz = self.Hz   
        
        # Z-check의 weight도 줄여야 함.
            



        # measure 순서를 바꾸던지.. 어쩌던지..
        # 그리고 distance 계산해보자.
        pass
