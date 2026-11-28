import numpy as np
import galois
import re
from docplex.mp.model import Model
from sympy import Poly

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
        self.dx_i = [None] * self.k    # logical qubit 별 보호수준 (protection level)
        self.dz_i = [None] * self.k


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
        # basis of G_self
        basis_self = G_self.row_space()
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
        for i, dual_op in enumerate(dual_logic_ops):
            if dual_op is not None:
                A = np.vstack((A, dual_op))
                if i == idx:
                    b = np.hstack((b, self.F(1)))
                else:
                    b = np.hstack((b, self.F(0)))
        
        # Ax=b 의 nullspace의 basis중에 G_self의 row space에 없는 것 찾기 (반드시 존재함)
        x_part, x_null = solve_gf2(A, b)
        # x_part는 row vector
        # x_null의 행이 해
        for row in x_null:
            x = x_part + row
            # G_self의 row space에 없는지 확인
            if np.linalg.matrix_rank(self.F(np.vstack((G_self, x)))) == np.linalg.matrix_rank(self.F(G_self)) + 1:
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
        if sol:
            result = np.array([int(sol.get_value(y[j])) for j in range(n)])
            return galois.GF(2)(result)
        else:
            return None



    def _find_min_op_weight(self, l_type, target=None):
        # target=None: l_type의 모든 logical operator 중 (up to multiplication with checks) 최소 weight
        # target=idx: logical qubit idx 에 nontrivial하게 작용하는 (다른 logical qubit을
        #             같이 건드려도 되는) 연산자 중 최소 weight -> 그 logical qubit의 보호수준
        if l_type == "X":
            G_opp = self.Gz
            G_self = self.Gx
            anti_logic_ops = self.logical_z_repr    # 적어도 하나의 Z-logic op과 anti-commute해야 non-trivial logical op.임.
        elif l_type == "Z":
            G_opp = self.Gx
            G_self = self.Gz
            anti_logic_ops = self.logical_x_repr

        assert anti_logic_ops is not None and all(v is not None for v in anti_logic_ops), (
            f"logical_{'z' if l_type=='X' else 'x'}_repr 에 None 이 있음. "
            "set_missing_logicals() 를 먼저 호출할 것.")

        model = Model(name="css_distance")
        e = model.binary_var_list(self.n, name='e')
        s = model.binary_var_list(self.k, name='s') # s_i: Z_i와 anti-commute하는지

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
        if target is None:
            model.add_constraint(model.sum(s)>=1)
        else:
            assert 0 <= target < self.k, f"target={target} 는 [0,{self.k}) 범위여야 함"
            model.add_constraint(s[target] == 1)
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
        self.set_missing_logicals()
        self.calculate_k()
        self.dx = self._find_min_op_weight("X")
        self.dz = self._find_min_op_weight("Z")
        self.d = min(self.dx,self.dz)
        return self.d

    def get_distance(self,):
        if self.dx is None or self.dz is None:
            self.calculate_distance()
        return self.dx, self.dz

    def logical_qubit_distance(self, idx: int):
        """logical qubit idx 의 (dx_i, dz_i) 보호수준: idx 에 nontrivial하게 작용하는
        (다른 logical qubit도 같이 건드려도 되는) 연산자 중 최소 weight."""
        wx = self._find_min_op_weight("X", target=idx)
        wz = self._find_min_op_weight("Z", target=idx)
        self.dx_i[idx], self.dz_i[idx] = wx, wz
        return wx, wz

    def logical_qubit_distances(self):
        """모든 logical qubit 의 (dx_i, dz_i). ILP 2k 번. min 이 코드 전체 distance."""
        out = [self.logical_qubit_distance(i) for i in range(self.k)]
        if self.k and all(w is not None for p in out for w in p):
            self.dx = min(p[0] for p in out)
            self.dz = min(p[1] for p in out)
            self.d = min(self.dx, self.dz)
        return out

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
    r"""
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
        self.d = d
        self.dx = self.dz = d   # rotated surface code는 dx = dz = d


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
