# project by Sina Hajiaghaei

import numpy as np


class Circuit:
    def __init__(self, nodes, ground="0"):
        self.nodes = list(nodes)
        self.ground = str(ground)
        self.resistors = []
        self.current_sources = []
        if self.ground not in self.nodes:
            self.nodes.append(self.ground)

    def add_resistor(self, n1, n2, R):
        self.resistors.append((str(n1), str(n2), float(R)))

    def add_current_source(self, n, I):
        self.current_sources.append((str(n), float(I)))


def node_map(circuit: Circuit):
    unknown_nodes = [n for n in circuit.nodes if n != circuit.ground]
    idx = {node: i for i, node in enumerate(unknown_nodes)}
    return unknown_nodes, idx


def build_y_matrix(circuit: Circuit):
    unknown_nodes, idx = node_map(circuit)
    n = len(unknown_nodes)

    Y = np.zeros((n, n), dtype=float)
    I = np.zeros(n, dtype=float)

    for (a, b, R) in circuit.resistors:
        if R == 0:
            raise ValueError("Resistor value cannot be zero.")
        g = 1.0 / R

        if a != circuit.ground and b != circuit.ground:
            i = idx[a]
            j = idx[b]
            Y[i, i] += g
            Y[j, j] += g
            Y[i, j] -= g
            Y[j, i] -= g
        elif a != circuit.ground and b == circuit.ground:
            i = idx[a]
            Y[i, i] += g
        elif b != circuit.ground and a == circuit.ground:
            j = idx[b]
            Y[j, j] += g

    for (node, current) in circuit.current_sources:
        if node == circuit.ground:
            continue
        I[idx[node]] += current

    return Y, I


def solve_circuit(circuit: Circuit):
    Y, I = build_y_matrix(circuit)
    unknown_nodes, idx = node_map(circuit)

    if len(unknown_nodes) == 0:
        return {circuit.ground: 0.0}, []

    V_unknown = np.linalg.solve(Y, I)

    V = {circuit.ground: 0.0}
    for node in unknown_nodes:
        V[node] = V_unknown[idx[node]]

    branch_currents = []
    for (a, b, R) in circuit.resistors:
        Va = V[a] if a in V else 0.0
        Vb = V[b] if b in V else 0.0
        Ir = (Va - Vb) / R
        branch_currents.append({"n1": a, "n2": b, "R": R, "I": Ir})

    return V, branch_currents


def validate_kcl(circuit: Circuit, V: dict, branch_currents: list, tol=1e-6):
    unknown_nodes, _ = node_map(circuit)

    Iinj = {n: 0.0 for n in circuit.nodes}
    for (n, I) in circuit.current_sources:
        Iinj[n] += I

    leaving = {n: 0.0 for n in circuit.nodes}
    for br in branch_currents:
        a, b, Ir = br["n1"], br["n2"], br["I"]
        leaving[a] += Ir
        leaving[b] -= Ir

    results = {}
    ok = True
    for node in unknown_nodes:
        err = leaving[node] - Iinj.get(node, 0.0)
        results[node] = err
        if abs(err) > tol:
            ok = False

    return ok, results


def print_results(title, V, branch_currents, kcl_ok, kcl_errors):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print("\nNode Voltages (V):")
    for node in sorted(V.keys(), key=lambda x: int(x) if x.isdigit() else x):
        print(f"  V({node}) = {V[node]:.6f} V")

    print("\nBranch Currents (A) (direction n1 -> n2):")
    for i, br in enumerate(branch_currents, start=1):
        print(f"  R{i}: {br['n1']} -> {br['n2']}  (R={br['R']:.3f} Ω)  I = {br['I']:.6f} A")

    print("\nKCL Check:")
    print("  PASS ✅" if kcl_ok else "  FAIL ❌")
    print("  Node errors (leaving - injected):")
    for node, err in kcl_errors.items():
        print(f"    Node {node}: error = {err:.8f}")


def test_circuits():
    c1 = Circuit(nodes=["0", "1", "2"], ground="0")
    c1.add_resistor("1", "0", 1000)
    c1.add_resistor("2", "0", 2000)
    c1.add_resistor("1", "2", 3000)
    c1.add_current_source("1", 0.005)

    V1, I1 = solve_circuit(c1)
    ok1, err1 = validate_kcl(c1, V1, I1)
    print_results("TEST 1", V1, I1, ok1, err1)

    c2 = Circuit(nodes=["0", "1", "2"], ground="0")
    c2.add_resistor("1", "2", 1000)
    c2.add_resistor("2", "0", 1000)
    c2.add_current_source("1", 0.01)

    V2, I2 = solve_circuit(c2)
    ok2, err2 = validate_kcl(c2, V2, I2)
    print_results("TEST 2", V2, I2, ok2, err2)

    c3 = Circuit(nodes=["0", "1", "2", "3"], ground="0")
    c3.add_resistor("1", "0", 1000)
    c3.add_resistor("2", "1", 2000)
    c3.add_resistor("3", "2", 3000)

    V3, I3 = solve_circuit(c3)
    ok3, err3 = validate_kcl(c3, V3, I3)
    print_results("TEST 3", V3, I3, ok3, err3)


if __name__ == "__main__":
    test_circuits()