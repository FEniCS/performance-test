from ufl import (Coefficient, Constant, FunctionSpace, Mesh, Measure,
                 TestFunction, TrialFunction, dx, grad, inner)
import basix
from basix.ufl import blocked_element, wrap_element

q_map_gll = {1: 1, 2: 3, 3: 4, 4: 6, 5: 8, 6: 10, 7: 12, 8: 14}
q_map_gl = {1: 2, 2: 4, 3: 6, 4: 8, 5: 10, 6: 12, 7: 14, 8: 16}

family = basix.ElementFamily.P
cell_type = basix.CellType.hexahedron
variant = basix.LagrangeVariant.gll_warped

ns = vars()
forms = []
for degree in range(1, 8):
    e = wrap_element(basix.create_tp_element(family, cell_type, degree, variant))

    coord_ele = basix.create_tp_element(family, cell_type, 1, variant)
    coord_element = blocked_element(wrap_element(coord_ele), (3,))
    mesh = Mesh(coord_element)

    V = FunctionSpace(mesh, e)

    u = TrialFunction(V)
    v = TestFunction(V)
    w0 = Coefficient(V)
    c0 = Constant(mesh)

    quad_rules = [["GL", degree],
                  ["GLL", degree],
                  ["GL", degree + 1],
                  ["GLL", degree + 1]]
    for (quad_type, quad_degree) in quad_rules:
        if quad_type == "GL":
            q_map = q_map_gl
        else:
            assert(quad_type == "GLL")
            q_map = q_map_gll

        dx = Measure("dx", metadata={"quadrature_rule": quad_type, "quadrature_degree": q_map[quad_degree]})

        a_name = f"a_{degree}_{quad_degree + 1}_{quad_type}"
        L_name = f"L_{degree}_{quad_degree + 1}_{quad_type}"
        ns[a_name] = c0 * inner(grad(u), grad(v)) * dx
        ns[L_name] = inner(w0, v) * dx

        forms += [ns[a_name], ns[L_name]]
