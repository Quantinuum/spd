def test_sparse_pauli_op_string_rendering(backend):
    _, module = backend
    spo = module.create_op({
        "XIZY": 1.5,
        "ZZ": -0.25j,
        "IIII": 2.0,
    })

    rendered = str(spo)

    assert "SparsePauliOp[" in rendered
    assert "XIZY" in rendered
    assert "ZZ" in rendered
    assert "IIII" in rendered
    assert "=> 1.5" in rendered
    assert "=> (-0-0.25j)" in rendered or "=> -0.25j" in rendered
    assert "=> 2.0" in rendered


def test_sparse_pauli_op_weight_distribution(backend):
    _, module = backend
    spo = module.create_op({
        "IIII": 1.0,
        "XIII": 2.0,
        "YZZI": 3.0,
        "XYZX": 4.0,
    })

    distribution = spo.get_Pauli_weight_distribution()

    assert distribution == {
        0: 1,
        1: 1,
        3: 1,
        4: 1,
    }
