""" Tests for calculations."""
import numpy as np

from aiida import orm

from aiida_yambo_wannier90.calculations.functions.kmesh import (
    find_commensurate_integers,
    find_commensurate_meshes,
    is_commensurate,
    kmapper,
)

# from . import TEST_DIR


def test_kmapper():
    """Test ``kmapper``."""

    kpoints = np.array(
        [
            [0.0000000, 0.0000000, 0.0000000, 0.0039062],
            [0.0000000, 0.0000000, 0.1250000, 0.0312500],
            [0.0000000, 0.0000000, 0.2500000, 0.0312500],
            [0.0000000, 0.0000000, 0.3750000, 0.0312500],
            [0.0000000, 0.0000000, -0.5000000, 0.0156250],
            [0.0000000, 0.1250000, 0.1250000, 0.0234375],
            [0.0000000, 0.1250000, 0.2500000, 0.0937500],
            [0.0000000, 0.1250000, 0.3750000, 0.0937500],
            [0.0000000, 0.1250000, -0.5000000, 0.0937500],
            [0.0000000, 0.1250000, -0.3750000, 0.0937500],
            [0.0000000, 0.1250000, -0.2500000, 0.0937500],
            [0.0000000, 0.1250000, -0.1250000, 0.0468750],
            [0.0000000, 0.2500000, 0.2500000, 0.0234375],
            [0.0000000, 0.2500000, 0.3750000, 0.0937500],
            [0.0000000, 0.2500000, -0.5000000, 0.0937500],
            [0.0000000, 0.2500000, -0.3750000, 0.0937500],
            [0.0000000, 0.2500000, -0.2500000, 0.0468750],
            [0.0000000, 0.3750000, 0.3750000, 0.0234375],
            [0.0000000, 0.3750000, -0.5000000, 0.0937500],
            [0.0000000, 0.3750000, -0.3750000, 0.0468750],
            [0.0000000, -0.5000000, -0.5000000, 0.0117188],
            [0.1250000, 0.2500000, 0.3750000, 0.0937500],
            [0.1250000, 0.2500000, -0.5000000, 0.1875000],
            [0.1250000, 0.2500000, -0.3750000, 0.0937500],
            [0.1250000, 0.3750000, -0.5000000, 0.0937500],
            [0.1250000, 0.3750000, -0.3750000, 0.1875000],
            [0.1250000, 0.3750000, -0.2500000, 0.0937500],
            [0.1250000, -0.5000000, -0.3750000, 0.0468750],
            [0.2500000, -0.5000000, -0.2500000, 0.0234375],
        ]
    )

    dense_mesh = orm.KpointsData()
    dense_mesh.set_kpoints(
        kpoints=kpoints[:, :3],
        cartesian=False,
        weights=kpoints[:, 3],
    )

    kpoints = np.array(
        [
            [0.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.25000000],
            [0.00000000, 0.00000000, 0.50000000],
            [0.00000000, 0.00000000, 0.75000000],
            [0.00000000, 0.25000000, 0.00000000],
            [0.00000000, 0.25000000, 0.25000000],
            [0.00000000, 0.25000000, 0.50000000],
            [0.00000000, 0.25000000, 0.75000000],
            [0.00000000, 0.50000000, 0.00000000],
            [0.00000000, 0.50000000, 0.25000000],
            [0.00000000, 0.50000000, 0.50000000],
            [0.00000000, 0.50000000, 0.75000000],
            [0.00000000, 0.75000000, 0.00000000],
            [0.00000000, 0.75000000, 0.25000000],
            [0.00000000, 0.75000000, 0.50000000],
            [0.00000000, 0.75000000, 0.75000000],
            [0.25000000, 0.00000000, 0.00000000],
            [0.25000000, 0.00000000, 0.25000000],
            [0.25000000, 0.00000000, 0.50000000],
            [0.25000000, 0.00000000, 0.75000000],
            [0.25000000, 0.25000000, 0.00000000],
            [0.25000000, 0.25000000, 0.25000000],
            [0.25000000, 0.25000000, 0.50000000],
            [0.25000000, 0.25000000, 0.75000000],
            [0.25000000, 0.50000000, 0.00000000],
            [0.25000000, 0.50000000, 0.25000000],
            [0.25000000, 0.50000000, 0.50000000],
            [0.25000000, 0.50000000, 0.75000000],
            [0.25000000, 0.75000000, 0.00000000],
            [0.25000000, 0.75000000, 0.25000000],
            [0.25000000, 0.75000000, 0.50000000],
            [0.25000000, 0.75000000, 0.75000000],
            [0.50000000, 0.00000000, 0.00000000],
            [0.50000000, 0.00000000, 0.25000000],
            [0.50000000, 0.00000000, 0.50000000],
            [0.50000000, 0.00000000, 0.75000000],
            [0.50000000, 0.25000000, 0.00000000],
            [0.50000000, 0.25000000, 0.25000000],
            [0.50000000, 0.25000000, 0.50000000],
            [0.50000000, 0.25000000, 0.75000000],
            [0.50000000, 0.50000000, 0.00000000],
            [0.50000000, 0.50000000, 0.25000000],
            [0.50000000, 0.50000000, 0.50000000],
            [0.50000000, 0.50000000, 0.75000000],
            [0.50000000, 0.75000000, 0.00000000],
            [0.50000000, 0.75000000, 0.25000000],
            [0.50000000, 0.75000000, 0.50000000],
            [0.50000000, 0.75000000, 0.75000000],
            [0.75000000, 0.00000000, 0.00000000],
            [0.75000000, 0.00000000, 0.25000000],
            [0.75000000, 0.00000000, 0.50000000],
            [0.75000000, 0.00000000, 0.75000000],
            [0.75000000, 0.25000000, 0.00000000],
            [0.75000000, 0.25000000, 0.25000000],
            [0.75000000, 0.25000000, 0.50000000],
            [0.75000000, 0.25000000, 0.75000000],
            [0.75000000, 0.50000000, 0.00000000],
            [0.75000000, 0.50000000, 0.25000000],
            [0.75000000, 0.50000000, 0.50000000],
            [0.75000000, 0.50000000, 0.75000000],
            [0.75000000, 0.75000000, 0.00000000],
            [0.75000000, 0.75000000, 0.25000000],
            [0.75000000, 0.75000000, 0.50000000],
            [0.75000000, 0.75000000, 0.75000000],
        ]
    )
    coarse_mesh = orm.KpointsData()
    coarse_mesh.set_kpoints(
        kpoints=kpoints,
        cartesian=False,
        weights=[1 / len(kpoints)] * len(kpoints),
    )

    result = kmapper(dense_mesh, coarse_mesh, orm.Int(1), orm.Int(14))

    assert isinstance(result, orm.List)

    result = result.get_list()

    solution = [
        [1, 1, 1, 14],
        [3, 3, 1, 14],
        [5, 5, 1, 14],
        [13, 13, 1, 14],
        [15, 15, 1, 14],
        [17, 17, 1, 14],
        [21, 21, 1, 14],
        [29, 29, 1, 14],
    ]
    assert result == solution


def test_find_commensurate_integers():
    """Test ``find_commensurate_integers``."""

    result = find_commensurate_integers(5, 2)
    assert result == (6, 2)

    result = find_commensurate_integers(11, 5)
    assert result == (12, 6)

    result = find_commensurate_integers(3, 5)
    assert result == (5, 5)

    result = find_commensurate_integers(5, 5, include_identical=False)
    assert result == (10, 5)


def test_find_commensurate_meshes():
    """Test ``find_commensurate_meshes``."""

    coarse_mesh = orm.KpointsData()
    coarse_mesh.set_kpoints_mesh([2, 5, 5])

    dense_mesh = orm.KpointsData()
    dense_mesh.set_kpoints_mesh([5, 11, 3])

    results = find_commensurate_meshes(dense_mesh, coarse_mesh)
    new_dense_mesh = results["dense_mesh"]
    new_coarse_mesh = results["coarse_mesh"]

    assert isinstance(new_dense_mesh, orm.KpointsData)
    assert isinstance(new_coarse_mesh, orm.KpointsData)

    new_dense_mesh, _ = new_dense_mesh.get_kpoints_mesh()
    new_coarse_mesh, _ = new_coarse_mesh.get_kpoints_mesh()

    assert new_dense_mesh == [6, 12, 5]
    assert new_coarse_mesh == [2, 6, 5]


def test_is_commensurate():
    """Test ``is_commensurate``."""

    dense = [5, 11, 3]
    coarse = [2, 5, 5]
    result = is_commensurate(dense, coarse)
    assert result is False

    coarse_mesh = orm.KpointsData()
    coarse_mesh.set_kpoints_mesh(coarse)
    dense_mesh = orm.KpointsData()
    dense_mesh.set_kpoints_mesh(dense)
    result = is_commensurate(dense_mesh, coarse_mesh)
    assert result is False

    dense = [5, 5, 4]
    coarse = [5, 5, 4]
    result = is_commensurate(dense, coarse)
    assert result is True

    coarse_mesh = orm.KpointsData()
    coarse_mesh.set_kpoints_mesh(coarse)
    dense_mesh = orm.KpointsData()
    dense_mesh.set_kpoints_mesh(dense)
    result = is_commensurate(dense_mesh, coarse_mesh)
    assert result is True

    dense = [10, 15, 6]
    coarse = [5, 5, 2]
    result = is_commensurate(dense, coarse)
    assert result is True

    coarse_mesh = orm.KpointsData()
    coarse_mesh.set_kpoints_mesh(coarse)
    dense_mesh = orm.KpointsData()
    dense_mesh.set_kpoints_mesh(dense)
    result = is_commensurate(dense_mesh, coarse_mesh)
    assert result is True
