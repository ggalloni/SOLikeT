import copy
import importlib
import os

import numpy as np
import pytest
from cobaya.model import get_model

from soliket.ccl import CCL

gammakappa_sacc_file = "tests/data/des_s-act_kappa.toy-sim.sacc.fits"
gkappa_sacc_file = "tests/data/gc_cmass-actdr4_kappa.sacc.fits"

cross_correlation_params = {
    "b1": 1.0,
    "s1": 0.4,
}
cross_correlation_theory = {
    "camb": None,
    "ccl": {"external": CCL, "nonlinear": False},
}


def test_galaxykappa_import():
    _ = importlib.import_module("soliket.cross_correlation").GalaxyKappaLikelihood


def test_shearkappa_import():
    _ = importlib.import_module("soliket.cross_correlation").ShearKappaLikelihood


def test_galaxykappa_with_wrong_types(request):
    from soliket.cross_correlation import GalaxyKappaLikelihood

    base_case = {
        "datapath": "valid_path",
        "use_spectra": ["valid"],
        "ncovsims": 5,
        "params": {},
    }
    wrong_type_cases = {
        "datapath": 12345,
        "use_spectra": 12345,
        "ncovsims": "not_an_int",
        "params": "not_a_dict",
    }

    for key, wrong_value in wrong_type_cases.items():
        case = copy.deepcopy(base_case)
        case[key] = wrong_value
        with pytest.raises(TypeError):
            _ = GalaxyKappaLikelihood(**case)


def test_shearkappa_with_wrong_types(request):
    from soliket.cross_correlation import ShearKappaLikelihood

    base_case = {
        "datapath": "valid_path",
        "use_spectra": ["valid"],
        "ncovsims": 5,
        "params": {},
        "z_nuisance_mode": "valid_str",
        "m_nuisance_mode": True,
        "ia_mode": "valid_str",
    }
    wrong_type_cases = {
        "datapath": 12345,
        "use_spectra": 12345,
        "ncovsims": "not_an_int",
        "params": "not_a_dict",
        "z_nuisance_mode": 12345,
        "m_nuisance_mode": "not_a_bool",
        "ia_mode": 12345,
    }

    for key, wrong_value in wrong_type_cases.items():
        case = copy.deepcopy(base_case)
        case[key] = wrong_value
        with pytest.raises(TypeError):
            _ = ShearKappaLikelihood(**case)


def test_galaxykappa_model(
    request, check_skip_pyccl, evaluate_one_info, test_cosmology_params
):
    from soliket.cross_correlation import GalaxyKappaLikelihood

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["params"].update(cross_correlation_params)
    evaluate_one_info["theory"] = cross_correlation_theory

    evaluate_one_info["likelihood"] = {
        "GalaxyKappaLikelihood": {
            "external": GalaxyKappaLikelihood,
            "datapath": os.path.join(request.config.rootdir, gkappa_sacc_file),
        }
    }

    _ = get_model(evaluate_one_info)


def test_shearkappa_model(
    request, check_skip_pyccl, evaluate_one_info, test_cosmology_params
):
    from soliket.cross_correlation import ShearKappaLikelihood

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = cross_correlation_theory

    evaluate_one_info["likelihood"] = {
        "ShearKappaLikelihood": {
            "external": ShearKappaLikelihood,
            "datapath": os.path.join(request.config.rootdir, gammakappa_sacc_file),
        }
    }

    _ = get_model(evaluate_one_info)


def test_galaxykappa_like(
    request, check_skip_pyccl, evaluate_one_info, test_cosmology_params, likelihood_refs
):
    from soliket.cross_correlation import GalaxyKappaLikelihood

    ref = likelihood_refs["galaxykappa"]

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["params"].update(cross_correlation_params)
    evaluate_one_info["theory"] = cross_correlation_theory

    evaluate_one_info["likelihood"] = {
        "GalaxyKappaLikelihood": {
            "external": GalaxyKappaLikelihood,
            "datapath": os.path.join(request.config.rootdir, gkappa_sacc_file),
            "use_spectra": [("gc_cmass", "ck_actdr4")],
        }
    }

    model = get_model(evaluate_one_info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes[0], ref["value"], rtol=ref["rtol"], atol=ref["atol"])


def test_shearkappa_like(request, check_skip_pyccl, evaluate_one_info, likelihood_refs):
    from soliket.cross_correlation import ShearKappaLikelihood

    ref = likelihood_refs["shearkappa"]

    evaluate_one_info["theory"] = cross_correlation_theory

    rootdir = request.config.rootdir

    cs82_file = "tests/data/cs82_gs-planck_kappa_binned.sim.sacc.fits"
    test_datapath = os.path.join(rootdir, cs82_file)

    evaluate_one_info["likelihood"] = {
        "ShearKappaLikelihood": {
            "external": ShearKappaLikelihood,
            "datapath": test_datapath,
        }
    }

    # Cosmological parameters for the test data, digitized from
    # Fig. 3 and Eq. 8 of Hall & Taylor (2014).
    # See https://github.com/simonsobs/SOLikeT/pull/58 for validation plots
    evaluate_one_info["params"] = {
        "omch2": 0.118,  # Planck + lensing + WP + highL
        "ombh2": 0.0222,
        "H0": 68.0,
        "ns": 0.962,
        "As": 2.1e-9,
        "tau": 0.094,
        "mnu": 0.0,
        "nnu": 3.046,
    }

    model = get_model(evaluate_one_info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes, ref["value"], rtol=ref["rtol"], atol=ref["atol"])



def test_shearkappa_tracerselect(
    request, check_skip_pyccl, evaluate_one_info, test_cosmology_params
):
    import copy

    from soliket.cross_correlation import ShearKappaLikelihood

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = cross_correlation_theory

    rootdir = request.config.rootdir

    test_datapath = os.path.join(rootdir, gammakappa_sacc_file)

    evaluate_one_info["likelihood"] = {
        "ShearKappaLikelihood": {
            "external": ShearKappaLikelihood,
            "datapath": test_datapath,
            "use_spectra": "all",
        }
    }

    info_onebin = copy.deepcopy(evaluate_one_info)
    info_onebin["likelihood"]["ShearKappaLikelihood"]["use_spectra"] = [
        ("gs_des_bin1", "ck_act")
    ]

    info_twobin = copy.deepcopy(evaluate_one_info)
    info_twobin["likelihood"]["ShearKappaLikelihood"]["use_spectra"] = [
        ("gs_des_bin1", "ck_act"),
        ("gs_des_bin3", "ck_act"),
    ]

    model = get_model(evaluate_one_info)
    loglikes, derived = model.loglikes()

    lhood = model.likelihood["ShearKappaLikelihood"]

    model_onebin = get_model(info_onebin)
    loglikes_onebin, derived_onebin = model_onebin.loglikes()

    lhood_onebin = model_onebin.likelihood["ShearKappaLikelihood"]

    model_twobin = get_model(info_twobin)
    loglikes_twobin, derived_twobin = model_twobin.loglikes()

    lhood_twobin = model_twobin.likelihood["ShearKappaLikelihood"]

    n_ell_perbin = len(lhood.data.x) // 4

    assert n_ell_perbin == len(lhood_onebin.data.x)
    assert np.allclose(lhood.data.y[:n_ell_perbin], lhood_onebin.data.y)

    assert 2 * n_ell_perbin == len(lhood_twobin.data.x)
    assert np.allclose(
        np.concatenate(
            [
                lhood.data.y[:n_ell_perbin],
                lhood.data.y[2 * n_ell_perbin : 3 * n_ell_perbin],
            ]
        ),
        lhood_twobin.data.y,
    )


def test_shearkappa_hartlap(request, check_skip_pyccl, evaluate_one_info):
    from soliket.cross_correlation import ShearKappaLikelihood

    evaluate_one_info["theory"] = cross_correlation_theory

    rootdir = request.config.rootdir

    cs82_file = "tests/data/cs82_gs-planck_kappa_binned.sim.sacc.fits"
    test_datapath = os.path.join(rootdir, cs82_file)

    evaluate_one_info["likelihood"] = {
        "ShearKappaLikelihood": {
            "external": ShearKappaLikelihood,
            "datapath": test_datapath,
        }
    }

    # Cosmological parameters for the test data, digitized from
    # Fig. 3 and Eq. 8 of Hall & Taylor (2014).
    # See https://github.com/simonsobs/SOLikeT/pull/58 for validation plots
    evaluate_one_info["params"] = {
        "omch2": 0.118,  # Planck + lensing + WP + highL
        "ombh2": 0.0222,
        "H0": 68.0,
        "ns": 0.962,
        # "As": 2.1e-9,
        "As": 2.5e-9,  # offset the theory to upweight inv_cov in loglike
        "tau": 0.094,
        "mnu": 0.0,
        "nnu": 3.046,
    }

    model = get_model(evaluate_one_info)
    loglikes, derived = model.loglikes()

    evaluate_one_info["likelihood"]["ShearKappaLikelihood"]["ncovsims"] = 5

    model = get_model(evaluate_one_info)
    loglikes_hartlap, derived = model.loglikes()

    assert np.isclose(
        np.abs(loglikes - loglikes_hartlap), 0.0010403, rtol=1.0e-5, atol=1.0e-5
    )


def test_shearkappa_deltaz(
    request, check_skip_pyccl, evaluate_one_info, test_cosmology_params, likelihood_refs
):
    from soliket.cross_correlation import ShearKappaLikelihood

    ref = likelihood_refs["shearkappa_deltaz"]


    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = cross_correlation_theory

    evaluate_one_info["likelihood"] = {
        "ShearKappaLikelihood": {
            "external": ShearKappaLikelihood,
            "datapath": os.path.join(request.config.rootdir, gammakappa_sacc_file),
            "z_nuisance_mode": "deltaz",
        }
    }

    model = get_model(evaluate_one_info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes[0], ref["value"], rtol=ref["rtol"], atol=ref["atol"])


def test_shearkappa_m(
    request, check_skip_pyccl, evaluate_one_info, test_cosmology_params, likelihood_refs
):
    from soliket.cross_correlation import ShearKappaLikelihood

    ref = likelihood_refs["shearkappa_m"]

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = cross_correlation_theory

    evaluate_one_info["likelihood"] = {
        "ShearKappaLikelihood": {
            "external": ShearKappaLikelihood,
            "datapath": os.path.join(request.config.rootdir, gammakappa_sacc_file),
            "m_nuisance_mode": True,
        }
    }

    model = get_model(evaluate_one_info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes[0], ref["value"], rtol=ref["rtol"], atol=ref["atol"])


def test_shearkappa_ia_nla_noevo(
    request, check_skip_pyccl, evaluate_one_info, test_cosmology_params, likelihood_refs
):
    from soliket.cross_correlation import ShearKappaLikelihood

    ref = likelihood_refs["shearkappa_ia_nla_noevo"]

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = cross_correlation_theory

    evaluate_one_info["likelihood"] = {
        "ShearKappaLikelihood": {
            "external": ShearKappaLikelihood,
            "datapath": os.path.join(request.config.rootdir, gammakappa_sacc_file),
            "ia_mode": "nla-noevo",
        }
    }

    model = get_model(evaluate_one_info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes[0], ref["value"], rtol=ref["rtol"], atol=ref["atol"])


def test_shearkappa_ia_nla(
    request, check_skip_pyccl, evaluate_one_info, test_cosmology_params, likelihood_refs
):
    from soliket.cross_correlation import ShearKappaLikelihood

    ref = likelihood_refs["shearkappa_ia_nla"]

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = cross_correlation_theory

    evaluate_one_info["likelihood"] = {
        "ShearKappaLikelihood": {
            "external": ShearKappaLikelihood,
            "datapath": os.path.join(request.config.rootdir, gammakappa_sacc_file),
            "ia_mode": "nla",
        }
    }

    evaluate_one_info["params"]["eta_IA"] = 1.7

    model = get_model(evaluate_one_info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes[0], ref["value"], rtol=ref["rtol"], atol=ref["atol"])


def test_shearkappa_ia_perbin(
    request, check_skip_pyccl, evaluate_one_info, test_cosmology_params, likelihood_refs
):
    from soliket.cross_correlation import ShearKappaLikelihood

    ref = likelihood_refs["shearkappa_ia_perbin"]

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = cross_correlation_theory

    evaluate_one_info["likelihood"] = {
        "ShearKappaLikelihood": {
            "external": ShearKappaLikelihood,
            "datapath": os.path.join(request.config.rootdir, gammakappa_sacc_file),
            "ia_mode": "nla-perbin",
        }
    }

    model = get_model(evaluate_one_info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes[0], ref["value"], rtol=ref["rtol"], atol=ref["atol"])


def test_shearkappa_hmcode(
    request, check_skip_pyccl, evaluate_one_info, test_cosmology_params, likelihood_refs
):
    from soliket.cross_correlation import ShearKappaLikelihood

    ref = likelihood_refs["shearkappa_hmcode"]


    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = cross_correlation_theory

    evaluate_one_info["likelihood"] = {
        "ShearKappaLikelihood": {
            "external": ShearKappaLikelihood,
            "datapath": os.path.join(request.config.rootdir, gammakappa_sacc_file),
        }
    }
    evaluate_one_info["theory"] = {
        "camb": {
            "extra_args": {"halofit_version": "mead2020_feedback", "HMCode_logT_AGN": 7.8}
        },
        "ccl": {"external": CCL, "nonlinear": False},
    }

    model = get_model(evaluate_one_info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes[0], ref["value"], rtol=ref["rtol"], atol=ref["atol"])
