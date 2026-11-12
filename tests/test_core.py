"""Fast sanity tests for aml_core. Run with:  python -m pytest -q"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aml_core as core  # noqa: E402


@pytest.fixture(scope="module")
def tabular():
    data = core.make_fraud_dataset(seed=1)
    models = core.train_tabular_models(data.X_train, data.y_train, seed=1)
    return data, models, core.build_defense_kit(data, seed=1)


def test_dataset_has_learnable_signal(tabular):
    data, models, _ = tabular
    assert abs(data.y_train.mean() - 0.15) < 0.01
    majority = 1 - data.y_test.mean()
    for name in core.CLASSIFIERS:
        acc = (models[name].predict(data.X_test) == data.y_test).mean()
        assert acc > majority + 0.03, f"{name} does no better than always predicting 'legit'"


@pytest.mark.parametrize("attack", list(core.TABULAR_ATTACKS))
def test_tabular_attacks_respect_budget(tabular, attack):
    data, models, _ = tabular
    eps = 0.3
    ctx = core.AttackContext(models, data.X_train, data.lo, data.hi, np.random.default_rng(0))
    X_adv, _ = core.run_tabular_attack(attack, data.X_test, data.y_test, eps, ctx)
    assert np.abs(X_adv - data.X_test).max() <= eps + 1e-9
    assert (X_adv >= data.lo - 1e-9).all() and (X_adv <= data.hi + 1e-9).all()
    if core.TABULAR_ATTACKS[attack].kind == "evasion":
        legit = data.y_test == 0
        assert np.array_equal(X_adv[legit], data.X_test[legit]), "evasion attacks must only touch fraud rows"


def test_evasion_attacks_reduce_recall(tabular):
    data, models, kit = tabular
    summary = core.tabular_summary(data, models, kit, eps=0.5, defense="None", seed=0)
    evasion = summary[summary["Metric"] == "Fraud recall"]
    assert (evasion["Attacked"] <= evasion["Clean"] + 1e-9).all()
    assert (evasion["Attacked"] < evasion["Clean"]).mean() >= 0.8


@pytest.mark.parametrize("defense", core.TABULAR_DEFENSES)
def test_defenses_return_binary_predictions(tabular, defense):
    data, models, kit = tabular
    pred = core.defended_predict(defense, models["Random Forest"], data.X_test, kit)
    assert pred.shape == data.y_test.shape and set(np.unique(pred)) <= {0, 1}


@pytest.mark.parametrize("attack", list(core.VISION_ATTACKS))
def test_vision_attacks_shapes_and_range(attack):
    torch.manual_seed(0)
    model = core.SimpleCNN().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    x = torch.rand(4, 1, 28, 28) * 2 - 1
    y = torch.randint(0, 10, (4,))
    x_adv = core.VISION_ATTACKS[attack](model, x, y, 0.1)
    assert x_adv.shape == x.shape
    assert x_adv.min() >= -1 - 1e-6 and x_adv.max() <= 1 + 1e-6


def test_linf_attacks_stay_in_ball():
    torch.manual_seed(0)
    model = core.SimpleCNN().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    x = torch.rand(4, 1, 28, 28) * 2 - 1
    y = torch.randint(0, 10, (4,))
    for fn in (core.attack_fgsm, core.attack_pgd, core.attack_cw, core.attack_deepfool,
               core.attack_ead, core.attack_feature):
        assert (fn(model, x, y, 0.1) - x).abs().max() <= 0.1 + 1e-5, fn.__name__


def test_bit_depth_levels():
    x = torch.linspace(-1, 1, 1000)
    assert len(torch.unique(core.bit_depth_reduction(x, bits=3))) == 8
