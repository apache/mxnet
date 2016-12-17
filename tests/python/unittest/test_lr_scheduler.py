from mxnet.lr_scheduler import FactorScheduler, MultiFactorScheduler
from check_utils import reldiff

def test_factor_scheduler():
    # normal training
    lr_scheduler = FactorScheduler(30, 0.1)
    lr_scheduler.base_lr = 0.1
    assert reldiff(lr_scheduler(0), 0.1) < 1e-5
    assert reldiff(lr_scheduler(1), 0.1) < 1e-5
    assert reldiff(lr_scheduler(30), 0.1) < 1e-5
    assert reldiff(lr_scheduler(31), 0.01) < 1e-5
    assert reldiff(lr_scheduler(60), 0.01) < 1e-5
    assert reldiff(lr_scheduler(61), 0.001) < 1e-5
    assert reldiff(lr_scheduler(90), 0.001) < 1e-5
    assert reldiff(lr_scheduler(91), 0.0001) < 1e-5
    # continual training (recover from 60)
    lr_scheduler = FactorScheduler(30, 0.1)
    lr_scheduler.base_lr = 0.1
    assert reldiff(lr_scheduler(60), 0.01) < 1e-5
    assert reldiff(lr_scheduler(61), 0.001) < 1e-5
    assert reldiff(lr_scheduler(90), 0.001) < 1e-5
    assert reldiff(lr_scheduler(91), 0.0001) < 1e-5

def test_multi_factor_scheduler():
    # normal training
    lr_scheduler = MultiFactorScheduler([30,60,95], 0.1)
    lr_scheduler.base_lr = 0.1
    assert reldiff(lr_scheduler(0), 0.1) < 1e-5
    assert reldiff(lr_scheduler(1), 0.1) < 1e-5
    assert reldiff(lr_scheduler(30), 0.1) < 1e-5
    assert reldiff(lr_scheduler(31), 0.01) < 1e-5
    assert reldiff(lr_scheduler(60), 0.01) < 1e-5
    assert reldiff(lr_scheduler(61), 0.001) < 1e-5
    assert reldiff(lr_scheduler(90), 0.001) < 1e-5
    assert reldiff(lr_scheduler(91), 0.001) < 1e-5
    assert reldiff(lr_scheduler(95), 0.001) < 1e-5
    assert reldiff(lr_scheduler(96), 0.0001) < 1e-5
    # continual training (recover from 60)
    lr_scheduler = MultiFactorScheduler([30,60,95], 0.1)
    lr_scheduler.base_lr = 0.1
    assert reldiff(lr_scheduler(60), 0.01) < 1e-5
    assert reldiff(lr_scheduler(61), 0.001) < 1e-5
    assert reldiff(lr_scheduler(90), 0.001) < 1e-5
    assert reldiff(lr_scheduler(91), 0.001) < 1e-5
    assert reldiff(lr_scheduler(95), 0.001) < 1e-5
    assert reldiff(lr_scheduler(96), 0.0001) < 1e-5
    # slow step
    lr_scheduler = MultiFactorScheduler([30,60,95], 0.1, slow_step=10)
    lr_scheduler.base_lr = 0.1
    assert reldiff(lr_scheduler(0), 0.01) < 1e-5
    assert reldiff(lr_scheduler(10), 0.01) < 1e-5
    assert reldiff(lr_scheduler(11), 0.1) < 1e-5
    assert reldiff(lr_scheduler(30), 0.1) < 1e-5
    assert reldiff(lr_scheduler(31), 0.01) < 1e-5
    # slow step, continual training
    lr_scheduler = MultiFactorScheduler([30,60,95], 0.1, slow_step=10)
    lr_scheduler.base_lr = 0.1
    assert reldiff(lr_scheduler(60), 0.01) < 1e-5
    assert reldiff(lr_scheduler(61), 0.001) < 1e-5
