import mxnet as mx
import mxnet.ndarray as nd
import time
import logging
from utils import *


def calc_potential(exe, params, label_name, noise_precision, prior_precision):
    exe.copy_params_from(params)
    exe.forward(is_train=False)
    ret = 0.0
    ret += (nd.norm(
        exe.outputs[0] - exe.arg_dict[label_name]).asscalar() ** 2) / 2.0 * noise_precision
    for v in params.values():
        ret += (nd.norm(v).asscalar() ** 2) / 2.0 * prior_precision
    return ret


def calc_grad(exe, exe_grads, params, X, Y, label_name=None, outgrad_f=None):
    exe.copy_params_from(params)
    exe.arg_dict['data'][:] = X
    if outgrad_f is None:
        exe.arg_dict[label_name][:] = Y
        exe.forward(is_train=True)
        exe.backward()
    else:
        exe.forward(is_train=True)
        exe.backward(outgrad_f(exe.outpus, Y))
    for k, v in exe_grads.items():
        v.wait_to_read()


def step_HMC(exe, exe_params, exe_grads, label_key, noise_precision, prior_precision, L=10,
             eps=1E-6):
    init_params = {k: v.copyto(v.context) for k, v in exe_params.items()}
    end_params = {k: v.copyto(v.context) for k, v in exe_params.items()}
    init_momentums = {k: mx.random.normal(0, 1, v.shape) for k, v in init_params.items()}
    end_momentums = {k: v.copyto(v.context) for k, v in init_momentums.items()}
    init_potential = calc_potential(exe, init_params, label_key, noise_precision, prior_precision)

    # 0. Calculate Initial Energy and Kinetic
    init_kinetic = sum([nd.sum(nd.square(momentum)) / 2.0
                        for momentum in init_momentums.values()]).asscalar()
    # 1. Make a half step for momentum at the beginning
    exe.copy_params_from(end_params)
    exe.forward(is_train=True)
    exe.backward()
    for k, v in exe_grads.items():
        v.wait_to_read()
    for k, momentum in end_momentums.items():
        momentum[:] = momentum - (eps / 2) * exe_grads[k]
    # 2. Alternate full steps for position and momentum
    for i in range(L):
        # 2.1 Full step for position
        for k, param in exe_params.items():
            param[:] = param + eps * end_momentums[k]
        # 2.2 Full step for the momentum, except at the end of trajectory we perform a half step
        exe.forward(is_train=True)
        exe.backward()
        for v in exe_grads.values():
            v.wait_to_read()
        if i != L - 1:
            for k, momentum in end_momentums.items():
                momentum[:] = momentum - eps * exe_grads[k]
        else:
            for k, momentum in end_momentums.items():
                # We should reverse the sign of the momentum at the end
                momentum[:] = -(momentum - eps / 2.0 * exe_grads[k])
    copy_param(exe, end_params)
    # 3. Calculate acceptance ratio and accept/reject the move
    end_potential = calc_potential(exe, end_params, label_key, noise_precision, prior_precision)
    end_kinetic = sum([nd.sum(nd.square(momentum)) / 2.0
                       for momentum in end_momentums.values()]).asscalar()
    # print init_potential, init_kinetic, end_potential, end_kinetic
    r = numpy.random.rand(1)
    if r < numpy.exp(-(end_potential + end_kinetic) + (init_potential + init_kinetic)):
        exe.copy_params_from(end_params)
        return end_params, 1
    else:
        exe.copy_params_from(init_params)
        return init_params, 0


def HMC(sym, data_inputs, X, Y, X_test, Y_test, sample_num,
        initializer=None, noise_precision=1 / 9.0, prior_precision=0.1,
        learning_rate=1E-6, L=10, dev=mx.gpu()):
    label_key = list(set(data_inputs.keys()) - set(['data']))[0]
    exe, exe_params, exe_grads, _ = get_executor(sym, dev, data_inputs, initializer)
    exe.arg_dict['data'][:] = X
    exe.arg_dict[label_key][:] = Y
    sample_pool = []
    accept_num = 0
    start = time.time()
    for i in xrange(sample_num):
        sample_params, is_accept = step_HMC(exe, exe_params, exe_grads, label_key, noise_precision,
                                            prior_precision, L, learning_rate)
        accept_num += is_accept

        if (i + 1) % 10 == 0:
            sample_pool.append(sample_params)
            if (i + 1) % 100000 == 0:
                end = time.time()
                print "Current Iter Num: %d" % (i + 1), "Time Spent: %f" % (end - start), "MSE:",
                print sample_test_regression(exe, X=X_test, Y=Y_test, sample_pool=sample_pool,
                                             minibatch_size=Y.shape[0],
                                             save_path='regression_HMC.txt')
                start = time.time()
        exe.copy_params_from(sample_params)
    print 'accept ratio', accept_num / float(sample_num)
    return sample_pool


def SGD(sym, data_inputs, X, Y, X_test, Y_test, total_iter_num,
        lr=None,
        lr_scheduler=None, prior_precision=1,
        out_grad_f=None,
        initializer=None,
        minibatch_size=100, dev=mx.gpu()):
    if out_grad_f is None:
        label_key = list(set(data_inputs.keys()) - set(['data']))[0]
    exe, params, params_grad, _ = get_executor(sym, dev, data_inputs, initializer)
    optimizer = mx.optimizer.create('sgd', learning_rate=lr,
                                    rescale_grad=X.shape[0] / minibatch_size,
                                    lr_scheduler=lr_scheduler,
                                    wd=prior_precision,
                                    arg_names=params.keys())
    updater = mx.optimizer.get_updater(optimizer)
    start = time.time()
    for i in xrange(total_iter_num):
        indices = numpy.random.randint(X.shape[0], size=minibatch_size)
        X_batch = X[indices]
        Y_batch = Y[indices]
        exe.arg_dict['data'][:] = X_batch
        if out_grad_f is None:
            exe.arg_dict[label_key][:] = Y_batch
            exe.forward(is_train=True)
            exe.backward()
        else:
            exe.forward(is_train=True)
            exe.backward(out_grad_f(exe.outputs, nd.array(Y_batch, ctx=dev)))
        for k in params:
            updater(k, params_grad[k], params[k])
        if (i + 1) % 500 == 0:
            end = time.time()
            print "Current Iter Num: %d" % (i + 1), "Time Spent: %f" % (end - start)
            sample_test_acc(exe, X=X_test, Y=Y_test, label_num=10, minibatch_size=100)
            start = time.time()
    return exe, params, params_grad


def SGLD(sym, X, Y, X_test, Y_test, total_iter_num,
         data_inputs=None,
         learning_rate=None,
         lr_scheduler=None, prior_precision=1,
         out_grad_f=None,
         initializer=None,
         minibatch_size=100, thin_interval=100, burn_in_iter_num=1000, task='classification',
         dev=mx.gpu()):
    if out_grad_f is None:
        label_key = list(set(data_inputs.keys()) - set(['data']))[0]
    exe, params, params_grad, _ = get_executor(sym, dev, data_inputs, initializer)
    optimizer = mx.optimizer.create('sgld', learning_rate=learning_rate,
                                    rescale_grad=X.shape[0] / minibatch_size,
                                    lr_scheduler=lr_scheduler,
                                    wd=prior_precision)
    updater = mx.optimizer.get_updater(optimizer)
    sample_pool = []
    start = time.time()
    for i in xrange(total_iter_num):
        indices = numpy.random.randint(X.shape[0], size=minibatch_size)
        X_batch = X[indices]
        Y_batch = Y[indices]
        exe.arg_dict['data'][:] = X_batch
        if out_grad_f is None:
            exe.arg_dict[label_key][:] = Y_batch
            exe.forward(is_train=True)
            exe.backward()
        else:
            exe.forward(is_train=True)
            exe.backward(out_grad_f(exe.outputs, nd.array(Y_batch, ctx=dev)))
        for k in params:
            updater(k, params_grad[k], params[k])
        if i < burn_in_iter_num:
            continue
        else:
            if 0 == (i - burn_in_iter_num) % thin_interval:
                if optimizer.lr_scheduler is not None:
                    lr = optimizer.lr_scheduler(optimizer.num_update)
                else:
                    lr = learning_rate
                sample_pool.append([lr, copy_param(exe)])
        if (i + 1) % 100000 == 0:
            end = time.time()
            if task == 'classification':
                print "Current Iter Num: %d" % (i + 1), "Time Spent: %f" % (end - start)
                test_correct, test_total, test_acc = \
                    sample_test_acc(exe, sample_pool=sample_pool, X=X_test, Y=Y_test, label_num=10,
                                    minibatch_size=minibatch_size)
                print "Test %d/%d=%f" % (test_correct, test_total, test_acc)
            else:
                print "Current Iter Num: %d" % (i + 1), "Time Spent: %f" % (end - start), "MSE:",
                print sample_test_regression(exe=exe, sample_pool=sample_pool,
                                             X=X_test,
                                             Y=Y_test, minibatch_size=minibatch_size,
                                             save_path='regression_SGLD.txt')
            start = time.time()
    return exe, sample_pool


def DistilledSGLD(teacher_sym, student_sym,
                  teacher_data_inputs, student_data_inputs,
                  X, Y, X_test, Y_test, total_iter_num,
                  teacher_learning_rate, student_learning_rate,
                  teacher_lr_scheduler=None, student_lr_scheduler=None,
                  student_optimizing_algorithm='sgd',
                  teacher_grad_f=None, student_grad_f=None,
                  teacher_prior_precision=1, student_prior_precision=0.001,
                  perturb_deviation=0.001,
                  student_initializer=None,
                  teacher_initializer=None,
                  minibatch_size=100,
                  task='classification',
                  dev=mx.gpu()):
    teacher_exe, teacher_params, teacher_params_grad, _ = \
        get_executor(teacher_sym, dev, teacher_data_inputs, teacher_initializer)
    student_exe, student_params, student_params_grad, _ = \
        get_executor(student_sym, dev, student_data_inputs, student_initializer)
    if teacher_grad_f is None:
        teacher_label_key = list(set(teacher_data_inputs.keys()) - set(['data']))[0]
    if student_grad_f is None:
        student_label_key = list(set(student_data_inputs.keys()) - set(['data']))[0]
    teacher_optimizer = mx.optimizer.create('sgld',
                                            learning_rate=teacher_learning_rate,
                                            rescale_grad=X.shape[0] / float(minibatch_size),
                                            lr_scheduler=teacher_lr_scheduler,
                                            wd=teacher_prior_precision)
    student_optimizer = mx.optimizer.create(student_optimizing_algorithm,
                                            learning_rate=student_learning_rate,
                                            rescale_grad=1.0 / float(minibatch_size),
                                            lr_scheduler=student_lr_scheduler,
                                            wd=student_prior_precision)
    teacher_updater = mx.optimizer.get_updater(teacher_optimizer)
    student_updater = mx.optimizer.get_updater(student_optimizer)
    start = time.time()
    for i in xrange(total_iter_num):
        # 1.1 Draw random minibatch
        indices = numpy.random.randint(X.shape[0], size=minibatch_size)
        X_batch = X[indices]
        Y_batch = Y[indices]

        # 1.2 Update teacher
        teacher_exe.arg_dict['data'][:] = X_batch
        if teacher_grad_f is None:
            teacher_exe.arg_dict[teacher_label_key][:] = Y_batch
            teacher_exe.forward(is_train=True)
            teacher_exe.backward()
        else:
            teacher_exe.forward(is_train=True)
            teacher_exe.backward(
                teacher_grad_f(teacher_exe.outputs, nd.array(Y_batch, ctx=dev)))

        for k in teacher_params:
            teacher_updater(k, teacher_params_grad[k], teacher_params[k])

        # 2.1 Draw random minibatch and do random perturbation
        if task == 'classification':
            indices = numpy.random.randint(X.shape[0], size=minibatch_size)
            X_student_batch = X[indices] + numpy.random.normal(0,
                                                               perturb_deviation,
                                                               X_batch.shape).astype('float32')
        else:
            X_student_batch = mx.random.uniform(-6, 6, X_batch.shape, mx.cpu())

        # 2.2 Get teacher predictions
        teacher_exe.arg_dict['data'][:] = X_student_batch
        teacher_exe.forward(is_train=False)
        teacher_pred = teacher_exe.outputs[0]
        teacher_pred.wait_to_read()

        # 2.3 Update student
        student_exe.arg_dict['data'][:] = X_student_batch
        if student_grad_f is None:
            student_exe.arg_dict[student_label_key][:] = teacher_pred
            student_exe.forward(is_train=True)
            student_exe.backward()
        else:
            student_exe.forward(is_train=True)
            student_exe.backward(student_grad_f(student_exe.outputs, teacher_pred))
        for k in student_params:
            student_updater(k, student_params_grad[k], student_params[k])

        if (i + 1) % 2000 == 0:
            end = time.time()
            if task == 'classification':
                print "Current Iter Num: %d" % (i + 1), "Time Spent: %f" % (end - start)
                test_correct, test_total, test_acc = \
                    sample_test_acc(student_exe, X=X_test, Y=Y_test, label_num=10,
                                    minibatch_size=minibatch_size)
                train_correct, train_total, train_acc = \
                    sample_test_acc(student_exe, X=X, Y=Y, label_num=10,
                                    minibatch_size=minibatch_size)
                teacher_test_correct, teacher_test_total, teacher_test_acc = \
                    sample_test_acc(teacher_exe, X=X_test, Y=Y_test, label_num=10,
                                    minibatch_size=minibatch_size)
                teacher_train_correct, teacher_train_total, teacher_train_acc = \
                    sample_test_acc(teacher_exe, X=X, Y=Y, label_num=10,
                                    minibatch_size=minibatch_size)
                print "Student: Test ACC %d/%d=%f, Train ACC %d/%d=%f" % (test_correct, test_total,
                                                    test_acc, train_correct, train_total, train_acc)
                print "Teacher: Test ACC %d/%d=%f, Train ACC %d/%d=%f" \
                      % (teacher_test_correct, teacher_test_total, teacher_test_acc,
                         teacher_train_correct, teacher_train_total, teacher_train_acc)
            else:
                print "Current Iter Num: %d" % (i + 1), "Time Spent: %f" % (end - start), "MSE:",
                print sample_test_regression(exe=student_exe, X=X_test, Y=Y_test,
                                             minibatch_size=minibatch_size,
                                             save_path='regression_DSGLD.txt')
            start = time.time()

    return student_exe, student_params, student_params_grad
