# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


from nose.tools import assert_true

def assert_results(MACHINE_TYPE, IS_GPU, GPU_NUM, profile_output, CPU_BENCHMARK_RESULTS, GPU_1_BENCHMARK_RESULTS, GPU_2_BENCHMARK_RESULTS, GPU_4_BENCHMARK_RESULTS, GPU_8_BENCHMARK_RESULTS):
    """
        Helps in asserting benchmarking results.
        Compares actual output result in profile_output with expected result in
        CPU_BENCHMARK_RESULTS if IS_GPU is True.
        Else, compares with GPU_1_BENCHMARK_RESULTS, GPU_2_BENCHMARK_RESULTS
        GPU_4_BENCHMARK_RESULTS and GPU_8_BENCHMARK_RESULTS.

        Uses keys - MODEL, TRAINING_TIME, MEM_CONSUMPTION, TRAIN_ACCURACY and TEST_ACCURACY
        to fetch data from provided actual and expected results input map stated above.
    """
    # Model type
    model = profile_output['MODEL']

    # Actual values.
    actual_training_time = profile_output['TRAINING_TIME']
    actual_memory_consumption = profile_output['MEM_CONSUMPTION']
    actual_train_accuracy = profile_output['TRAIN_ACCURACY']
    actual_test_accuracy = profile_output['TEST_ACCURACY']

    # Expected values
    expected_training_time = 0.0
    expected_memory_consumption = 0.0
    expected_train_accuracy = 1.0
    expected_test_accuracy = 1.0

    # Set right set of expected values based on current run type
    if(IS_GPU):
        if GPU_NUM == 1:
            expected_training_time = GPU_1_BENCHMARK_RESULTS['TRAINING_TIME']
            expected_memory_consumption = GPU_1_BENCHMARK_RESULTS['MEM_CONSUMPTION']
            expected_train_accuracy = GPU_1_BENCHMARK_RESULTS['TRAIN_ACCURACY']
            expected_test_accuracy = GPU_1_BENCHMARK_RESULTS['TEST_ACCURACY']
        elif GPU_NUM == 2:
            expected_training_time = GPU_2_BENCHMARK_RESULTS['TRAINING_TIME']
            expected_memory_consumption = GPU_2_BENCHMARK_RESULTS['MEM_CONSUMPTION']
            expected_train_accuracy = GPU_2_BENCHMARK_RESULTS['TRAIN_ACCURACY']
            expected_test_accuracy = GPU_2_BENCHMARK_RESULTS['TEST_ACCURACY']
        elif GPU_NUM == 4:
            expected_training_time = GPU_4_BENCHMARK_RESULTS['TRAINING_TIME']
            expected_memory_consumption = GPU_4_BENCHMARK_RESULTS['MEM_CONSUMPTION']
            expected_train_accuracy = GPU_4_BENCHMARK_RESULTS['TRAIN_ACCURACY']
            expected_test_accuracy = GPU_4_BENCHMARK_RESULTS['TEST_ACCURACY']
        elif GPU_NUM == 8:
            expected_training_time = GPU_8_BENCHMARK_RESULTS['TRAINING_TIME']
            expected_memory_consumption = GPU_8_BENCHMARK_RESULTS['MEM_CONSUMPTION']
            expected_train_accuracy = GPU_8_BENCHMARK_RESULTS['TRAIN_ACCURACY']
            expected_test_accuracy = GPU_8_BENCHMARK_RESULTS['TEST_ACCURACY']
    else:
        expected_training_time = CPU_BENCHMARK_RESULTS['TRAINING_TIME']
        expected_memory_consumption = CPU_BENCHMARK_RESULTS['MEM_CONSUMPTION']
        expected_train_accuracy = CPU_BENCHMARK_RESULTS['TRAIN_ACCURACY']
        expected_test_accuracy = CPU_BENCHMARK_RESULTS['TEST_ACCURACY']

    # Validate Results
    assert_true(actual_training_time < expected_training_time,'{0} on {1} machine with {2} GPU usage FAILED. Expected Training Time - {3} secs but was {4} secs.'.format(model, MACHINE_TYPE, GPU_NUM, expected_training_time, actual_training_time))
    assert_true(actual_memory_consumption < expected_memory_consumption, '{0} on {1} machine with {2} GPU usage FAILED. Expected Mem Consumption - {3} MB but was {4} MB.'.format(model, MACHINE_TYPE, GPU_NUM, expected_memory_consumption, actual_memory_consumption))
    assert_true(actual_train_accuracy > expected_train_accuracy, '{0} on {1} machine with {2} GPU usage FAILED. Expected Train Accuracy - {3} but was {4}.'.format(model, MACHINE_TYPE, GPU_NUM, expected_train_accuracy, actual_train_accuracy))
    assert_true(actual_test_accuracy > expected_test_accuracy, '{0} on {1} machine with {2} GPU usage FAILED. Expected Test Accuracy - {3} but was {4}.'.format(model, MACHINE_TYPE, GPU_NUM, expected_test_accuracy, actual_test_accuracy))
