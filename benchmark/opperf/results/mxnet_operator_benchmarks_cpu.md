<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# MXNet Operator Benchmarks

## Settings

1. MXNet - v1.4.1
2. Instance - C5.8x

| Operator | Avg Forward Time (ms) | Avg. Backward Time (ms) | Max Mem Usage (Bytes) | Inputs |
| :---: | :---: | :---: | :---:| :--- |
| not_equal | 4.924 | --- | --- | {'lhs': (1024, 1024), 'rhs': (1024, 1024)} |
| not_equal | 0.3699 | --- | --- | {'lhs': (10000, 10), 'rhs': (10000, 10)} |
| not_equal | 15.061 | --- | --- | {'lhs': (10000, 1), 'rhs': (10000, 100)} |
| divide | 4.0465 | 14.9366 | --- | {'lhs': (1024, 1024), 'rhs': (1024, 1024)} |
| divide | 0.411 | 1.2941 | --- | {'lhs': (10000, 10), 'rhs': (10000, 10)} |
| divide | 10.9992 | 224.3602 | --- | {'lhs': (10000, 1), 'rhs': (10000, 100)} |
| logical_and | 7.0478 | --- | --- | {'lhs': (1024, 1024), 'rhs': (1024, 1024)} |
| logical_and | 0.6158 | --- | --- | {'lhs': (10000, 10), 'rhs': (10000, 10)} |
| logical_and | 15.9246 | --- | --- | {'lhs': (10000, 1), 'rhs': (10000, 100)} |
| greater_equal | 4.2214 | --- | --- | {'lhs': (1024, 1024), 'rhs': (1024, 1024)} |
| greater_equal | 0.3798 | --- | --- | {'lhs': (10000, 10), 'rhs': (10000, 10)} |
| greater_equal | 11.6242 | --- | --- | {'lhs': (10000, 1), 'rhs': (10000, 100)} |
| dot | 20.2176 | 40.6154 | --- | {'lhs': (1024, 1024), 'rhs': (1024, 1024)} |
| dot | 0.8916 | 1.6323 | --- | {'lhs': (1000, 10), 'rhs': (1000, 10), 'transpose_b': True} |
| dot | 0.0549 | 0.1352 | --- | {'lhs': (1000, 1), 'rhs': (100, 1000), 'transpose_a': True, 'transpose_b': True} |
| lesser | 4.6047 | --- | --- | {'lhs': (1024, 1024), 'rhs': (1024, 1024)} |
| lesser | 0.4203 | --- | --- | {'lhs': (10000, 10), 'rhs': (10000, 10)} |
| lesser | 11.5406 | --- | --- | {'lhs': (10000, 1), 'rhs': (10000, 100)} |
| logical_xor | 8.0958 | --- | --- | {'lhs': (1024, 1024), 'rhs': (1024, 1024)} |
| logical_xor | 0.6891 | --- | --- | {'lhs': (10000, 10), 'rhs': (10000, 10)} |
| logical_xor | 16.4655 | --- | --- | {'lhs': (10000, 1), 'rhs': (10000, 100)} |
| logical_not | 4.3194 | --- | --- | {'data': (1024, 1024)} |
| logical_not | 0.3498 | --- | --- | {'data': (10000, 10)} |
| logical_not | 0.0593 | --- | --- | {'data': (10000, 1)} |
| multiply | 5.1172 | 14.493 | --- | {'lhs': (1024, 1024), 'rhs': (1024, 1024)} |
| multiply | 0.5056 | 1.1935 | --- | {'lhs': (10000, 10), 'rhs': (10000, 10)} |
| multiply | 11.1305 | 224.7644 | --- | {'lhs': (10000, 1), 'rhs': (10000, 100)} |
| batch_dot | 691.0597 | 1272.681 | --- | {'lhs': (32, 1024, 1024), 'rhs': (32, 1024, 1024)} |
| batch_dot | 41.9577 | 54.7069 | --- | {'lhs': (32, 1000, 10), 'rhs': (32, 1000, 10), 'transpose_b': True} |
| batch_dot | 2.0083 | 4.8756 | --- | {'lhs': (32, 1000, 1), 'rhs': (32, 100, 1000), 'transpose_a': True, 'transpose_b': True} |
| greater | 4.6383 | --- | --- | {'lhs': (1024, 1024), 'rhs': (1024, 1024)} |
| greater | 0.4199 | --- | --- | {'lhs': (10000, 10), 'rhs': (10000, 10)} |
| greater | 12.0976 | --- | --- | {'lhs': (10000, 1), 'rhs': (10000, 100)} |
| modulo | 28.3522 | 11.9746 | --- | {'lhs': (1024, 1024), 'rhs': (1024, 1024)} |
| modulo | 2.8529 | 1.1428 | --- | {'lhs': (10000, 10), 'rhs': (10000, 10)} |
| modulo | 31.587 | 233.5845 | --- | {'lhs': (10000, 1), 'rhs': (10000, 100)} |
| equal | 4.344 | --- | --- | {'lhs': (1024, 1024), 'rhs': (1024, 1024)} |
| equal | 0.3686 | --- | --- | {'lhs': (10000, 10), 'rhs': (10000, 10)} |
| equal | 10.8763 | --- | --- | {'lhs': (10000, 1), 'rhs': (10000, 100)} |
| subtract | 4.2592 | 5.7328 | --- | {'lhs': (1024, 1024), 'rhs': (1024, 1024)} |
| subtract | 0.4767 | 0.5264 | --- | {'lhs': (10000, 10), 'rhs': (10000, 10)} |
| subtract | 12.0121 | 51.3822 | --- | {'lhs': (10000, 1), 'rhs': (10000, 100)} |
| negative | 3.3177 | --- | --- | {'data': (1024, 1024)} |
| negative | 0.345 | --- | --- | {'data': (10000, 10)} |
| lesser_equal | 5.3383 | --- | --- | {'lhs': (1024, 1024), 'rhs': (1024, 1024)} |
| lesser_equal | 0.4142 | --- | --- | {'lhs': (10000, 10), 'rhs': (10000, 10)} |
| lesser_equal | 11.7726 | --- | --- | {'lhs': (10000, 1), 'rhs': (10000, 100)} |
| add | 4.2209 | 5.4922 | --- | {'lhs': (1024, 1024), 'rhs': (1024, 1024)} |
| add | 0.418 | 0.4788 | --- | {'lhs': (10000, 10), 'rhs': (10000, 10)} |
| add | 10.0945 | 46.5165 | --- | {'lhs': (10000, 1), 'rhs': (10000, 100)} |
| logical_or | 7.8223 | --- | --- | {'lhs': (1024, 1024), 'rhs': (1024, 1024)} |
| logical_or | 0.6505 | --- | --- | {'lhs': (10000, 10), 'rhs': (10000, 10)} |
| logical_or | 16.7953 | --- | --- | {'lhs': (10000, 1), 'rhs': (10000, 100)} |
| power | 27.4658 | 83.9539 | --- | {'base': (1024, 1024), 'exp': (1024, 1024)} |
| power | 1.8994 | 4.8235 | --- | {'base': (10000, 10), 'exp': (10000, 10)} |