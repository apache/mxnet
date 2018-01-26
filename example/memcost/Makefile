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
# under the License

.PHONY: no_optimization with_inplace with_sharing with_both

no_optimization:
	@echo "Estimating the cost with no optimization..."
	@NNVM_EXEC_ENABLE_INPLACE=false NNVM_EXEC_MATCH_RANGE=0 python inception_memcost.py

with_inplace:
	@echo "Estimating the cost with inplace optimization..."
	@NNVM_EXEC_ENABLE_INPLACE=true MXNET_EXEC_MATCH_RANGE=0 python inception_memcost.py

with_sharing:
	@echo "Estimating the cost with memory sharing ..."
	@NNVM_EXEC_ENABLE_INPLACE=false python inception_memcost.py

with_both:
	@echo "Estimating the cost with all optimizations ..."
	@python inception_memcost.py

forward_only:
	@echo "Estimating the cost of forward only ..."
	@python inception_memcost.py 'null'
