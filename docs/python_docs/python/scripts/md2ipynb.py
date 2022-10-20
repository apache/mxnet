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

import sys
import os
import time
import notedown
import nbformat

def md2ipynb():
    assert len(sys.argv) == 3, 'usage: input.md output.rst'
    (src_fn, input_fn, output_fn) = sys.argv

    # timeout for each notebook, in sec
    timeout = 60 * 60
    # if enable evaluation
    do_eval = int(os.environ.get('EVAL', True))
    
    # Skip these notebooks as some APIs will no longer be used
    skip_list = ["pytorch.md", "mnist.md", "custom-loss.md", "fit_api_tutorial.md", \
        "01-ndarray-intro.md", "02-ndarray-operations.md", "03-ndarray-contexts.md", \
        "gotchas_numpy_in_mxnet.md", "csr.md", "row_sparse.md", "fine_tuning_gluon.md", \
        "inference_on_onnx_model.md", "amp.md", "profiler.md"]

    require_gpu = []
    # the files will be ignored for execution
    ignore_execution = skip_list + require_gpu

    reader = notedown.MarkdownReader(match='strict')
    with open(input_fn, 'r', encoding="utf8") as f:
        notebook = reader.read(f)
    if do_eval:
        if not any([i in input_fn for i in ignore_execution]):
            tic = time.time()
            notedown.run(notebook, timeout)
            print(f'{src_fn}: Evaluated {input_fn} in {time.time()-tic} sec')
    # need to add language info to for syntax highlight
    notebook['metadata'].update({'language_info':{'name':'python'}})
    with open(output_fn, 'w', encoding='utf-8') as f:
        f.write(nbformat.writes(notebook))
    print(f'{src_fn}: Write results into {output_fn}')

if __name__ == '__main__':
    md2ipynb()
