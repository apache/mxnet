def mx_lib = 'lib/libmxnet.so, lib/libmxnet.a, dmlc-core/libdmlc.a, nnvm/lib/libnnvm.a'
def mx_run = 'tests/ci_build/ci_build.sh'

def pack_lib(name, mx_lib) {
  sh """
echo "Packing ${mx_lib} into ${name}"
echo ${mx_lib} | sed -e 's/,/ /g' | xargs md5sum
"""
  stash includes: mx_lib, name: name
}


def unpack_lib(name, mx_lib) {
  unstash name
  sh """
echo "Unpacked ${mx_lib} from ${name}"
echo ${mx_lib} | sed -e 's/,/ /g' | xargs md5sum
"""
}

def init_git() {
  sh 'timeout 50s git submodule update --init'            
}
stage('Build') {
    node {
      ws('workspace/amalgamation') {
        checkout scm
        init_git()
        def flag = '-C amalgamation/ USE_BLAS=openblas MIN=1'        
        sh "${mx_run} cpu make ${flag}"        
      }
    }
  
}


