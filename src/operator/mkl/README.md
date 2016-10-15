# MKL2017 PLUGIN

MKL2017 is one INTEL released library to accelerate Deep Neural Network (DNN) applications on Intel architecture.
This README shows user how to setup and install MKL2017 library with mxnet.


## Build/Install Instructions:
```
Download MKL:
  1. Navigate here - https://registrationcenter.intel.com/en/forms/?productid=2558&licensetype=2
  2. Input email/country and hit submit, you will receive an email shortly
  3. Save the serial # in the email for future use
  4. Click the link under the serial # "Intel? Software Development Products Registration Center"
  5. Select the "Intel Math Kernel Library" radio button, click "Download Now"
  6. Copy l_mkl_2017.0.098.tgz to your system and untar
  7. Run l_mkl_2017.0.098/install.sh
  8. Choose your installation distribution (root, sudo or user)
  9. Hit Enter at the next prompt
  10. Read through license agreement, type accept, hit Enter
  11. Hit Enter, enter your serial #
  12. Select whichever option you'd like, hit enter
  13. Select 1, hit enter, wait for installation to complete
```

## Build/Install MxNet
```
  1. Navigate to directory you installed MKL into (default is /opt/intel/compilers_and_libraries_2017.0.098/linux/mkl/
  2. Run 'source bin/mklvars.sh intel64'
  3. Navigate to your MxNet directory
  5. Enable USE_MKL2017=1 in make/config.mk
  7. Run 'make -jX'
  8. Navigate into the python directory
  9. Run 'python setup.py install'
```
