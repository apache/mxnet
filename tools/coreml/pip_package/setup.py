from setuptools import setup
from setuptools import find_packages

# We are overriding the default behavior of bdist_wheel which is generating
# pure python wheels while we need platform specific wheel since this tool
# can only work on MacOS.
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None


def readme():
    """
    Reads README.rst file and allows us to provide
    a better experience for pypi webpage.
    """
    with open('README.rst') as f:
        return f.read()

setup(name='mxnet-coreml-converter',
      version='0.1.0a13',
      description='Tool to convert MXNet models into Apple CoreML model format.',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 2.7',
        'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      keywords='Apache MXNet Apple CoreML Converter Deep Learning',
      url='https://github.com/apache/incubator-mxnet/tree/master/tools/coreml',
      author='pracheer',
      author_email='pracheer_gupta@hotmail.com',
      license='Apache 2.0',
      package_dir = {'': '..'},
      packages=['converter'],
      install_requires=[
          'mxnet',
          'coremltools',
          'pyyaml',
      ],
      scripts=['../mxnet_coreml_converter.py'],
      python_requires='~=2.7',
      zip_safe=False,
      cmdclass={'bdist_wheel': bdist_wheel},)
