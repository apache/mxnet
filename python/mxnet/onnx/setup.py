from setuptools import setup, find_packages

setup(
    name='mx2onnx',
    version='0.0.0',
    description='Module to convert MXNet models to the ONNX format',
    author='',
    author_email='',
    url='https://github.com/apache/incubator-mxnet/tree/v1.x/python/mxnet/onnx',
    install_requires=[
        'onnx',
        'onnxoptimizer >= 0.2.4',
        'onnxruntime >= 1.6.0',
        'protobuf >= 3.7.0'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    packages=find_packages(),
    python_requires='>=3.6'
)
