# Clojure MXNet

A clojure package to the MXNet Deep Learning library

## Introduction

MXNet is a first class, modern deep learning library that AWS has officially picked as its chosen library. It supports multiple languages on a first class basis and is incubating as an Apache project.

The motivation for creating a Clojure package is to be able to open the deep learning library to the Clojure ecosystem and build bridges for future development and innovation for the community. It provides all the needed tools including low level and high level apis, dynamic graphs, and things like GAN and natural language support.

For high leverage, the Clojure package has been built on the existing Scala package using interop. This has allowed rapid development and close parity with the Scala functionality. This also leaves the door open to directly developing code against the jni-bindings with Clojure in the future in an incremental fashion, using the test suites as a refactoring guide.

