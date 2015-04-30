# cxx-minerva

* Use google c style
* Put module header in [include](include)
  - move them to ```project-name/include``` when we finalized the name
* Depend on [dmlc-core](https://github.com/dmlc/dmlc-core)
* Doxygen comment every function, class and variable for the module headers
  - Ref headers in [dmlc-core/include](https://github.com/dmlc/dmlc-core/tree/master/include/dmlc)
  - Use the same style as dmlc-core
* Try write some use-cases of interface in [test](test)
  - They do not need to link, but need to pass compile
