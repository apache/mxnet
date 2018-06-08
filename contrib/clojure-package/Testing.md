## Help with Testing

If you want to give the repo a spin and help make it stable and ready for prime time that would be awesome.

Here is what you can do.

* Clone the project
* Edit the project.clj file and uncomment the line that is for your system (OSX, Linux CPU, or Linux GPU)
* Run `lein deps` (this might take a bit - the jars are big!)
* Run `lein test` - there should be no errors. The tests are all cpu
* Run `lein install` to install the clojure-package locally
* Go to the module examples `cd examples/module`
* Either run `lein run` or `lein run :gpu`

If you find any problems, please log on issue.

Thanks!

## Want to explore more?

The examples/tutorial is a good REPL walkthrough
The examples/pre-trained-modules is nice too
The examples/gan is just plain fun :)
