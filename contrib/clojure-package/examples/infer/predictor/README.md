# predictor

Run model prediction using clojure infer package.

## Installation

Before you run this example, make sure that you have the clojure package installed.
In the main clojure package directory, do `lein install`. Then you can run
`lein install` in this directory.

## Usage

```
$ chmod +x scripts/get_resnet_18_data.sh
$ ./scripts/get_resnet_18_data.sh
$
$ lein run -- --help
$ lein run -- -m models/resnet-18/resnet-18 -i images/kitten.jpg
$
$ lein uberjar
$ java -jar target/predictor-0.1.0-SNAPSHOT-standalone.jar --help
$ java -jar target/predictor-0.1.0-SNAPSHOT-standalone.jar \
    -m models/resnet-18/resnet-18 -i images/kitten.jpg
```
