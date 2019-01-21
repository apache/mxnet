# objectdetector

Run object detection on images using clojure infer package.

## Installation

Before you run this example, make sure that you have the clojure package installed.
In the main clojure package directory, do `lein install`. Then you can run
`lein install` in this directory.

## Usage

```
$ chmod +x scripts/get_ssd_data.sh
$ ./scripts/get_ssd_data.sh
$
$ lein run -- --help
$ lein run -- -m models/resnet50_ssd/resnet50_ssd_model -i images/dog.jpg -d images/
$ 
$ # or the available lein alias
$ lein run-detector
$
$ lein uberjar
$ java -jar target/objectdetector-0.1.0-SNAPSHOT-standalone.jar --help
$ java -jar target/objectdetector-0.1.0-SNAPSHOT-standalone.jar \
    -m models/resnet50_ssd/resnet50_ssd_model -i images/dog.jpg -d images/
```
