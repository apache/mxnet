# objectdetector

Run object detection on images using clojure infer package.

## Installation

`lein install`

## Usage

```
$ chmod +x scripts/get_ssd_data.sh
$ ./scripts/get_ssd_data.sh
$ lein uberjar
$ java -jar target/objectdetector-0.1.0-SNAPSHOT-standalone.jar --help
$ java -jar target/objectdetector-0.1.0-SNAPSHOT-standalone.jar \
    -m models/resnet50_ssd/resnet50_ssd_model -i images/kitten.jpg -d images/
```
