integration_test() {
    # install gradle
    add-apt-repository ppa:cwchien/gradle
    apt-get update -y
    apt-get install gradle -y
    # build java prokect
    cd /work/mxnet/java-package
    ./gradle build -x javadoc
    # generate native library
    ./gradlew :native:buildLocalLibraryJarDefault
    ./gradlew :native:mkl-linuxJar
    # run integration
    ./gradlew :integration:run
}

