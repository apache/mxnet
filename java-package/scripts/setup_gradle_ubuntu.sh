install_gradle() {
    add-apt-repository ppa:cwchien/gradle
    apt-get update -y
    apt-get install gradle -y
}
