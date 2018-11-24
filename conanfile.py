from conans import ConanFile

class IncubatorMXNetConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    requires = "openblas/0.2.20@conan/stable", "opencv/3.4.3@conan/stable", "lapack/3.7.1@conan/stable"
    generators = ["cmake"]

    def configure(self):
        if self.settings.compiler == "Visual Studio":
            self.options["lapack"].visual_studio = True
            self.options["lapack"].shared = True
