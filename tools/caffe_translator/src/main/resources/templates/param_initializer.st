class ParamInitializer():
    lst_patterns = []
    lst_initializers = []

    def add_param(self, pattern, initializer):
        self.lst_patterns.append(pattern)
        self.lst_initializers.append(initializer)

    def get_initializer(self, default_initializer):
        self.lst_patterns.append(".*")
        self.lst_initializers.append(default_initializer)
        return mx.initializer.Mixed(self.lst_patterns, self.lst_initializers)
