
class Types:
    def __init__(self, *values):
        self.values = values


class AllTypes(Types):
    def __init__(self):
        Types.__init__(self, "float32", "float64", "float16",
                       "uint8", "int8", "int32", "int64")


class RealTypes(Types):
    def __init__(self):
        Types.__init__(self, "float32", "float64", "float16")
