import logging as log

class Singleton:
    def __init__(self, decrated):
        log.debug("Singleton Init %s" % decrated)
        self._decorated = decrated

    def getInstance(self):
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __new__(class_, *args, **kwargs):
        print "__new__"
        class_.instances[class_] = super(Singleton, class_).__new__(class_, *args, **kwargs)
        return class_.instances[class_]

    def __call__(self):
        raise TypeError("Singletons must be accessed through 'getInstance()'")


class SingletonInstane:
  __instance = None

  @classmethod
  def __getInstance(cls):
    return cls.__instance

  @classmethod
  def instance(cls, *args, **kargs):
    cls.__instance = cls(*args, **kargs)
    cls.instance = cls.__getInstance
    return cls.__instance

