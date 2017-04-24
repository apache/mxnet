import logging
import logging.handlers


class SingletonType(type):
    def __call__(cls, *args, **kwargs):
        try:
            return cls.__instance
        except AttributeError:
            cls.__instance = super(SingletonType, cls).__call__(*args, **kwargs)
            return cls.__instance


class LogUtil(object):
    __metaclass__ = SingletonType
    _logger = None
    _filename = None

    def __init__(self, filename=None):
        self._filename = filename

        # logger
        self._logger = logging.getLogger('logger')
        # remove default handler
        self._logger.propagate = False

        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter('[%(levelname)8s][%(asctime)s.%(msecs)03d] %(message)s',
                                             datefmt='%Y/%m/%d %H:%M:%S')
        stream_handler.setFormatter(stream_formatter)

        if self._filename is not None:
            file_max_bytes = 10 * 1024 * 1024

            file_handler = logging.handlers.RotatingFileHandler(filename='./log/' + self._filename,
                                                               maxBytes=file_max_bytes,
                                                               backupCount=10)
            file_formatter = logging.Formatter('[%(levelname)8s][%(asctime)s.%(msecs)03d] %(message)s',
                                               datefmt='%Y/%m/%d %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            self._logger.addHandler(file_handler)

        self._logger.addHandler(stream_handler)
        self._logger.setLevel(logging.DEBUG)

    def getlogger(self):
        return self._logger
