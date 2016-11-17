import doctest
import logging
import mxnet
import numpy

def import_into(globs, module, names=None, error_on_overwrite=True):
    """Import names from module into the globs dict.

    Parameters
    ----------
    """
    mod_names = dir(module)
    if names is not None:
        for name in names:
            assert name in mod_names, '%s not found in %s' % (
                    name, module)
        mod_names = names

    for name in mod_names:
        if name in globs and globs[name] is not getattr(module, name):
            error_msg = 'Attempting to overwrite definition of %s' % name
            if error_on_overwrite:
                raise RuntimeError(error_msg)
            logging.warning('%s', error_msg)
        globs[name] = getattr(module, name)

    return globs


def test_symbols():
    globs = {'numpy': numpy, 'mxnet': mxnet, 'test_utils': mxnet.test_utils}

    # make sure all the operators are available
    import_into(globs, mxnet.symbol)

    doctest.testmod(mxnet.symbol_doc, globs=globs)


if __name__ == '__main__':
    test_symbols()
