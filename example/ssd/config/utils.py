import collections

class DotDict(dict):
    """
    Simple class for dot access elements in dict, support nested initialization
    Example:
    d = DotDict({'child': 'dotdict'}, name='dotdict', index=1, contents=['a', 'b'])
    # add new key
    d.new_key = '!' # or d['new_key'] = '!'
    # update values
    d.new_key = '!!!'
    # delete keys
    del d.new_key
    """
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]


def namedtuple_with_defaults(typename, field_names, default_values=()):
    """ create a namedtuple with default values """
    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None, ) * len(T._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T

def merge_dict(a, b):
    """ merge dict a, b, with b overriding keys in a """
    c = a.copy()
    c.update(b)
    return c

def zip_namedtuple(nt_list):
    """ accept list of namedtuple, return a dict of zipped fields """
    if not nt_list:
        return dict()
    if not isinstance(nt_list, list):
        nt_list = [nt_list]
    for nt in nt_list:
        assert type(nt) == type(nt_list[0])
    ret = {k : [v] for k, v in nt_list[0]._asdict().items()}
    for nt in nt_list[1:]:
        for k, v in nt._asdict().items():
            ret[k].append(v)
    return ret

def config_as_dict(cfg):
    """ convert raw configuration to unified dictionary """
    ret = cfg.__dict__.copy()
    # random cropping params
    del ret['rand_crop_samplers']
    assert isinstance(cfg.rand_crop_samplers, list)
    ret = merge_dict(ret, zip_namedtuple(cfg.rand_crop_samplers))
    num_crop_sampler = len(cfg.rand_crop_samplers)
    ret['num_crop_sampler'] = num_crop_sampler  # must specify the #
    ret['rand_crop_prob'] = 1.0 / (num_crop_sampler + 1) * num_crop_sampler
    # random padding params
    del ret['rand_pad']
    ret = merge_dict(ret, cfg.rand_pad._asdict())
    # color jitter
    del ret['color_jitter']
    ret = merge_dict(ret, cfg.color_jitter._asdict())
    return ret
