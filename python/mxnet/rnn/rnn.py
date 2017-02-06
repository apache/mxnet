# coding: utf-8
# pylint: disable=too-many-arguments, no-member
"""Functions for constructing recurrent neural networks."""
import warnings

def rnn_unroll(cell, length, inputs=None, begin_state=None, input_prefix='', layout='NTC'):
    """Deprecated. Please use cell.unroll instead"""
    warnings.warn('rnn_unroll is deprecated. Please call cell.unroll directly.')
    return cell.unroll(length=length, inputs=inputs, begin_state=begin_state,
                       input_prefix=input_prefix, layout=layout)
