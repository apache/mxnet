"""Matchers for target assignment.
Matchers are commonly used in object-detection for anchor-groundtruth matching.
The matching process is a prerequisite to training target assignment.
Matching is usually not required during testing.
"""
from mxnet import gluon
from .registry import register, alias, create


@register
class CompositeMatcher(gluon.HybridBlock):
    """A Matcher that combines multiple strategies.

    Parameters
    ----------
    matchers : list of Matcher
        Matcher is a Block/HybridBlock used to match two groups of boxes
    """
    def __init__(self, matchers):
        super(CompositeMatcher, self).__init__()
        assert len(matchers) > 0, "At least one matcher required."
        for matcher in matchers:
            assert isinstance(matcher, (gluon.Block, gluon.HybridBlock))
        self._matchers = matchers

    def hybrid_forward(self, F, x, *args, **kwargs):
        matches = [matcher(x) for matcher in self._matchers]
        return self._compose_matches(F, matches)

    def _compose_matches(self, F, matches):
        """Given multiple match results, compose the final match results.
        The order of matches matters. Only the unmatched(-1s) in the current
        state will be substituded with the matching in the rest matches.

        Parameters
        ----------
        matches : list of NDArrays
            N match results, each is an output of a different Matcher

        Returns
        -------
         one match results as (B, N, M) NDArray
        """
        result = matches[0]
        for match in matches[1:]:
            result = F.where(result > -0.5, result, match)
        return result


@register
class BipartiteMatcher(gluon.HybridBlock):
    """A Matcher implementing bipartite matching strategy.

    Parameters
    ----------
    threshold : float
        Threshold used to ignore invalid paddings
    is_ascend : bool
        Whether sort matching order in ascending order. Default is False.
    """
    def __init__(self, threshold=1e-12, is_ascend=False):
        super(BipartiteMatcher, self).__init__()
        self._threshold = threshold
        self._is_ascend = is_ascend

    def hybrid_forward(self, F, x, *args, **kwargs):
        match = F.contrib.bipartite_matching(x, threshold=self._threshold,
                                             is_ascend=self._is_ascend)
        return match[0]


@register
class MaximumMatcher(gluon.HybridBlock):
    """A Matcher implementing maximum matching strategy."""
    def __init__(self, threshold):
        super(MaximumMatcher, self).__init__()
        self._threshold = threshold

    def hybrid_forward(self, F, x, *args, **kwargs):
        argmax = F.argmax(x, axis=-1)
        match = F.where(F.pick(x, argmax, axis=-1) > self._threshold, argmax,
                        F.ones_like(argmax) * -1)
        return match
