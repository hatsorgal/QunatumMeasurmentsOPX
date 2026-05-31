import pickle
from collections.abc import Mapping

class _Skip:
    """Internal sentinel for values that must be pruned."""
    pass

_SKIP = _Skip()

_CONTAINER_TYPES = (Mapping, list, tuple, set, frozenset)


def _prune_unpicklable(obj, *, _seen_ids=None, _verbose=False):
    """Recursively copy *obj*, deleting any element that cannot be pickled."""
    if _seen_ids is None:
        _seen_ids = set()

    # ── Fast path: atomic / non-container objects ─────────────────────────────
    if not isinstance(obj, _CONTAINER_TYPES):
        try:
            pickle.dumps(obj)
            return obj
        except Exception:
            return _SKIP

    # ── Containers (dict, list, tuple, set, frozenset) ───────────────────────
    oid = id(obj)
    if oid in _seen_ids:           # genuine cycle detection
        return _SKIP
    _seen_ids.add(oid)

    # Dictionaries ------------------------------------------------------------
    if isinstance(obj, Mapping):
        cleaned = {}
        for k, v in obj.items():
            # Key must be picklable
            try:
                pickle.dumps(k)
            except Exception:
                if _verbose:
                    print(f"✘  removing key {k!r}  (key itself unpicklable)")
                continue

            v_clean = _prune_unpicklable(v, _seen_ids=_seen_ids, _verbose=_verbose)
            if v_clean is _SKIP:
                if _verbose:
                    print(f"✘  removing key {k!r}  (value unpicklable)")
            else:
                cleaned[k] = v_clean
        return cleaned

    # List / Tuple / Set / Frozenset -----------------------------------------
    container_type = type(obj)
    items = []
    for x in obj:
        x_clean = _prune_unpicklable(x, _seen_ids=_seen_ids, _verbose=_verbose)
        if x_clean is not _SKIP:
            items.append(x_clean)
        elif _verbose:
            # These items live in sequences, so no “key” to print.
            print("✘  dropping unpicklable sequence element")

    if container_type is tuple:
        return tuple(items)
    return container_type(items)   # list, set, or frozenset


def remove_unpicklable(dct, *, verbose=False):
    """
    Return a picklable clone of *dct* (which may contain nested containers),
    omitting anything that breaks pickling.

    Parameters
    ----------
    dct : Mapping
        The original dictionary (possibly nested).
    verbose : bool, optional
        If True, prints the name of every dictionary key that gets pruned.
    """
    if not isinstance(dct, Mapping):
        raise TypeError("Expected a dictionary-like object at the top level.")
    return _prune_unpicklable(dct, _verbose=verbose)
