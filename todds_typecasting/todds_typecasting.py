"""
Type-driven casting utilities for arguments, mappings, sequences, sets, tuples,
and dataclass fields.

This module provides:
    1. custom_caster: Safely cast arbitrary runtime values into annotated Python
       types (including Union, Optional, containers, datetime, and primitives).
    2. CustomCastingDataclass: A dataclass mixin that auto-casts fields in
       __post_init__ based on type annotations.
    3. auto_cast: A decorator that casts function arguments to their annotated
       types before execution.

The implementation is intentionally permissive for container protocols
(e.g., allowing list input for Sequence) while still validating structure.

Supported Features
------------------
- Optional[T] / Union[...] resolution
- Homogeneous & heterogeneous tuples (Tuple[T, ...] and Tuple[T1, T2, ...])
- Sequence / Mapping / MutableMapping / Set ABCs
- Builtins: list, tuple, dict, set, frozenset
- Recursive casting of nested containers
- Datetime ISO8601 parsing
- Smart bool parsing from common string tokens

Examples
--------
Basic casting:
>>> custom_caster("true", bool)
True
>>> custom_caster(["1", "2", "3"], list[int])
[1, 2, 3]

Optional:
>>> from typing import Optional
>>> custom_caster(None, Optional[int]) is None
True

Dataclass usage:
>>> @dataclass
... class Config(CustomCastingDataclass):
...     port: int
...     flags: list[bool]
>>> Config(port="8080", flags=["true", "0"])
Config(port=8080, flags=[True, False])

Decorator usage:
>>> @auto_cast
... def run(port: int, debug: bool = False):
...     return port, debug
>>> run("5000", debug="yes")
(5000, True)
"""

import inspect
import json
from dataclasses import dataclass, fields
from datetime import datetime
from functools import wraps
from typing import Any, get_origin, get_args, Union, get_type_hints
from collections.abc import (
    Mapping as AbcMapping,
    MutableMapping as AbcMutableMapping,
    Sequence as AbcSequence,
    Set as AbcSet,
)

# ---------- Casting helpers ----------

_CONTAINER_BUILTINS = (list, tuple, dict, set, frozenset)
_ABCS = (AbcSequence, AbcMapping, AbcMutableMapping, AbcSet)


def _origin(tp):
    """
    Extended get_origin.

    Returns the concrete builtin or ABC for unparameterized container
    annotations so downstream logic can treat them uniformly.

    Parameters
    ----------
    tp : Any
        A typing annotation or runtime type.

    Returns
    -------
    type | None
        The origin type if recognized; otherwise None.
    """
    o = get_origin(tp)
    if o is not None:
        return o
    # Plain builtins (e.g., annotation is just `list`)
    if tp in _CONTAINER_BUILTINS:
        return tp
    # ABCs (e.g., annotation is just `collections.abc.Mapping`)
    if isinstance(tp, type) and issubclass(tp, _ABCS):
        return tp
    return None


def _targs(tp):
    """Safe get_args with empty-tuple fallback."""
    try:
        return tuple(get_args(tp)) or ()
    except Exception:
        return ()


def _is_optional(tp) -> bool:
    """Check if the type is Optional."""
    return _origin(tp) is Union and type(None) in _targs(tp)


def _unwrap_optional(tp):
    """Extract the underlying type from an Optional type."""
    if _is_optional(tp):
        non_none = tuple(t for t in _targs(tp) if t is not type(None))
        return non_none[0] if len(non_none) == 1 else Union[non_none]
    return tp


def _isinstance_typing(val, tp) -> bool:
    """
    Lightweight structural instance check respecting typing constructs.

    Notes
    -----
    This does NOT guarantee element type correctness for containers;
    it only asserts that the outer shape is compatible so we can
    defer element conversion to custom_caster.
    """
    if tp is Any:
        return True

    o = _origin(tp)

    if o is Union:
        return any(_isinstance_typing(val, t) for t in _targs(tp))

    # For parameterized generics, we purposely return False so that
    # custom_caster is invoked to recursively cast inner elements.
    # Only when no type arguments are supplied do we allow a structural True.

    # Sequence (exclude str/bytes to avoid accidental iteration over chars)
    if o in (list, AbcSequence):
        if isinstance(val, (str, bytes, bytearray)):
            return False
        if not isinstance(val, (list, tuple)):
            return False
        return len(_targs(tp)) == 0  # unparameterized sequence counts as instance; parameterized forces cast

    # Tuple shape validation (variable-length ellipsis or fixed-length)
    if o is tuple:
        if not isinstance(val, tuple):
            return False
        args = _targs(tp)
        if len(args) == 0:
            return True  # unparameterized tuple => any tuple ok
        # If parameterized, force casting regardless of shape correctness by returning False
        # but still allow early rejection if gross length mismatch for fixed tuple
        if len(args) == 2 and args[1] is Ellipsis:
            return False  # cause casting path which will process each element
        # If fixed-length but length mismatched, fail fast
        if len(args) != len(val):
            return False
        return False  # lengths match but we want casting for element conversion

    # Mapping types
    if o in (dict, AbcMapping, AbcMutableMapping):
        if not isinstance(val, dict):
            return False
        return len(_targs(tp)) == 0  # parameterized dict should be cast

    # Set / frozenset
    if o in (set, frozenset, AbcSet):
        if not isinstance(val, (set, frozenset)):
            return False
        return len(_targs(tp)) == 0  # parameterized set should be cast

    # Fallback direct isinstance for concrete classes
    return isinstance(val, tp) if isinstance(tp, type) else True


def custom_caster(val, target_type):
    """
    Recursively cast a runtime value to the specified target typing annotation.

    Parameters
    ----------
    val : Any
        Value to cast.
    target_type : Any
        A typing annotation (e.g., list[int], dict[str, float], Union[int, str],
        Optional[datetime], tuple[int, str], set[bool], etc.).

    Returns
    -------
    Any
        The value converted to the desired type (best effort).

    Raises
    ------
    TypeError
        If the value cannot structurally match the requested type.
    ValueError
        For semantic mismatches (e.g., tuple length mismatch, invalid bool token).

    Notes
    -----
    - Container elements are cast recursively.
    - Strings representing JSON arrays/objects are parsed for list/dict/set targets.
    - Bool casting accepts: true/false, 1/0, t/f, y/n, yes/no (case-insensitive).
    - For Union types, the first successful arm is returned.

    Examples
    --------
    >>> custom_caster("42", int)
    42
    >>> custom_caster("[1, 2, 3]", list[int])
    [1, 2, 3]
    >>> from typing import Union
    >>> custom_caster("3.14", Union[int, float])
    3.14
    """
    # Fast path: Any or None in Optional
    if target_type is Any:
        return val
    if _is_optional(target_type) and val is None:
        return None

    tp = _unwrap_optional(target_type)
    o = _origin(tp)

    # Union enhanced resolution strategy:
    # 1. If the runtime value already exactly matches an arm's concrete type, return it unchanged (pass-through).
    # 2. If value is a string and no arm is str, attempt json.loads(value); if parsed value's exact type matches an arm, return parsed.
    # 3. Otherwise, attempt casting in declared order returning on first success.
    if o is Union:
        arms = _targs(tp)

        # Step 1: existing exact-type match for concrete, non-parameterized arms.
        for arm in arms:
            if isinstance(arm, type) and type(val) is arm:
                return val

        # Step 2: string -> json.loads heuristic (only if "str" is not among union arms)
        if isinstance(val, str) and not any((a is str) for a in arms):
            try:
                parsed = json.loads(val)
            except Exception:
                parsed = None
            if parsed is not None:
                for arm in arms:
                    # Only accept direct, non-parameterized matches so we don't skip inner element casting.
                    if isinstance(arm, type) and type(parsed) is arm:
                        return parsed

        # Step 3: ordered attempt (original semantics)
        last_err = None
        for arm in arms:
            try:
                return custom_caster(val, arm)
            except Exception as e:
                last_err = e
        raise last_err or TypeError(f"Cannot cast {val!r} to {tp}")

    # Sequence (list / Sequence ABC)
    # - Accept tuple, but normalize to list
    # - Reject str to avoid accidental char splitting
    if o in (list, AbcSequence):
        (elem_t,) = _targs(tp) or (Any,)
        if isinstance(val, str):
            val = json.loads(val)
        if isinstance(val, tuple):
            val = list(val)
        if not isinstance(val, list):
            raise TypeError(f"Expected list/sequence, got {type(val).__name__}")
        return [custom_caster(v, elem_t) for v in val]

    # Tuple (fixed-length or homogeneous with Ellipsis)
    if o is tuple:
        args = _targs(tp)
        if isinstance(val, str):
            val = json.loads(val)
        if not isinstance(val, (list, tuple)):
            raise TypeError(f"Expected tuple/list, got {type(val).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            elem_t = args[0]
            return tuple(custom_caster(v, elem_t) for v in val)
        if args and len(args) != len(val):
            raise ValueError(f"Tuple length mismatch: expected {len(args)}, got {len(val)}")
        elem_ts = args or (Any,) * len(val)
        return tuple(custom_caster(v, t) for v, t in zip(val, elem_ts))

    # Mapping (dict / Mapping / MutableMapping)
    # - Keys and values both cast recursively
    if o in (dict, AbcMapping, AbcMutableMapping):
        k_t, v_t = _targs(tp) or (Any, Any)
        if isinstance(val, str):
            val = json.loads(val)
        if not isinstance(val, dict):
            raise TypeError(f"Expected dict/mapping, got {type(val).__name__}")
        return {custom_caster(k, k_t): custom_caster(v, v_t) for k, v in val.items()}

    # Sets (set / frozenset / Set ABC)
    # - Parse from JSON array if given a string
    # - Preserve chosen concrete type
    if o in (set, frozenset, AbcSet):
        (elem_t,) = _targs(tp) or (Any,)
        if isinstance(val, str):
            val = json.loads(val)  # expect JSON array
        if not isinstance(val, (set, frozenset, list, tuple)):
            raise TypeError(f"Expected set-like, got {type(val).__name__}")
        seq = list(val) if not isinstance(val, (set, frozenset)) else list(val)
        casted = [custom_caster(v, elem_t) for v in seq]
        return set(casted) if o is set or isinstance(val, set) else frozenset(casted)

    # Primitive / special terminals
    # - datetime from ISO 8601
    # - dict from JSON if needed
    # - robust bool token parsing
    # - fallback to constructor for other concrete classes
    if tp is datetime:
        return val if isinstance(val, datetime) else datetime.fromisoformat(val)
    if tp is dict:
        return val if isinstance(val, dict) else json.loads(val)
    if tp is bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            s = val.strip().lower()
            if s in {"true", "1", "t", "y", "yes"}:
                return True
            if s in {"false", "0", "f", "n", "no"}:
                return False
            raise ValueError(f"Cannot cast string {val!r} to bool")
        return bool(val)
    if isinstance(tp, type):
        return tp(val)
    return val


# ---------- Dataclass auto-casting ----------


@dataclass
class CustomCastingDataclass:
    """
    Dataclass mixin that auto-casts fields to their annotated types.

    On instantiation, each field is examined; if its current value does not
    satisfy the annotated type (structurally for containers), it is converted
    via custom_caster.

    Notes
    -----
    - Uses get_type_hints to resolve postponed annotations and Optional/Union.
    - Skips casting when a value already matches structurally (performance).
    - Nested containers are handled recursively.

    Examples
    --------
    >>> @dataclass
    ... class AppConfig(CustomCastingDataclass):
    ...     host: str
    ...     port: int
    ...     tags: list[str]
    >>> AppConfig(host=123, port="8080", tags='["a","b"]')
    AppConfig(host='123', port=8080, tags=['a', 'b'])
    """

    def __post_init__(self):
        """
        On post-init, cast each field to its annotated type using custom_caster.

        Parameters
        ----------
        self : CustomCastingDataclass
            The dataclass instance being initialized.

        Notes
        -----
        - Uses typing.get_type_hints to robustly handle Optional/Union and
          postponed annotations.
        - Only attempts casting if structural type check fails.
        """
        type_hints = get_type_hints(self.__class__)
        for f in fields(self):
            target = type_hints.get(f.name, Any)
            current = getattr(self, f.name)
            # Skip if already correct type
            if not _isinstance_typing(current, target):
                casted = custom_caster(current, target)
                setattr(self, f.name, casted)


# ---------- Function decorator (args + kwargs) ----------


def auto_cast(func):
    """
    Decorator that auto-casts function arguments to their annotated types.

    Parameters
    ----------
    func : Callable
        Target function with standard Python type annotations.

    Returns
    -------
    Callable
        Wrapped function that performs pre-execution argument casting.

    Notes
    -----
    - Default parameter values are included (bind + apply_defaults).
    - Only casts when the current value fails a structural check.
    - Useful for CLI / JSON / environment variable ingestion.

    Examples
    --------
    >>> @auto_cast
    ... def greet(times: int, excited: bool = False):
    ...     return times, excited
    >>> greet("5", excited="yes")
    (5, True)
    """
    sig = inspect.signature(func)
    type_hints = {
        name: param.annotation for name, param in sig.parameters.items() if param.annotation is not inspect._empty
    }

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Bind positional & keyword arguments respecting signature defaults
        bound = sig.bind_partial(*args, **kwargs)  # captures both args & kwargs
        bound.apply_defaults()  # include defaults so everything is present

        for name, value in list(bound.arguments.items()):
            expected = type_hints.get(name, Any)
            # Only cast when necessary
            if not _isinstance_typing(value, expected):
                bound.arguments[name] = custom_caster(value, expected)

        # Call original function with casted arguments
        return func(*bound.args, **bound.kwargs)

    return wrapper
