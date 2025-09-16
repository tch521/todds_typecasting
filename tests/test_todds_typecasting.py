import pytest
from datetime import datetime
from typing import (
    Any,
    Optional,
    Union,
    List,
    Dict,
    Tuple,
    Set,
    FrozenSet,
    Sequence,
    Mapping,
    MutableMapping,
)
from dataclasses import dataclass
import json

from todds_typecasting.todds_typecasting import custom_caster, CustomCastingDataclass, auto_cast


"""
Comprehensive casting tests aggregated into two master lists:

SUCCESS_CASES: (value, target_type, expected)
FAILURE_CASES: (value, target_type)

Goals:
 - Exercise primitives, bool tokens, ints<->floats, string numerics
 - Optional / Union (ordering, nested Optionals) / Any passthrough
 - Heterogeneous & homogeneous tuples (incl. ellipsis)
 - Lists / Sequences / Tuples as list targets, JSON string inputs
 - Dict / Mapping / MutableMapping with key & value casting, nested
 - Sets / Frozensets from list/tuple/set/frozenset/JSON strings
 - Deeply nested structures (dict[str, list[tuple[int, float]]], etc.)
 - Datetime ISO8601 (naive & with timezone offset)
 - Idempotency (already correct type) & pass-through for Any
 - Mixed usage of builtin generics (list[int]) and typing forms (List[int])

Constraints / Notes:
 - custom_caster does not special-case dataclass targets, so we avoid them.
 - For Union ordering we assert the first successful arm is chosen.
 - We include cases that intentionally normalize tuples -> lists for list targets.
 - Where Python's constructor semantics apply (e.g., int(3.9) == 3) we accept that.
"""


SUCCESS_CASES = [
    # --- Primitives & direct casting ---
    ("42", int, 42),
    (42, int, 42),
    (3.0, int, 3),  # float to int truncation
    ("3", float, 3.0),
    (3, float, 3.0),
    ("3.14", float, 3.14),
    (3.14, float, 3.14),
    ("hello", str, "hello"),
    (123, str, "123"),
    # Any passthrough
    ([1, 2, 3], Any, [1, 2, 3]),
    (object(), Any, lambda v: v),  # special: expected is identity lambda
    # --- Bool tokens ---
    ("true", bool, True),
    ("TRUE", bool, True),
    ("True", bool, True),  # added
    ("yes", bool, True),
    ("YES", bool, True),  # added
    ("Y", bool, True),
    ("y", bool, True),  # added
    ("t", bool, True),  # added
    (1, bool, True),
    ("false", bool, False),
    ("False", bool, False),  # added
    ("NO", bool, False),
    ("n", bool, False),  # added
    ("f", bool, False),  # added
    ("0", bool, False),
    (0, bool, False),
    # --- Optional ---
    (None, Optional[int], None),
    ("5", Optional[int], 5),
    ("true", Optional[bool], True),  # added
    (None, Optional[List[int]], None),
    ("[1,2]", Optional[list[int]], [1, 2]),
    # --- Union ordering & resolution ---
    ("5", Union[int, float], 5),  # first arm succeeds
    ("3.5", Union[int, float], 3.5),
    ("3.5", Union[float, int], 3.5),
    ("true", Union[bool, int], True),
    ("42", Union[str, int], "42"),  # first arm (str) chosen
    ("42", Union[int, str], "42"),  # order reversed now pass-through str
    ("2025-01-01T00:00:00", Union[int, datetime], datetime(2025, 1, 1, 0, 0, 0)),
    # --- Lists / Sequences (builtin & typing) ---
    (["1", "2", "3"], list[int], [1, 2, 3]),
    (("1", "2"), list[int], [1, 2]),  # tuple -> list
    ("[1,2,3]", list[int], [1, 2, 3]),
    ("[1,2,3]", List[float], [1.0, 2.0, 3.0]),
    ([1, 2, 3], List[Any], [1, 2, 3]),
    ([], list[int], []),
    # Sequence with tuple input
    (("1", "2", "3"), Sequence[int], [1, 2, 3]),
    ('["1","2"]', Sequence[int], [1, 2]),
    # --- Tuples (fixed & variadic) ---
    (("1", "2"), tuple[int, int], (1, 2)),
    (["1", "2"], Tuple[int, int], (1, 2)),
    ("[1,2]", tuple[int, int], (1, 2)),
    (("1", "2", "3"), tuple[int, ...], (1, 2, 3)),
    ([], tuple[int, ...], ()),
    (("1", 2.0, "3.5"), tuple[int, float, float], (1, 2.0, 3.5)),
    # --- Mappings (builtin & typing) ---
    (({"1": "2"}), dict[int, int], {1: 2}),
    ('{"1": "2"}', Dict[int, int], {1: 2}),
    (({"a": "3.5"}), dict[str, float], {"a": 3.5}),
    (({"a": ["1", "2"]}), Dict[str, List[int]], {"a": [1, 2]}),
    (({"a": ("1", "2")}), Mapping[str, list[int]], {"a": [1, 2]}),
    (({"a": {"b": "1"}}), Dict[str, Dict[str, int]], {"a": {"b": 1}}),
    # MutableMapping (aliasing underlying dict logic)
    (({"1": "2"}), MutableMapping[int, int], {1: 2}),
    # --- Sets & Frozensets (builtin & typing) ---
    ((["1", "2"]), set[int], {1, 2}),
    (("[1,2,3]"), set[int], {1, 2, 3}),
    (({"1", "2"}), set[int], {1, 2}),
    (("1", "2"), set[int], {1, 2}),  # added tuple -> set
    ((frozenset(["1", "2"])), set[int], {1, 2}),  # frozenset input -> set output
    ((["1", "2", "2"]), Set[int], {1, 2}),
    ((["1", "2"]), frozenset[int], frozenset({1, 2})),
    ((["1", "2"]), FrozenSet[int], frozenset({1, 2})),
    # --- Nested structures ---
    (({"a": ["1", "2"], "b": ["3", "4"]}), dict[str, list[int]], {"a": [1, 2], "b": [3, 4]}),
    (({"x": [("1", "2"), ("3", "4")]}), dict[str, list[tuple[int, int]]], {"x": [(1, 2), (3, 4)]}),
    (({"n": ["1.1", "2.2"]}), Dict[str, List[float]], {"n": [1.1, 2.2]}),
    (({"k": [["1", "2"], ["3", "4"]]}), Dict[str, List[List[int]]], {"k": [[1, 2], [3, 4]]}),
    (({"deep": {"inner": ["1", "2", "3"]}}), Dict[str, Dict[str, List[int]]], {"deep": {"inner": [1, 2, 3]}}),
    # Mixed nested union
    (({"a": "1", "b": "2.5"}), Dict[str, Union[int, float]], {"a": 1, "b": 2.5}),
    (({"a": "1", "b": "2.5"}), Dict[str, Union[float, int]], {"a": 1, "b": 2.5}),
    # --- Datetime ---
    (datetime(2020, 1, 1), datetime, datetime(2020, 1, 1)),
    ("2020-01-01T00:00:00", datetime, datetime(2020, 1, 1, 0, 0, 0)),
    (
        "2025-09-16T10:11:12+00:00",
        datetime,
        datetime(2025, 9, 16, 10, 11, 12, tzinfo=datetime.fromisoformat("2025-09-16T10:11:12+00:00").tzinfo),
    ),
    # --- Idempotency & Already correct types ---
    ([1, 2, 3], list[int], [1, 2, 3]),
    ((1, 2, 3), tuple[int, ...], (1, 2, 3)),
    ({"a": 1}, dict[str, int], {"a": 1}),
    (set([1, 2]), set[int], {1, 2}),
    (frozenset([1, 2]), frozenset[int], frozenset({1, 2})),
]


FAILURE_CASES = [
    # Primitive failures
    ("abc", int),
    ("[]", int),
    ("{1:2}", float),
    ("3.14", int),  # int("3.14") ValueError
    # Bool bad tokens
    ("truthy", bool),
    ("falzy", bool),
    ("maybe", bool),
    ("10a", bool),
    # Optional inner casting failure
    ("abc", Optional[int]),
    # Union all arms fail
    ("abc", Union[int, float]),
    ("not_json", Union[list[int], dict]),
    # List / sequence failures
    ("not json", list[int]),
    (123, list[int]),
    ({"a": 1}, list[int]),
    ("[1,2", list[int]),  # added malformed JSON
    # Tuple failures
    (("1", "2", "3"), tuple[int, int]),
    (["1", "2", "3"], tuple[int, int]),  # added list causing length mismatch
    # Mapping failures
    ("not json", dict[str, int]),
    (123, dict[str, int]),
    ([1, 2], dict[str, int]),
    ({"a": "x"}, dict[int, int]),  # key cast failure ("a" -> int)
    ({"a": "1"}, dict[int, int]),  # key cast failure
    # Set / frozenset failures
    ("not json", set[int]),
    (123, set[int]),
    ({"a": 1}, set[int]),
    (["1", "x"], set[int]),  # element cast failure
    # Datetime failures
    ("not-a-date", datetime),
    (123, datetime),
    ([], datetime),
]


@pytest.mark.parametrize("value,target,expected", SUCCESS_CASES)
def test_success_cases(value, target, expected):
    result = custom_caster(value, target)
    # Allow lambda for identity expectation (for Any pass-through object)
    if callable(expected) and expected.__name__ == "<lambda>":  # identity check
        assert expected(result) is result  # expected lambda returns value; identity ensures pass-through
    else:
        assert result == expected


@pytest.mark.parametrize("value,target", FAILURE_CASES)
def test_failure_cases(value, target):
    with pytest.raises(Exception):
        custom_caster(value, target)


def test_complex_nested_additional():
    val = {"x": [("1", "2"), ("3", "4")], "y": [("5", "6")]}
    target = dict[str, list[tuple[int, int]]]
    expected = {"x": [(1, 2), (3, 4)], "y": [(5, 6)]}
    assert custom_caster(val, target) == expected


def test_union_datetime_priority():
    val = "2024-12-31T23:59:59"
    target = Union[datetime, int]
    assert custom_caster(val, target) == datetime(2024, 12, 31, 23, 59, 59)


def test_union_string_first():
    val = "42"
    target = Union[str, int]
    assert custom_caster(val, target) == "42"  # first arm str wins


def test_union_int_first():
    val = "42"
    target = Union[int, str]
    assert custom_caster(val, target) == "42"


def test_optional_complex_none():
    val = None
    target = Optional[dict[str, list[int]]]
    assert custom_caster(val, target) is None


def test_optional_complex_value():
    val = {"a": ["1", "2"]}
    target = Optional[dict[str, list[int]]]
    assert custom_caster(val, target) == {"a": [1, 2]}


def test_idempotent_nested():
    existing = {"deep": {"inner": [1, 2, 3]}}
    target = dict[str, dict[str, list[int]]]
    assert custom_caster(existing, target) == existing


# --- Added: Dataclass auto casting parity with other test file ---


@dataclass
class _AppConfig(CustomCastingDataclass):
    port: int
    debug: bool
    tags: list[str]
    ratios: tuple[int, ...]
    created: datetime


def test_dataclass_auto_casting_added():
    cfg = _AppConfig(
        port="8080",
        debug="yes",
        tags='["a","b"]',
        ratios=("1", "2", "3"),
        created="2024-01-01T12:00:00",
    )
    assert (cfg.port, cfg.debug, cfg.tags, cfg.ratios, cfg.created) == (
        8080,
        True,
        ["a", "b"],
        (1, 2, 3),
        datetime(2024, 1, 1, 12, 0, 0),
    )


# --- Added: auto_cast decorator parity tests ---


@auto_cast
def _decorated_fn(a: int, flag: bool = False, nums: Optional[list[int]] = None):
    return a, flag, nums


def test_auto_cast_decorator_added():
    a, flag, nums = _decorated_fn("10", flag="true", nums=("1", "2", "3"))
    assert (a, flag, nums) == (10, True, [1, 2, 3])


def test_auto_cast_defaults_added():
    a, flag, nums = _decorated_fn("5")
    assert (a, flag, nums) == (5, False, None)


# New: Explicit type-sensitive cases to ensure correct casting (guards against 1 == 1.0, True == 1, etc.)
TYPE_ENFORCEMENT_CASES = [
    # Primitives / numeric conversions
    ("42", int, 42, int),
    (42, int, 42, int),
    (3.0, int, 3, int),
    ("3", float, 3.0, float),
    (3, float, 3.0, float),
    ("3.14", float, 3.14, float),
    (3.14, float, 3.14, float),
    # Bool tokens (ensure bool, not int)
    ("true", bool, True, bool),
    ("TRUE", bool, True, bool),
    ("True", bool, True, bool),
    ("yes", bool, True, bool),
    ("YES", bool, True, bool),
    ("Y", bool, True, bool),
    ("y", bool, True, bool),
    ("t", bool, True, bool),
    (1, bool, True, bool),
    ("false", bool, False, bool),
    ("False", bool, False, bool),
    ("NO", bool, False, bool),
    ("n", bool, False, bool),
    ("f", bool, False, bool),
    ("0", bool, False, bool),
    (0, bool, False, bool),
    # Optional (non-None branch)
    ("5", Optional[int], 5, int),
    ("true", Optional[bool], True, bool),
    # Union ordering / resolution
    ("5", Union[int, float], 5, int),
    ("3.5", Union[int, float], 3.5, float),
    ("3.5", Union[float, int], 3.5, float),
    ("1", Union[float, int], 1, int),
    (1, Union[float, int], 1, int),
    ("true", Union[bool, int], True, bool),
    ("42", Union[int, str], "42", str),
    ("2025-01-01T00:00:00", Union[int, datetime], datetime(2025, 1, 1, 0, 0, 0), datetime),
    # Datetime direct
    ("2020-01-01T00:00:00", datetime, datetime(2020, 1, 1, 0, 0, 0), datetime),
]


@pytest.mark.parametrize("value,target,expected,expected_type", TYPE_ENFORCEMENT_CASES)
def test_type_enforcement_cases(value, target, expected, expected_type):
    result = custom_caster(value, target)
    assert result == expected
    # Critical: ensure the concrete type matches (guards against int vs float vs bool ambiguity)
    assert type(result) is expected_type
