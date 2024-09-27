# Copyright 2024 Nicolai Sommervold

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the “Software”), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import enum
from typing import TypeVar, Generic

import restruct


@restruct.dataformat(a=restruct.IntegerField(1))
class NestedStruct(restruct.Struct):
    a: int


class _TestIntEnum(enum.IntEnum):
    A = 1
    B = 2


class _TestStrEnum(enum.StrEnum):
    A = "abcd"
    B = "1234"


def field_choice(dependency: int) -> restruct.Field[int]:
    if dependency == 5:
        return restruct.IntegerField(1)
    raise ValueError("dependency must be 5")


_Struct = TypeVar("_Struct", bound=restruct.Struct)


@restruct.dataformat(
    array=restruct.ArrayField(None, restruct.IntegerField(1), term=None)
)
class _DynamicLengthArrayStruct(restruct.Struct):
    array: list[int]


@restruct.dataformat(
    a=restruct.IntegerField(2),
    b=restruct.IntegerField(2),
    c=restruct.StringField(13),
    d=restruct.FloatField(),
    e=restruct.ArrayField(2, restruct.IntegerField(2)),
    f=restruct.StructField(NestedStruct),
    g=restruct.UnusedField(3),
    h=restruct.IntegerBitField(3),
    i=restruct.IntegerBitField(4),
    j=restruct.BooleanBitField(),
    k=restruct.DoubleField(),
    l=restruct.IntEnumField(1, _TestIntEnum),
    m=restruct.IntEnumBitField(5, _TestIntEnum),
    n=restruct.UnusedBitField(3),
    o=restruct.StrEnumField(4, _TestStrEnum),
    p=restruct.IntegerField(2),
    q=restruct.VariableField(field_choice, 1, dependency="p"),
    r=restruct.TerminatedStringField(),
)
class _TestStruct(restruct.Struct, Generic[_Struct]):
    a: int
    b: int
    c: str
    d: float
    e: list[int]
    f: NestedStruct
    g: None
    h: int
    i: int
    j: bool
    k: float
    l: _TestIntEnum
    m: _TestIntEnum
    n: None
    o: _TestStrEnum
    p: int
    q: _Struct | int
    r: str


def test_pack_struct():
    # the float is chosen to be able to be represented exactly
    data = _TestStruct(
        1,
        2,
        "Hello, world!",
        5.25,
        [1, 2],
        NestedStruct(1),
        None,
        2,
        4,
        True,
        10.5,
        _TestIntEnum.A,
        _TestIntEnum.B,
        None,
        _TestStrEnum.A,
        5,
        2,
        "Hello, world",
    ).pack()
    assert data == (
        b"\x00\x01\x00\x02Hello, world!\x00\x00"
        b"\xa8@\x00\x01\x00\x02\x01\x00\x00\x00\x49"
        b"\x00\x00\x00\x00\x00\x00%@"
        b"\x01\x10abcd"
        b"\x00\x05\x02"
        b"Hello, world\x00"
    )


def test_unpack_struct():
    data = (
        b"\x10\x01\x99\x44Hello, world!\x00\x00"
        b"\xa8@\x00\x01\x00\x02\x01\x00\x00\x00\x49"
        b"\x00\x00\x00\x00\x00\x00%@"
        b"\x01\x10abcd"
        b"\x00\x05\x02"
        b"Hello, world\x00"
    )
    struct: _TestStruct[NestedStruct] = _TestStruct.unpack(data)
    assert struct.a == 4097
    assert struct.b == 39236
    assert struct.c == "Hello, world!"
    assert struct.d == 5.25
    assert struct.e == [1, 2]
    assert struct.f == NestedStruct(1)
    assert struct.g == None
    assert struct.h == 2
    assert struct.i == 4
    assert struct.j == True
    assert struct.k == 10.5
    assert isinstance(struct.l, _TestIntEnum)
    assert struct.l == _TestIntEnum.A
    assert isinstance(struct.m, _TestIntEnum)
    assert struct.m == _TestIntEnum.B
    assert isinstance(struct.o, _TestStrEnum)
    assert struct.o == _TestStrEnum.A
    assert struct.p == 5
    assert isinstance(struct.q, int)
    assert struct.q == 2
    assert struct.r == "Hello, world"


@restruct.dataformat(
    length=restruct.IntegerField(1), payload=restruct.StringField("length")
)
class _DynamicStruct(restruct.Struct):
    length: int
    payload: str


def test_dynamic_length_unpack():
    data = b"\x05Hello"
    struct = _DynamicStruct.unpack(data)
    assert struct.length == 5
    assert struct.payload == "Hello"


def test_dynamic_length_pack():
    struct = _DynamicStruct(5, "Hello")
    assert struct.pack() == b"\x05Hello"


def test_dynamic_length_array_struct_pack():
    struct = _DynamicLengthArrayStruct([1, 2, 3, 4])
    assert struct.pack() == b"\x01\x02\x03\x04"


def test_dynamic_length_array_struct_unpack():
    struct = _DynamicLengthArrayStruct.unpack(b"\x01\x02\x03\x04")
    assert struct.array == [1, 2, 3, 4]
