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

"""Module for packing and unpacking binary data and python objects"""

import abc
import enum
from typing import (
    Any,
    TypeVar,
    Literal,
    Generic,
    Generator,
    Self,
    TypeAlias,
    dataclass_transform,
    Callable,
)
import inspect
import struct
from dataclasses import dataclass
import dataclasses

__all__ = [
    "dataformat",
    "Field",
    "BitField",
    "Struct",
    "IntegerField",
    "StringField",
    "FloatField",
    "DoubleField",
    "ArrayField",
    "StructField",
    "UnusedField",
    "ByteField",
    "IntegerBitField",
    "BooleanBitField",
    "UnusedBitField",
    "IntEnumField",
    "IntEnumBitField",
    "StrEnumField",
    "BooleanField",
    "VariableField",
    "TerminatedStringField",
]

_T = TypeVar("_T")
FieldSize: TypeAlias = int | str
_IntEnum = TypeVar("_IntEnum", bound=enum.IntEnum)
_StrEnum = TypeVar("_StrEnum", bound=enum.StrEnum)


def dataformat(**fields: "Field[Any] | BitField[Any]"):
    """Sets class attributes of the decorated class based on the keyword
    arguments provided. This is useful to circumvent type hinting issues
    and default init argument set by `@dataclass`.

    Example::

        @dataformat(a = IntegerField(2))
        class ExampleStruct(Struct):
            a: int
            b: int
        ExampleStruct.a # IntegerField(2)
        ExampleStruct.b # raises AttributeError
    """

    def wrapper(cls):
        for name, field_type in fields.items():
            setattr(cls, name, field_type)
        return cls

    return wrapper


class Field(abc.ABC, Generic[_T]):
    """byte packing and unpacking specification to be used in a
    :class:`~Struct` class.

    :param size: size of the field in bytes. If an integer is
        provided, the field will have a static size. If a string
        is provided, it is treated as a reference to the value of
        the parent struct's field with that name. This is useful
        for dynamic length fields in a `~Struct`
    :type size: :class:`~FieldSize`
    """

    def __init__(
        self, size: FieldSize | None, term: bytes | None = b"", **dependencies: str
    ):
        # Storing size in the field class is fine for cases where
        # the size is static, however falls short when dynamic field
        # sized come into play.
        # in those cases, the field size can be different for every
        # pack and unpack, so the size cannot be stored statically.
        # Instead, the name of the field the size depends on is stored
        # instead. This does mean that the object has to store
        # information outside its own scope, however I have not found
        # another solution that works in all required use cases and
        # still provide a smooth experience for the end user.
        self.dependencies = dependencies
        self.termination_character = term
        self.size_bytes = None
        if isinstance(size, str):
            # This is a dependency on another field
            self.dependencies["size"] = size
        elif isinstance(size, int):
            # Static size
            self.size_bytes = size
        else:
            if term == b"":
                raise ValueError(
                    "Must provide either field size or termination character"
                )
            self.termination_character = term

    @abc.abstractmethod
    def pack(self, obj: _T, size: int) -> bytes: ...

    @abc.abstractmethod
    def unpack(self, data: bytes) -> _T: ...

    def size(self) -> int:
        """Return size of field in bytes

        :raises ValueError: if the field does not have a static size
        :return: size in bytes
        :rtype: int
        """
        if self.size_bytes is None:
            raise ValueError("Field does not have a static size")
        return self.size_bytes


class BitField(abc.ABC, Generic[_T]):
    """bit packing specification to be used in a :class:`~Struct`

    :param size: size of the field in bits
    :type size: int
    """

    def __init__(self, size: int):
        self.size = size

    @abc.abstractmethod
    def pack(self, obj: _T) -> int: ...

    @abc.abstractmethod
    def unpack(self, data: int) -> _T: ...


class _BitMapField(Field[list[BitField[Any]]]):
    def __init__(self, **bitfields: BitField):
        size = sum(bitfield.size for bitfield in bitfields.values())
        if size % 8 != 0:
            raise ValueError(
                "the sum of the sizes of the bitfields has to be a multiple of 8"
            )
        super().__init__(size // 8)
        self.bitfields = bitfields

    def pack(self, obj: list[Any], size: int) -> bytes:
        result = 0
        for bitfield, value in zip(self.bitfields.values(), obj):
            result <<= bitfield.size
            result |= bitfield.pack(value)
        return int.to_bytes(result, size)

    def unpack(self, data: bytes) -> list[Any]:
        num = int.from_bytes(data)
        result = []
        for bitfield in list(self.bitfields.values())[::-1]:
            mask = (1 << bitfield.size) - 1
            result.insert(0, bitfield.unpack(num & mask))
            num >>= bitfield.size
        return result

    def __iter__(self):
        for name in self.bitfields.keys():
            yield name


@dataclass_transform()
class _StructMeta(type):
    def __new__(cls, *args, **kwargs):
        cls = super().__new__(cls, *args, **kwargs)
        return dataclass(cls)


class Struct(metaclass=_StructMeta):
    """Class that aids with converting between python objects and bytes.

    by creating a dataclass that subclasses from Struct, you can use the
    methods :meth:`~Struct.pack` and :meth:`~Struct.unpack` to
    convert between an instance of that object, to a bytes object.
    The format of the struct is made by setting class variables equal to
    a field representing how that attribute should be handled.

    This can be aided by the use of the :func:`~dataformat` decorator,
    which converts the class to a dataclass, and adds field attributes.

    Example::

        @dataformat(
            a=IntegerField(2),
            b=StringField(5),
        )
        class A(Struct):
            a: int
            b: str

        obj = A(1, "Hello")
        obj.pack() # Returns b"\\x00\\x01Hello"

        obj2 = A.unpack(b"\\x00\\x01Hello")

        assert obj == obj2 # Evaluates to True


    A Struct class also has support for dynamic length fields, which
    can be specified by referencing another field in the struct::

        @dataformat(
            length=IntegerField(1),
            data=StringField("length"),
        )
        class B(Struct):
            length: int
            data: str

        B.unpack(b"\\x05Hello") == B(5, "Hello") # evaluates to True


    If the class type annotations are specified, they will be included
    in the __init__ method. If ommitted, the attributes will still
    be set on unpack, and a value is expected on pack.

    You can override pack and unpack to set and check non-init values
    automatically, like calculating length immediately before packing.
    """

    @classmethod
    def _iter_fields(cls) -> Generator[tuple[str, Field[Any]], None, None]:
        bitfields = {}
        for name, value in vars(cls).items():
            if name.startswith("__"):
                continue  # ignore special attributes

            if (
                inspect.ismethod(value)
                or inspect.isfunction(value)
                or inspect.ismethoddescriptor(value)
            ):
                continue  # ignore methods and functions

            if isinstance(value, BitField):
                bitfields[name] = value
                continue

            if not isinstance(value, Field):
                raise ValueError(
                    f"class attributes must be of type 'Field', not '{type(value)}'"
                )

            if len(bitfields) > 0:
                yield "", _BitMapField(**bitfields)
                bitfields.clear()

            yield name, value

        if len(bitfields) > 0:
            yield "", _BitMapField(**bitfields)

    def pack(self) -> bytes:
        """pack self into bytes

        :return: bytes representation of self
        :rtype: bytes
        """
        result = bytearray()
        for field_name, field_type in self._iter_fields():
            if isinstance(field_type, _BitMapField):
                value = []
                for name in field_type:
                    value.append(getattr(self, name))
            else:
                value = getattr(self, field_name)
            dependencies = {}
            for k, v in field_type.dependencies.items():
                dependencies[k] = getattr(self, v)

            # If the field is a VariableField, replace with the actual
            # field and re-calculate its dependencies.
            if isinstance(field_type, VariableField):
                kwargs = dependencies.copy()
                kwargs.pop("size", None)
                field_type = field_type.get_field(**kwargs)
                dependencies.clear()
                for k, v in field_type.dependencies.items():
                    dependencies[k] = getattr(self, v)

            size = dependencies.pop("size", None)
            if size is None:
                try:
                    size = field_type.size()
                except ValueError:
                    # Field does not have static size, let it be None.
                    # It is not required.
                    pass

            field_bytes = field_type.pack(value, size, **dependencies)
            if len(field_bytes) != size and size is not None:
                raise ValueError(
                    f"Length of packed bytes for field "
                    f"'{type(self).__name__}.{field_name}' is incorrect. got: "
                    f"'{len(field_bytes)}, expected: '{size}'"
                )
            result += field_bytes

        return bytes(result)

    @classmethod
    def unpack(cls, buffer: bytes) -> Self:
        """unpack a bytes object into an instance of `cls`

        :param buffer: bytes object to unpack
        :type buffer: bytes
        :return: `cls` instance representation of bytes
        :rtype: Self
        """
        result = {}
        index = 0

        for field_name, field_type in cls._iter_fields():
            dependencies = {}
            for k, v in field_type.dependencies.items():
                dependencies[k] = result[v]

            # If the field is a VariableField, replace with the actual
            # field and re-calculate its dependencies.
            if isinstance(field_type, VariableField):
                kwargs = dependencies.copy()
                kwargs.pop("size", None)
                field_type = field_type.get_field(**kwargs)
                dependencies.clear()
                for k, v in field_type.dependencies.items():
                    dependencies[k] = result[v]

            size = dependencies.pop("size", None)
            if field_type.termination_character != b"":
                if field_type.termination_character is None:
                    # Set to full buffer
                    size = len(buffer[index:])
                else:
                    size = buffer[index:].index(field_type.termination_character)
                    size += len(field_type.termination_character)
            elif size is None:
                size = field_type.size()

            data = buffer[index : index + size]
            index += size

            if isinstance(field_type, _BitMapField):
                result.update(
                    {
                        x: y
                        for x, y in zip(
                            field_type, field_type.unpack(data, **dependencies)
                        )
                    }
                )
            else:
                result[field_name] = field_type.unpack(data, **dependencies)

        fields = dataclasses.fields(cls)
        init_fields = {}
        for field in fields:
            init_fields[field.name] = result[field.name]
            del result[field.name]

        instance = cls(**init_fields)
        for name, value in result.items():
            setattr(instance, name, value)

        return instance

    @classmethod
    def size(cls) -> int:
        """return size of a static size struct

        :raises ValueError: if the struct does not have a static size
        :return: size of the struct in bytes
        :rtype: int
        """
        return sum(x.size() for _, x in cls._iter_fields())

    def __len__(self) -> int:
        """Return instance length of struct in bytes"""
        return len(self.pack())


class IntegerField(Field[int]):
    """field to represent an `int` object

    :param size: size of the integer in bytes
    :type size: :class:`~FieldSize`
    :param byteorder: byteorder of the int, defaults to "big"
    :type byteorder: Literal["little", "big"], optional
    :param signed: if the int is signed or not, defaults to False
    :type signed: bool, optional
    """

    def __init__(
        self,
        size: FieldSize,
        byteorder: Literal["little", "big"] = "big",
        *,
        signed: bool = False,
    ):
        super().__init__(size)
        # pylance complains when there is no explicit typing here
        self._byteorder: Literal["little", "big"] = byteorder
        self._signed = signed

    def unpack(self, data: bytes) -> int:
        return int.from_bytes(data, self._byteorder)

    def pack(self, data: int, size: int) -> bytes:
        return int.to_bytes(data, size, self._byteorder, signed=self._signed)


class StringField(Field[str]):
    """field to represent a `str` object

    :param length: length of the string in characters
    :type length: :class:`~FieldSize`
    :param encoding: string encoding. Valid inputs are the same
        as encoding in `str.encode`, defaults to "ascii"
    :type encoding: str, optional
    :param errors: scheme to use when treating errors. Valid inputs
        are the same as errors in `str.encode`, defaults to "strict"
    :type errors: str, optional
    """

    def __init__(
        self, length: FieldSize, encoding: str = "ascii", errors: str = "strict"
    ):
        super().__init__(length)
        self.encoding = encoding
        self.errors = errors

    def pack(self, obj: str, _: int) -> bytes:
        return obj.encode(self.encoding, self.errors)

    def unpack(self, data: bytes) -> str:
        return data.decode(self.encoding, self.errors)


class FloatField(Field[float]):
    def __init__(self):
        super().__init__(4)

    def pack(self, obj: float, _: int) -> bytes:
        return struct.pack("f", obj)

    def unpack(self, data: bytes) -> float:
        return struct.unpack("f", data)[0]


class DoubleField(Field[float]):
    def __init__(self):
        super().__init__(8)

    def pack(self, obj: float, _: int) -> bytes:
        return struct.pack("d", obj)

    def unpack(self, data: bytes) -> float:
        return struct.unpack("d", data)[0]


class ArrayField(Field[list[_T]]):
    """field representing a list of items

    :param length: length of the list in number of items. Dynamic length
        reference is allowed. If set to None, it will take the rest of
        the buffer.
    :type length: int
    :param subfield: :class:`~Field` object representing the items in
        the array. The subfield has to be of static length.
    :type type: Field[_T]
    """

    def __init__(
        self, length: FieldSize | None, subfield: Field[_T], term: bytes | None = b""
    ):
        if length is None and term != b"":
            term = None

        super().__init__(length, term=term)
        self.subfield = subfield

    def pack(self, obj: list[_T], _: int) -> bytes:
        result = bytearray()
        for item in obj:
            result += self.subfield.pack(item, self.subfield.size())
        return bytes(result)

    def unpack(self, data: bytes) -> list[_T]:
        size = self.subfield.size()
        result = []
        for index in range(0, len(data), size):
            result.append(self.subfield.unpack(data[index : index + size]))
        return result

    def size(self) -> int:
        length = super().size()
        return length * self.subfield.size()


class StructField(Field[Struct]):
    """field representing a :class:`~Struct` object

    Struct with dynamic length is not supported in StructField

    :param struct: struct for this field
    :type struct: type[Struct]
    """

    def __init__(self, struct: type[Struct], size: str | None = None):
        if size is None:
            super().__init__(struct.size())
        else:
            super().__init__(size)
        self.struct = struct

    def pack(self, obj: Struct, _: int) -> bytes:
        return obj.pack()

    def unpack(self, data: bytes) -> Struct:
        return self.struct.unpack(data)


class UnusedField(Field[None]):
    """field for unused bytes

    :param size: number of unused bytes
    :type size: FieldSize
    """

    def __init__(self, size: FieldSize):
        super().__init__(size)

    def pack(self, obj: None, size: int) -> bytes:
        del obj  # not used
        return bytes(size)

    def unpack(self, data: bytes) -> None:
        del data  # not used
        return None


class ByteField(Field[bytes]):
    """field representing a bytes object"""

    def __init__(self, size: int | str):
        super().__init__(size)

    def pack(self, obj: bytes, _: int) -> bytes:
        return obj

    def unpack(self, data: bytes) -> bytes:
        return data


class IntegerBitField(BitField[int]):
    """Integer bits"""

    def pack(self, obj: int) -> int:
        return obj

    def unpack(self, data: int) -> int:
        return data


class BooleanBitField(BitField[bool]):
    """Boolean bit"""

    def __init__(self):
        super().__init__(1)

    def pack(self, obj: bool) -> int:
        return int(obj)

    def unpack(self, data: int) -> bool:
        return bool(data)


class UnusedBitField(BitField[None]):
    """Unused bits"""

    def __init__(self, size: int):
        super().__init__(size)

    def pack(self, obj: None) -> int:
        del obj
        return 0

    def unpack(self, data: int) -> None:
        del data
        return None


class IntEnumField(IntegerField, Generic[_IntEnum]):
    """"""

    def __init__(
        self,
        size: int | str,
        enum: type[_IntEnum],
        byteorder: Literal["little"] | Literal["big"] = "big",
        *,
        signed: bool = False,
    ):
        self._enum = enum
        super().__init__(size, byteorder, signed=signed)

    def pack(self, data: _IntEnum, size: int) -> bytes:
        return super().pack(data, size)

    def unpack(self, data: bytes) -> _IntEnum:
        value = super().unpack(data)
        return self._enum(value)


class IntEnumBitField(IntegerBitField, Generic[_IntEnum]):
    """"""

    def __init__(self, size: int, enum: type[_IntEnum]):
        self._enum = enum
        super().__init__(size)

    def pack(self, data: _IntEnum) -> int:
        return super().pack(data)

    def unpack(self, data: int) -> _IntEnum:
        value = super().unpack(data)
        return self._enum(value)


class StrEnumField(StringField, Generic[_StrEnum]):
    """"""

    def __init__(
        self,
        size: int | str,
        enum: type[_StrEnum],
        encoding: str = "ascii",
        errors: str = "strict",
    ):
        self._enum = enum
        super().__init__(size, encoding=encoding, errors=errors)

    def pack(self, data: _StrEnum, size: int) -> bytes:
        return super().pack(data, size)

    def unpack(self, data: bytes) -> _StrEnum:
        value = super().unpack(data)
        return self._enum(value)


class BooleanField(IntegerField):
    def __init__(self):
        super().__init__(1, "big", signed=False)

    def pack(self, data: bool, size: int) -> bytes:
        return super().pack(data, size)

    def unpack(self, data: bytes) -> bool:
        return bool(super().unpack(data))


class VariableField(Field[Any]):
    def __init__(
        self,
        field_choice: Callable[..., Field[Any]],
        size: int | str | None,
        **dependencies: str,
    ):
        if size is None:
            term = None
        else:
            term = b""
        super().__init__(size, term, **dependencies)
        self.field_choice = field_choice

    def get_field(self, **dependencies):
        return self.field_choice(**dependencies)

    def pack(self, obj: Any, size: int, **dependencies) -> bytes:
        raise NotImplementedError("VariableField is special and cannot pack")

    def unpack(self, data: bytes, **dependencies) -> Any:
        raise NotImplementedError("VariableField is special and cannot unpack")


class TerminatedStringField(Field[str]):
    def __init__(
        self,
        term_char: bytes | None = b"\x00",
        encoding: str = "ascii",
        errors: str = "strict",
    ):
        super().__init__(None, term=term_char)
        self.encoding = encoding
        self.errors = errors

    def pack(self, obj: str, size: int) -> bytes:
        return obj.encode(self.encoding, self.errors) + b"\x00"

    def unpack(self, data: bytes) -> str:
        return data[:-1].decode(self.encoding, self.errors)
