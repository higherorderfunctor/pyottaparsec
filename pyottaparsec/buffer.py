from typing import Optional, Union, Iterator
from dataclasses import dataclass, field, InitVar

InputBuffer = Union[
    bytes,
    bytearray,
    memoryview,
    Iterator[bytes],
    Iterator[bytearray],
    Iterator[memoryview]
]

@dataclass
class Buffer:
    _fp: bytearray = field(init=False)
    fp: InitVar[InputBuffer]

    def __post_init__(self, fp: InputBuffer) -> None:
        if isinstance(fp, (bytes, bytearray, memoryview)):
            self._fp = bytearray(fp)
        elif isinstance(fp, Iterator):
            self._fp = bytearray(next(fp))
        else:
            raise TypeError()


    def unsafe_drop(self, n: int) -> memoryview:
        if n < 0 or n > len(self._fp):
            raise IndexError
        return memoryview(self._fp)[n:]
