from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Sequence, Text

A = TypeVar('A')
B = TypeVar('B')

@dataclass
class Maybe(Generic[A]):
    pass

@dataclass
class Just(Maybe[A]):
    value: A

@dataclass
class Nothing(Maybe[A]):
    pass
