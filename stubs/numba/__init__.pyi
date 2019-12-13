from typing import Any, Callable, TypeVar, Mapping, Text, Sequence, Tuple

T = TypeVar('T', bound=type)

FuncType = Callable[..., Any]  # type: ignore
F = TypeVar('F', bound=FuncType)

class int64: ...

def jit(nopython: bool = False) -> Callable[[F], F]: ...

def jitclass(spec: Sequence[Tuple[Text, type]]) -> Callable[[T], T]: ...
