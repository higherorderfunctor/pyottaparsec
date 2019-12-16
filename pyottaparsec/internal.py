from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Sequence, Text

A = TypeVar('A')
R = TypeVar('R')
S = TypeVar('S')

A_co = TypeVar('A_co', covariant=True)
R_co = TypeVar('R_co', covariant=True)
S_co = TypeVar('S_co', covariant=True)

A_con = TypeVar('A_con', contravariant=True)
R_con = TypeVar('R_con', contravariant=True)
S_con = TypeVar('S_con', contravariant=True)


# 
# A = TypeVar('A')
# B = TypeVar('B')
# C = TypeVar('C')
# 
# 
# # concatMap :: (a -> [b]) -> [a] -> [b]
# # Modified to support currying of 2-tuple
# def concat_map(
#         f: Callable[[A, C], Sequence[Tuple[B, C]]], xs: Sequence[Tuple[A, C]]
# ) -> Sequence[Tuple[B, C]]:
#     return [
#         y
#         for x in xs
#         for y in f(*x)
#     ]
# 
# 
# # newtype Parser a = Parser { parse :: String -> [(a,String)] }

# from typing import NewType, TYPE_CHECKING, Type
# from typing_extensions import Protocol
# 
# Pos = NewType('Pos', int)
# 
# @dataclass
# class Result(Generic[R]):
#     def fmap(self, f: Callable[[R], S]) -> 'Result[S]':
#         pass
# 
# @dataclass
# class Fail(Result[R]):
#     data: memoryview
#     stack: Sequence[Text]
#     msg: Text
# 
#     def fmap(self, f: Callable[[R], S]) -> Result[S]:
#         return Fail(self.data, self.stack, self.msg)
# 
# 
# class PartialT(Protocol[R]):
#     def __call__(self, i: memoryview) -> Result[R]: ...
# 
# @dataclass
# class Partial(Result[R]):
#     run_parser: PartialT[R]
# 
#     def fmap(self, f: Callable[[R], S]) -> Result[S]:
#         def run(i: memoryview) -> Result[S]:
#             return self.run_parser(i).fmap(f)
#         return Partial(run)
# 
# @dataclass
# class Done(Result[R]):
#     data: memoryview
#     result: R
# 
#     def fmap(self, f: Callable[[R], S]) -> Result[S]:
#         return Done(self.data, f(self.result))
# 
# @dataclass
# class _More:
#     pass
# 
# More = Type[_More]
# 
# @dataclass
# class Complete(_More):
#     pass
# 
# @dataclass
# class Incomplete(_More):
#     pass
# 
# class Failure(Protocol[R]):
#     @staticmethod
#     def __call__(
#             data: memoryview, pos: Pos, more: More, stack: Sequence[Text], msg: Text
#     ) -> Result[R]: ...
# 
# class Success(Protocol[A_con, R]):
#     @staticmethod
#     def __call__(
#             data: memoryview, pos: Pos, more: More, res: A_con
#     ) -> Result[R]: ...
# 
# def failK(
#         data: memoryview, pos: Pos, more: More, stack: Sequence[Text], msg: Text
# ) -> Result[A]:
#     return Fail(data[pos:], stack, msg)
# 
# 
# def successK(
#         data: memoryview, pos: Pos, more: More, res: A_con
# ) -> Result[R]:
#     return Done(data[pos:], res)
# 
# 
# @dataclass
# class Parser(Generic[A, R]):
#     def run_parser(
#         self,
#         data: memoryview,
#         pos: Pos,
#         more: More,
#         lose: Failure[R],
#         succ: Success[A, R]
#     ) -> Result[R]: ...
# 
#     @staticmethod
#     def pure(v: Text) -> Parser[A, R]:
#         class ReturnParser(Parser[A, R]):
#             def run_parser(
#                 self,
#                 data: memoryview,
#                 pos: Pos,
#                 more: More,
#                 lose: Failure[R],
#                 succ: Success[A, R]
#             ) -> Result[R]:
#                 return succ(data, pos, more, val)
#         return ReturnParser()
# 
#     ret = pure
# 
#     @staticmethod
#     def fail(msg: Text) -> Parser[A, R]:
#         class FailParser(Parser[A, R]):
#             def run_parser(
#                 self,
#                 data: memoryview,
#                 pos: Pos,
#                 more: More,
#                 lose: Failure[R],
#                 succ: Success[A, R]
#             ) -> Result[R]:
#                 return lose(data, pos, more, [], f'Failed reading: {msg}')
#         return FailParser()
# 
# 
# def parse(parser: Parser[A, R], data: memoryview) -> Result[R]:
#     return parser.run_parser(data, Pos(0), Incomplete, failK, successK)
# 
# def length_at_least(pos: Pos, n: int, data: memoryview) -> bool:
#     return data.nbytes >= n
# 
# @dataclass
# class Maybe(Generic[A]):
#     pass
# 
# @dataclass
# class Just(Maybe[A]):
#     value: A
# 
# @dataclass
# class Nothing(Maybe[A]):
#     pass
# 
# 
# def advance(n):
#     class AdvanceParser(Parser[A, R]):
#         def run_parser(
#             self,
#             data: memoryview,
#             pos: Pos,
#             more: More,
#             lose: Failure[R],
#             succ: Success[A, R]
#         ) -> Result[R]:
#             return succ(data, pos+n, more, [], f'Failed reading: {msg}')
#     return FailParser()
# 
#  
# 
# def satisfy(p: Callable[[int], bool]) -> Parser[int]:
#     return peekWord8.bind(lambda h: (
#         advance(1).and_then(Parser.ret(h))
#         if p(h)
#         else fail('satisfy')
#     ))
# 
# 
# def word8(c: int) -> Parser[int]:
#     return satisfy(lambda x: x == c)

