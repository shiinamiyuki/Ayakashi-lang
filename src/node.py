from lex import *
import copy
from abc import abstractmethod
from typing import List
from copy import deepcopy

class Node:
    def __init__(self):
        self.tok = nil_token  # type: Token
        self.sub_nodes = []
        self.precedence = 10000  # just for better code generation
        self.parent = None
        self.index = 0
        self.depth = 1

    def add(self, node):
        assert isinstance(node, Node)
        self.sub_nodes.append(node)
        if node.depth > self.depth - 1:
            self.depth = node.depth + 1

    def first(self):
        return self.sub_nodes[0]

    def second(self):
        return self.sub_nodes[1]

    def third(self):
        return self.sub_nodes[2]

    def info(self) -> str:
        return self.type() + '[' + copy.copy(self.tok.tok)

    def to_str(self, lev=0) -> str:
        s = '  ' * lev + self.info() + ']\n'
        for i in self.sub_nodes:
            s += i.to_str(lev + 1)
        return s

    def __repr__(self):
        return self.to_str(0)

    def type(self) -> str:
        return 'Node'

    def __len__(self) -> int:
        return len(self.sub_nodes)

    def copy(self):
        n = copy.copy(self)
        n.tok = self.tok.copy()
        n.depth = self.depth
        n.index = self.index
        n.precedence = self.precedence
        n.sub_nodes = []
        for i in self.sub_nodes:
            n.sub_nodes.append(i.copy())
        n.parent = self.parent
        return n

    def check(self, other):
        assert self is not other
        for i in self.sub_nodes:
            for j in other.sub_nodes:
                assert i is not j

    @abstractmethod
    def accept(self, visitor):
        raise NotImplementedError('You should not attempt to visit an abstract class')

    def reset(self, index: int, node):
        self.sub_nodes[index] = node
        node.index = index
        node.parent = self

    def link(self):
        idx = 0
        for i in self.sub_nodes:
            if i is self:
                raise RuntimeError('Internal Error')
            i.link()
            i.parent = self
            i.index = idx
            idx += 1


class BinaryExpr(Node):
    def __init__(self, tok: Token, p: int):
        super().__init__()
        self.tok = tok
        self.precedence = p

    def type(self):
        return 'BinaryExpr'

    def accept(self, visitor):
        visitor.visit_binary_expr(self)


class UnaryExpr(Node):
    def __init__(self, tok: Token):
        super().__init__()
        self.tok = tok

    def type(self):
        return 'UnaryExpr'

    def accept(self, visitor):
        visitor.visit_unary_expr(self)


class CastExpr(Node):
    def __init__(self):
        super().__init__()

    def type(self):
        return 'CastExpr'

    def accept(self, visitor):
        visitor.visit_cast_expr(self)


class Number(Node):
    def __init__(self, tok: Token):
        super().__init__()
        self.tok = tok

    def type(self):
        return 'Number'

    def accept(self, visitor):
        visitor.visit_number(self)


class Identifier(Node):
    def __init__(self, tok: Token):
        super().__init__()
        self.tok = tok

    def type(self):
        return 'Identifier'

    def accept(self, visitor):
        visitor.visit_identifier(self)

    @staticmethod
    def make_identifier(s):
        return Identifier(Token(s, 'identifier', 0, 0))


class String(Node):
    def __init__(self, tok: Token):
        super().__init__()
        self.tok = tok

    def type(self):
        return 'String'

    def accept(self, visitor):
        visitor.visit_string(self)


class Index(Node):
    def __init__(self):
        super().__init__()

    def type(self):
        return 'Index'

    def accept(self, visitor):
        visitor.visit_index(self)


class Call(Node):
    def __init__(self):
        super().__init__()

    def type(self):
        return 'Call'

    def accept(self, visitor):
        visitor.visit_call(self)


class CallArg(Node):
    def __init__(self):
        super().__init__()

    def type(self):
        return 'CallArg'

    def accept(self, visitor):
        visitor.visit_call_arg(self)


class Block(Node):
    def __init__(self):
        super().__init__()

    def type(self):
        return 'Block'

    def accept(self, visitor):
        visitor.visit_block(self)


class IfStmt(Node):
    def __init__(self):
        super().__init__()

    def type(self):
        return 'If'

    def accept(self, visitor):
        visitor.visit_if_stmt(self)


class WhileStmt(Node):
    def __init__(self):
        super().__init__()

    def type(self):
        return 'While'

    def accept(self, visitor):
        visitor.visit_while_stmt(self)


class Return(Node):
    def __init__(self):
        super().__init__()

    def type(self):
        return 'Return'

    def accept(self, visitor):
        visitor.visit_return(self)


class FuncDef(Node):
    def __init__(self, tok: Token):
        super().__init__()
        self.tok = tok

    def type(self):
        return 'FuncDef'

    def accept(self, visitor):
        visitor.visit_func_def(self)

    def call_signature(self):
        ret = self.second()
        arg = FuncTypeArg()
        for i in self.first().sub_nodes:
            arg.add(i.first())
        f = FuncType()
        f.add(arg)
        f.add(ret)
        return f


class FuncDefArg(Node):
    def __init__(self):
        super().__init__()

    def type(self):
        return 'FuncDefArg'

    def accept(self, visitor):
        visitor.visit_func_def_arg(self)


class Declaration(Node):
    def __init__(self, tok: Token):
        super().__init__()
        self.tok = tok

    def type(self):
        return 'Declaration'

    def accept(self, visitor):
        visitor.visit_declaration(self)


class Struct(Node):
    def __init__(self, tok: Token):
        super().__init__()
        self.tok = tok

    def type(self):
        return 'Struct'

    def accept(self, visitor):
        visitor.visit_struct(self)


class Type(Node):
    def __init__(self):
        super().__init__()

    def type(self):
        return 'Type'

    @abstractmethod
    def accept(self, visitor):
        pass

    @abstractmethod
    def signature(self) -> str:
        pass

    def is_int(self):
        return False

    def is_float(self):
        return False

    def is_double(selfs):
        return False

    def is_array(self):
        return False

    def is_pointer(self):
        return False

    def is_reference(self):
        return False

    def is_arithmetic(self):
        return False

    def is_real(self):
        return False

    def __eq__(self, other):
        return self.signature() == other.signature()

    def is_structure(self):
        return False

    def is_primitive(self):
        return False


class PrimitiveType(Type):
    def __init__(self, tok: Token, is_struct=False):
        super().__init__()
        self.tok = tok
        self.is_struct = is_struct

    @staticmethod
    def make_primitive(ty: str):
        return PrimitiveType(Token(ty, '', -1, -1))

    def is_structure(self):
        return self.is_struct

    def is_primitive(self):
        return True

    def type(self):
        return 'PrimitiveType'

    def accept(self, visitor):
        visitor.visit_primitive_type(self)

    def signature(self):
        return self.tok.tok

    def is_int(self):
        return self.tok.tok in ('int', 'i32', 'u32', 'i8', 'u8', 'i64', 'u64')

    def is_float(self):
        return self.tok.tok in ('float', 'f32')

    def is_double(self):
        return self.tok.tok in ('double', 'f64')

    def is_arithmetic(self):
        return self.is_int() or self.is_float() or self.is_double()

    def is_real(self):
        return self.is_double() or self.is_float()

    def __eq__0(self, other):
        if self.signature() == other.signature():
            return True
        ty1 = self.signature()
        ty2 = self.signature()
        if ty1.is_int() and ty2.is_int():
            pass


class ArrayType(Type):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def info(self) -> str:
        return super().info() + (str(self.size) if self.size >= 0 else 'undefined')

    def type(self):
        return 'ArrayType'

    def accept(self, visitor):
        visitor.visit_array_type(self)

    def signature(self):
        return 'a{0}{1}'.format(self.first().signature(),
                                self.size if self.size >= 0 else 'u')

    def is_array(self):
        return True


class PointerType(Type):
    def __init__(self):
        super().__init__()

    def type(self):
        return 'PtrType'

    def accept(self, visitor):
        visitor.visit_ptr_type(self)

    def signature(self):
        return 'p{0}'.format(self.first().signature())

    def is_pointer(self):
        return True


class RefType(Type):
    def __init__(self):
        super().__init__()

    def type(self):
        return 'RefType'

    def accept(self, visitor):
        visitor.visit_ref_type(self)

    def signature(self):
        return 'r{0}'.format(self.first().signature())

    def is_reference(self):
        return True


class FuncType(Type):
    def __init__(self):
        super().__init__()

    def type(self):
        return 'FuncType'

    def accept(self, visitor):
        visitor.visit_func_type(self)

    def signature(self):
        return 'f{0}{1}'.format(self.first().signature(), self.second().signature())


class FuncTypeArg(Type):
    def __init__(self):
        super().__init__()

    def type(self):
        return 'FuncTypeArg'

    def accept(self, visitor):
        visitor.visit_func_type_arg(self)

    def signature(self) -> str:
        s = 'A'
        for i in self.sub_nodes:
            s += i.signature() + 'c'
        s += 'Z'
        return s


class Chunk(Node):
    def __init__(self):
        super().__init__()

    def type(self):
        return 'Chunk'

    def accept(self, visitor):
        visitor.visit_chunk(self)


class Import(Node):
    def __init__(self):
        super().__init__()

    def type(self):
        return 'Import'

    def accept(self, visitor):
        visitor.visit_import(self)


class CHeader(Node):
    def __init__(self, tok: Token):
        super().__init__()
        self.tok = tok

    def type(self):
        return 'CHeader'

    def accept(self, visitor):
        visitor.visit_c_header(self)


class CDefinition(Node):
    def __init__(self, tok: Token):
        super().__init__()
        self.tok = tok

    def type(self):
        return 'CDefinition'

    def accept(self, visitor):
        visitor.visit_c_definition(self)


class CType(Node):
    def __init__(self, tok: Token):
        super().__init__()
        self.tok = tok

    def type(self):
        return 'CType'

    def accept(self, visitor):
        visitor.visit_c_type(self)


class Implementation(Node):
    def __init__(self, tok: Token):
        super().__init__()
        self.tok = tok

    def type(self):
        return 'Implementation'

    def accept(self, visitor):
        visitor.visit_implementation(self)


class MethodDef(Node):
    def __init__(self, f: FuncDef):
        super().__init__()
        self.sub_nodes = f.sub_nodes
        self.tok = f.tok
        self.precedence = f.precedence
        self.class_name = ''

    def type(self):
        return 'MethodDef'

    def accept(self, visitor):
        visitor.visit_method_def(self)

    def call_signature(self):
        ret = self.second()
        arg = FuncTypeArg()
        for i in self.first().sub_nodes:
            arg.add(i.first())
        f = FuncType()
        f.add(arg)
        f.add(ret)
        return f


class Interface(Node):
    def __init__(self, tok: Token):
        super().__init__()
        self.tok = tok

    def type(self):
        return 'Interface'

    def accept(self, visitor):
        visitor.visit_interface(self)


class ImplFor(Node):
    def __init__(self, interface: Token, tok: Token):
        super().__init__()
        self.tok = tok
        self.interface = interface

    def type(self):
        return 'ImplFor'

    def accept(self, visitor):
        visitor.visit_impl_for(self)

    def copy(self):
        n = super().copy()
        n.interface = deepcopy(self.interface)
        return n


class Generic(Node):
    def __init__(self, t: List[Type]):
        super().__init__()
        self.type_list = t

    def type(self):
        return 'Generic'

    def accept(self, visitor):
        visitor.visit_generic(self)

    def info(self) -> str:
        s = self.type() + '['
        for i in self.type_list:
            s += i.signature() + ','
        return s

    def real_type_list(self) -> List[Type]:
        return self.type_list

    def copy(self):
        n = super().copy()
        n.type_list = []
        for i in self.type_list:
            n.type_list.append(i.copy())
        return n

    def signature(self)->str:
        s = self.first().signature() + '_GG_'
        for i in self.type_list:
            s += i.signature() + '_GG_'
        return s

    def is_array(self):
        return False

    def is_pointer(self):
        return False

    def is_reference(self):
        return False

    def is_arithmetic(self):
        return False

    def __eq__(self, other):
        return self.signature() == other.signature()

class TypeInference(Node):
    def __init__(self):
        super().__init__()

    def type(self):
        return 'TypeInference'

    def accept(self, visitor):
        visitor.visit_type_inference(self)
