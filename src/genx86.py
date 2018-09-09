from node import *
from visitor import Visitor
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import parse


class Buffer:
    def __init__(self):
        self.spaces = 0
        self.buffer = ''

    def indent(self):
        assert self.spaces >= 0
        self.spaces += 4

    def dedent(self):
        self.spaces -= 4
        assert self.spaces >= 0

    def write(self, s: str):
        self.buffer += s

    def write_indent(self):
        self.buffer += self.spaces * ' '





class VarInfo:
    def __init__(self, name: str, ty: Type = None):
        self.name = name
        self.type = ty
        self.fields = dict()  # type: Dict[str,VarInfo]
        self.methods = dict()  # type: Dict[str,VarInfo]

    def add_field(self, v):
        self.fields[v.name] = v

    def add_method(self, v):
        self.methods[v.name] = v

    def retrieve(self, v: str):
        if v in self.fields:
            return self.fields[v]
        if v in self.methods:
            return self.methods[v]
        return None

    def retrieve_field(self, v: str):
        if v in self.fields:
            return self.fields[v]
        return None

    def retrieve_method(self, v: str):
        if v in self.methods:
            return self.methods[v]
        return None


class CodeGen(Visitor):
    def __init__(self):
        super().__init__()
        self.produced_source = ''
        self.temp_stack = []
        self.temp = Buffer()
        self.typedefs = ''
        self.types = []  # type :List[str
        self.type = None  # type : Type
        self.type_stack = []  # type:List[Type]
        self.symbol_table = []  # type: List[Dict[str,VarInfo]]
        self.interfaces = dict()  # type: Dict[str,VarInfo]
        self.generics = dict()  # type: Dict[str, Generic]
        self.generic_impls = []  # type: List[Generic]
        self.line = 0
        self.col = 0
        self.showCastWarning = True
        self.filename = '#memory#'
        for i in ['int', 'float', 'void', 'double', 'char']:
            self.add_type(i)

    def add_type(self, sign: str):
        if sign not in self.types:
            self.types.append(sign)

    def has_type(self, sign: str):
        return sign in self.types

    def update_pos(self, tok: Token):
        self.line = tok.line
        self.col = tok.col

    def error(self, msg: str):
        raise RuntimeError('\033[4;33;40m{0}:{1}:{2}\033[0m \033[1;31;40merror: {3}\033[0m'.format(
            self.filename, self.line, self.col, msg
        ))

    def warning(self, msg: str):
        s = ('\033[4;33;40m{0}:{1}:{2}\033[0m \033[1;33;40mwarning: {3}\033[0m'.format(
            self.filename, self.line, self.col, msg
        ))
        print(s)

    def retrieve_generic(self, s: str) -> Optional[Generic]:
        if s in self.generics:
            return (self.generics[s]).copy()
        return None

    def add_generic(self, s: str, n: Node):
        if s not in self.generics:  # crucial
            self.generics[s] = n

    def push_type(self, ty: Type):
        self.type_stack.append(ty)

    def pop_type(self) -> Type:
        self.type = self.type_stack.pop()
        return self.type

    def retrieve_symbol(self, var: str) -> Optional[VarInfo]:
        for i in reversed(self.symbol_table):
            if var in i:
                return i[var]
        return None

    def add_global_symbol(self, var: str, ty: Type):
        self.symbol_table[0][var] = VarInfo(var, ty)

    def add_global_symbol_directly(self, v: VarInfo):
        self.symbol_table[0][v.name] = v

    def add_symbol(self, var: str, ty: Type):
        self.symbol_table[-1][var] = VarInfo(var, ty)

    def add_symbol_directly(self, v: VarInfo):
        self.symbol_table[-1][v.name] = v

    def push_lexical_scope(self):
        self.symbol_table.append(dict())

    def pop_lexical_scope(self):
        if len(self.symbol_table) <= 1:
            raise RuntimeError('Internal Error')
        self.symbol_table.pop()

    def write_typedef(self, s: str):
        self.typedefs += s

    def push_temp(self):
        self.temp_stack.append(self.temp)
        self.temp = Buffer()

    def pop_temp(self):
        try:
            self.temp = self.temp_stack.pop()
        except IndexError:
            self.temp.buffer = ''

    def write_temp(self, s: str):
        self.temp.write(s)

    def write_and_push(self, s: str):
        self.temp.write(s)
        self.push_temp()

    def clear_temp(self):
        self.temp.buffer = ''

    def write_temp_to_source_and_destroy_temp(self):
        self.produced_source += self.temp.buffer
        self.temp = Buffer()

    def write_typedefs_to_source(self):
        self.produced_source += self.typedefs

    def visit_number(self, node: Number):
        self.write_and_push(node.tok.tok)
        s = node.tok.tok
        if '.' in s:
            if 'f' in s:
                self.push_float_type()
            else:
                self.push_double_type()
        else:
            self.push_int_type()

    def push_int_type(self):
        self.push_primitive_type('int')

    def push_float_type(self):
        self.push_primitive_type('float')

    def push_double_type(self):
        self.push_primitive_type('double')

    def push_primitive_type(self, t):
        ty = PrimitiveType(Token(t, None, 0, 0))
        self.push_type(ty)

    def is_equal_type(self, ty1: Type, ty2: Type):
        if ty1 == ty2:
            return True
        if ty1.is_pointer() or ty1.is_reference() or ty1.is_array() or ty1.is_null():
            if ty2.is_pointer() or ty2.is_reference() or ty2.is_array() or ty2.is_null():
                if ty1.is_void_ptr() or ty2.is_void_ptr():
                    return True

        return False

    def check_binary_expr_type_int_only_op(self, op, ty1: Type, ty2: Type):
        if op not in [x for x in '& ^ | %' if x]:
            return False
        if not ty1.is_int() or not ty2.is_int():
            self.error('integer expected')
        self.push_int_type()
        return True

    def check_binary_expr_type_real_auto_promote(self, op, ty1: Type, ty2: Type):
        if ty1.is_arithmetic() and ty2.is_arithmetic():
            if ty1.is_double() or ty2.is_double():
                self.push_double_type()
                return True
            elif ty1.is_float() or ty2.is_float():
                self.push_float_type()
                return True
            else:
                self.push_int_type()
                return True
        if (ty1.is_pointer() or ty2.is_pointer()) and (op not in ['!=', '==', '=']):
            self.error('attempt to perform pointer arithmetic')
        if ty1.is_reference() or ty2.is_reference():
            self.error('operator overloading not implemented yet!')
        return False

    def check_type_field_and_method(self, op, ty1: Type, ty2: Type) -> bool:
        if op not in ('.', '->'):
            return False
        if not ty2.is_primitive():
            self.error('identifier expected after \'.\'')
        struct_name = ty1.signature()
        field_name = ty2.signature()
        struct_type = self.retrieve_symbol(struct_name).type
        if struct_type.is_reference() or struct_type.is_pointer():
            struct_type = struct_type.first()
        if struct_type.is_reference() or struct_type.is_pointer():
            # still a reference
            self.error("don't do that")
        struct = self.retrieve_symbol(struct_type.signature())
        if not struct:
            # (struct_type)
            self.error('www')
        field = struct.retrieve(field_name)
        if not field:
            # print(struct)
            # print(self.type_stack,self.symbol_table)
            self.error('no such field ' + field_name)
        self.push_type(field.type)
        return True

    def check_binary_expr_type(self, op: str):
        ty2 = self.pop_type()
        ty1 = self.pop_type()
        if self.check_type_field_and_method(op, ty1, ty2):
            return
        if self.check_binary_expr_type_int_only_op(op, ty1, ty2):
            return
        if self.check_binary_expr_type_real_auto_promote(op, ty1, ty2):
            return
        if not self.is_equal_type(ty1, ty2):
            self.error('\nincompatible type:\n\t{0}\nand\n\t{1}\n'.format(ty1, ty2))

    def check_index_expr_type(self):
        ty2 = self.pop_type()
        ty1 = self.pop_type()
        if not ty2.is_int():
            self.error('int expected in index expr')
        if not ty1.is_array():
            self.error('array type expected in index expr')
        self.push_type(ty1.first())

    def visit_declaration(self, node):
        ty = node.first()
        ty.accept(self)
        self.push_type(node.first())
        ty = node.first()
        self.pop_temp()
        s = self.temp.buffer + ' '
        var = node.tok.tok
        s += var
        self.add_symbol(var, ty)
        self.clear_temp()
        if len(node) == 2:
            node.second().accept(self)
            self.pop_temp()
            s += ' = ' + self.temp.buffer
            self.clear_temp()
        self.write_and_push(s)

    def visit_identifier(self, node):
        self.write_and_push(node.tok.tok)
        self.update_pos(node.tok)
        if node.tok.tok == 'null':
            self.push_type(PrimitiveType.make_primitive('null'))
        elif node.tok.tok in ['false', 'true']:
            self.push_type(PrimitiveType.make_primitive('int'))
        elif node.tok.tok == 'none':
            self.push_type(PrimitiveType.make_primitive('void'))
            self.pop_temp()
            self.clear_temp()
            self.write_and_push(';')
        else:
            v = self.retrieve_symbol(node.tok.tok)
            if v:
                if v.type.is_reference():
                    self.pop_temp()
                    s = '*(' + self.temp.buffer + ')'
                    self.clear_temp()
                    self.write_and_push(s)
                    self.push_type(self.remove_reference(v.type))
                else:
                    self.push_type(v.type)
            else:
                self.error('unresolvable reference to ' + node.tok.tok)

    def visit_string(self, node):
        self.write_and_push(node.tok.tok)
        t = PrimitiveType(Token('char', '', 0, 0))
        a = ArrayType(-1)
        a.add(t)
        self.push_type(a)

    def remove_reference(self, ty: Type) -> Type:
        if ty.is_reference() or ty.is_pointer():
            ty = ty.first()
        elif ty.is_array():
            self.error("could not dereference array type")
        return ty

    def name_decorate(self, type_name: str, method_name: str) -> str:
        return type_name + '_' + method_name

    def dot_expr(self, node: BinaryExpr) -> bool:
        if node.tok.tok not in ['.', '->']:
            return False
        left = node.first()
        right = node.second()
        if right.type() != 'Identifier':
            return False
        s = ''
        left.accept(self)
        self.pop_temp()
        s += '(' + self.temp.buffer + ')'
        self.clear_temp()
        ty = self.pop_type()
        name = self.remove_reference(ty).signature()
        struct = self.retrieve_symbol(name)
        if not struct:
            self.error('{0} seems not to be a struct'.format(name))
        member = right.tok.tok
        field = struct.retrieve_field(member)
        if field:
            self.push_type(field.type)
            if ty.is_reference() or ty.is_pointer():
                s += '->'
            else:
                s += '.'
            s += member
            self.clear_temp()
            self.write_and_push(s)
            return True
        # is not a field, method?
        method = struct.retrieve_method(member)
        if method:
            method_name = self.name_decorate(name, member)
            self.push_type(method.type)
            s = method_name
            self.clear_temp()
            self.write_and_push(s)
            return True
        else:
            self.error('{0} does not have member {1}'.format(name, member))

    def visit_binary_expr(self, node: BinaryExpr):
        if self.dot_expr(node):
            return
        left = node.first()
        right = node.second()
        right.accept(self)
        left.accept(self)
        self.pop_temp()
        left_out = self.temp.buffer
        self.pop_temp()
        right_out = self.temp.buffer
        self.clear_temp()
        s = ''
        if left.precedence < node.precedence:
            left_out = '(' + left_out + ')'
        if right.precedence < node.precedence:
            right_out = '(' + right_out + ')'
        if node.tok.tok == 'and':
            node.tok.tok = '&&'
        if node.tok.tok == 'or':
            node.tok.tok = '||'
        op = ' ' + node.tok.tok + ' '
        s = left_out + op + right_out
        self.update_pos(node.tok)
        self.clear_temp()
        self.write_and_push(s)
        self.check_binary_expr_type(node.tok.tok)

    def visit_if_stmt(self, node: IfStmt):
        s = 'if('
        node.first().accept(self)
        self.pop_temp()
        s += self.temp.buffer + '){\n'
        self.clear_temp()
        node.second().accept(self)
        self.pop_temp()
        s += self.temp.buffer + '}\n'
        self.clear_temp()
        i = 2
        while i < len(node.sub_nodes):
            if i == len(node.sub_nodes) - 1:
                s += 'else{\n'
                node.sub_nodes[i].accept(self)
                self.pop_temp()
                s += self.temp.buffer + '}\n'
                self.clear_temp()
                break
            else:
                s += 'else if('
                node.sub_nodes[i].accept(self)
                s += self.temp.buffer + '){\n'
                self.clear_temp()
                node.sub_nodes[i + 1].accept(self)
                self.pop_temp()
                s += self.temp.buffer + '}\n'
                self.clear_temp()
                i += 2
        self.clear_temp()
        self.write_and_push(s)

    def visit_index(self, node):
        node.first().accept(self)
        node.second().accept(self)
        self.check_index_expr_type()
        s = ''
        self.pop_temp()
        s = '[' + self.temp.buffer + ']'
        self.pop_temp()
        s = self.temp.buffer + s
        self.clear_temp()
        self.write_and_push(s)

    def visit_unary_expr(self, node: UnaryExpr):
        op = node.tok.tok
        if op == 'not':
            op = '!'
        s = op + '('
        node.first().accept(self)
        self.update_pos(node.tok)
        self.pop_temp()
        s += self.temp.buffer + ')'
        self.clear_temp()
        self.write_and_push(s)
        if op == 'sizeof':
            self.push_type(node.first())
        ty = self.pop_type()

        if op == '&':
            t = RefType()
            t.add(ty)
            self.push_type(t)
        elif op == '*':
            if ty.is_pointer() or ty.is_reference():
                self.push_type(ty.first())
            else:
                self.error("could not dereference: {0}".format(ty))
        elif op == '-':
            if ty.is_arithmetic():
                self.push_type(PrimitiveType.make_primitive('int'))
            else:
                self.error('illegal use of ' + op)
        elif op == '!':
            self.push_type(PrimitiveType.make_primitive('int'))
        elif op == 'sizeof':
            self.push_type(PrimitiveType.make_primitive('int'))

    def method_call(self, node: Call) -> bool:
        method = node.first()
        if len(method) < 2:
            return False
        if not self.dot_expr(method):
            return False
        arg = node.second()
        self.method_call_arg(arg, method.first())
        return True

    def method_call_arg(self, node: CallArg, caller: Node):
        f = self.pop_type()
        arg = f.first().sub_nodes
        self.write_and_push('(')
        caller.accept(self)
        ty = self.pop_type()
        for i in node.sub_nodes:
            i.accept(self)
        s = ')'
        ty_list = []
        for i in node.sub_nodes:
            self.pop_temp()
            if s != ')':
                s = self.temp.buffer + ', ' + s
            else:
                s = self.temp.buffer + s
            ty_list.insert(0, self.pop_type())
        try:
            for i in range(0, len(ty_list)):
                if arg[i + 1].signature() == '...':
                    break
                if not self.is_equal_type(arg[i + 1], ty_list[i]):
                    self.error('\nincompatible type at {2}th argument:\n\t{0}\nand\n\t{1}'.format(arg[i + 1]
                                                                                                  , ty_list[i],
                                                                                                  i))
        except IndexError:
            self.error('{0} arguments expected, {1} provided'.format(len(arg) - 1, len(ty_list)))
        self.push_type(f.second())
        self.pop_temp()
        ref = ''
        if not ty.is_reference() and not ty.is_pointer():
            ref = '&'
        if s != ')':
            s = ref + self.temp.buffer + ', ' + s
        else:
            s = ref + self.temp.buffer + s
        self.pop_temp()
        s = '(' + s
        self.pop_temp()
        s = self.temp.buffer + s
        self.clear_temp()
        self.write_and_push(s)

    def visit_call(self, node: Call):
        """
        Visits call node
        notice:
            for method calls, caller is evaluated first
            for others, arguments are evaluated first because we want to deduct
            corresponding function type when dealing with generics
        :param node:
        :return:
        """
        if self.method_call(node):
            return
        node.second().accept(self)
        # node.first().accept(self)
        self.pop_temp()
        s = self.temp.buffer
        self.clear_temp()
        self.check_arg_and_visit_caller(node.first(), node.second())
        # s = ''
        self.pop_temp()
        s = self.temp.buffer + s
        self.clear_temp()
        self.write_and_push(s)

    def get_arg_type_list_from_stack(self, n: int) -> List[Type]:
        ty = []
        for i in range(0, n):
            ty.append(self.pop_type())
        ty.reverse()
        return ty

    def is_generic(self, var: str):
        return var in self.generics

    def generic_deduction(self, func_name: str, ty: List[Type]):
        g = self.retrieve_generic(func_name)
        type_list = g.real_type_list()
        # print(type_list, ty)
        # return Generic(self.deduct(type_list, ty))
        arg = g.first().first()
        # print(arg)
        a = []
        for i in arg.sub_nodes:
            a.append(i.first())
        # print(a)
        return Generic(self.deduct(a, type_list, ty))
        # return Generic(ty)

    def reduce(self, g: Type, ty: Type) -> List[Tuple[Type, Type]]:
        """
        if g: T, T = ty, returns (T, ty)
        if g: G<T>, is ty is not of G, error
        if g: G<T>, ty = G<U>, T = U returns (T, U)
        :param g: a type from arg
        :param ty: a type from ty
        :return:
        """
        if g.type() == 'PrimitiveType':
            return [(g, ty)]
        if g.type() == 'Generic':
            if ty.type() != 'Generic':
                self.error('\nincompatible generic argument:\n\t{0}\nand\n\t{1}'.format(g, ty))
            a = len(g.real_type_list())
            b = len(ty.real_type_list())
            if a != b:
                self.error('\nincompatible generic argument:\n\t{0}\nand\n\t{1}'.format(g, ty))
            result = []
            for i in range(0, a):
                result.append((g.real_type_list()[i], ty.real_type_list()[i]))
            return result
        elif g.type() == 'RefType' or g.type() == 'ArrayType' or g.type() == 'PointerType':
            if ty.type() != g.type():
                self.error('incompcatible type\n {0}\n {1}'.format(g, ty))
            else:
                return self.reduce(g.first(), ty.first())
        else:
            self.error(g)

    def deduct(self, arg: List[Type], g: List[Type], ty: List[Type]) -> List[Type]:
        m = dict()
        for i in range(0, len(arg)):
            r = self.reduce(arg[i], ty[i])
            for a, b in r:
                if a.signature() in m and m[a.signature()] != b.signature():
                    self.error(('conflicting generic argument {0} with \n\t' +
                                '{1}\nand\n\t{2}').format(a.signature(),
                                                          m[a.signature()],
                                                          b))
                m[a.signature()] = b.signature()
        x = []
        for i in m:
            if self.has_type(i):
                x.append(i)
        for i in x:
            del m[i]
        if len(m) != len(g):
            self.error('{0} arguments expected but found {1} with:\n\t{2}\nand\n\t{3}'.format(
                len(g),
                len(m),
                g, m))
        result = []
        for i in g:
            result.append(PrimitiveType.make_primitive(m[i.signature()]))
        return result

    def check_arg_and_visit_caller(self, caller: Node, arg: CallArg):
        ty = self.get_arg_type_list_from_stack(len(arg))
        if caller.type() == 'Generic':
            caller.accept(self)
            func_type = self.pop_type()
            self.check_arg(func_type, ty)
        elif caller.type() == 'Identifier':
            func_name = caller.tok.tok
            self.update_pos(caller.tok)
            if self.is_generic(func_name):
                # generic = Generic(ty)
                generic = self.generic_deduction(func_name, ty)
                generic.add(caller)
                generic.parent = caller.parent
                generic.accept(self)
                func_type = self.pop_type()
                self.push_type(func_type.second())
            else:
                caller.accept(self)
                func_type = self.pop_type()
                self.check_arg(func_type, ty)
        else:
            caller.accept(self)
            func_type = self.pop_type()
            self.check_arg(func_type, ty)

    def check_arg(self, func: Type, ty: List[Type]):
        arg = func.first().sub_nodes
        try:
            for i in range(0, len(ty)):
                if arg[i].signature() == '...':
                    break
                if not self.is_equal_type(arg[i], ty[i]):
                    self.error('\nincompatible type at {2}th argument:\n\t{0}\nand\n\t{1}'.format(arg[i], ty[i], i + 1))
        except IndexError:
            self.error('{0} arguments expected, {1} provided'.format(len(arg), len(ty)))
        self.push_type(func.second())

    def visit_call_arg(self, node: CallArg):
        self.visit_arg_template(node)

    def check_cast_type(self, ty1: Type, ty2: Type):
        if ty1 == ty2:
            return
        if ty1.is_arithmetic() and ty2.is_arithmetic:
            return
        if self.is_equal_type(ty1, ty2) and self.showCastWarning:
            self.warning('\nconversion from\n\t{0}\nto\n\t{1}'.format(ty1, ty2))
        else:
            self.error('\nincompatible type during cast:\n\t{0}\nand\n\t{1}'.format(ty1, ty2))

    def visit_cast_expr(self, node):
        node.first().accept(self)
        ty1 = self.pop_type()
        ty2 = node.second()
        self.push_type(ty2)
        self.check_cast_type(ty1, ty2)
        self.pop_temp()
        s = self.temp.buffer
        self.clear_temp()
        node.second().accept(self)
        self.pop_temp()
        s = '(' + self.temp.buffer + ')' + s
        self.clear_temp()
        self.write_and_push(s)

    def visit_while_stmt(self, node: WhileStmt):
        s = 'while('
        node.first().accept(self)
        self.pop_temp()
        s += self.temp.buffer + '){\n'
        self.clear_temp()
        node.second().accept(self)
        self.pop_temp()
        s += self.temp.buffer + '}\n'
        self.clear_temp()
        self.write_and_push(s)

    def visit_return(self, node):
        s = 'return '
        if len(node) > 0:
            node.first().accept(self)
            self.pop_temp()
            s += self.temp.buffer
            self.clear_temp()
        self.write_and_push(s)

    def visit_block(self, block: Block):
        s = ''
        for i in block.sub_nodes:
            i.accept(self)
            self.pop_temp()
            s += self.temp.buffer + ';\n'
            self.clear_temp()
        self.clear_temp()
        self.write_and_push(s)

    def visit_chunk(self, chunk: Chunk):
        s = ''
        self.push_lexical_scope()  # global scope
        for i in chunk.sub_nodes:
            i.accept(self)
            self.pop_temp()
            s += self.temp.buffer + '\n'
            self.clear_temp()
        self.clear_temp()
        self.write_and_push(s)

    def gen_struct(self, node, decorated_name=''):
        s = node.tok.tok
        if decorated_name != '':
            s = decorated_name
        self.add_type(s)
        struct = VarInfo(s, PrimitiveType.make_primitive(s))
        self.write_typedef('struct ' + s + ';\ntypedef struct ' + s + ' ' + s + ";\n")
        s = 'struct ' + s + '{\n'
        # hack here, this is rubbish code. don't do that again
        for i in node.sub_nodes:
            field_name = i.tok.tok
            struct.add_field(VarInfo(field_name, i.first()))
            level = len(self.symbol_table) - 1
            i.accept(self)
            del self.symbol_table[level][field_name]
            self.pop_temp()
            s += self.temp.buffer + ';\n'
            self.clear_temp()
        s += '};\n'
        self.clear_temp()
        self.write_and_push(s)
        self.add_global_symbol_directly(struct)

    def visit_struct(self, node):
        self.gen_struct(node)

    def gen_func_def(self, node: FuncDef, decorated_name=''):
        temp = self.type_stack
        self.type_stack = []
        func_name = node.tok.tok
        self.update_pos(node.tok)
        if decorated_name != '':
            func_name = decorated_name
        self.add_global_symbol(func_name, node.call_signature())
        self.push_lexical_scope()
        node.second().accept(self)
        self.pop_temp()
        s = self.temp.buffer + ' '
        self.clear_temp()
        s += func_name
        node.first().accept(self)
        self.pop_temp()
        s += self.temp.buffer
        self.clear_temp()
        self.write_typedef(s + ';\n')
        s += '{\n'
        node.third().accept(self)
        self.pop_temp()
        s += self.temp.buffer + '}\n'
        self.clear_temp()
        self.write_and_push(s)
        self.pop_lexical_scope()
        self.type_stack = temp

    def visit_func_def(self, node: FuncDef):
        self.gen_func_def(node)

    def visit_func_def_arg(self, node):
        self.visit_arg_template(node)

    def visit_primitive_type(self, node):
        self.write_and_push(node.tok.tok)

    def visit_array_type(self, node: ArrayType):
        size = node.size
        node.first().accept(self)
        self.pop_temp()
        sign = node.signature()
        self.add_type(sign)
        if size >= 0:
            s = 'typedef ' + self.temp.buffer \
                + ' ' + sign + ('[{0}]'.format(size)) + ';\n'
        else:
            s = 'typedef ' + self.temp.buffer + ' * ' + sign + ';\n'
        self.clear_temp()
        self.write_typedef(s)
        self.write_and_push(sign)

    def visit_ptr_type(self, node):
        node.first().accept(self)
        self.pop_temp()
        sign = node.signature()
        self.add_type(sign)
        s = 'typedef ' + self.temp.buffer + ' * ' + sign + ';\n'
        self.clear_temp()
        self.write_typedef(s)
        self.write_and_push(sign)

    def visit_ref_type(self, node):
        node.first().accept(self)
        self.pop_temp()
        sign = node.signature()
        self.add_type(sign)
        s = 'typedef ' + self.temp.buffer + ' * ' + sign + ';\n'
        self.clear_temp()
        self.write_typedef(s)
        self.write_and_push(sign)

    def visit_func_type(self, node: FuncType):
        signature = node.signature()
        self.add_type(signature)
        ret_type = node.second()
        arg = node.first()
        ret_type.accept(self)
        self.pop_temp()
        s = 'typedef ' + self.temp.buffer + ' (*' + signature + ')'
        arg.accept(self)
        self.pop_temp()
        s += self.temp.buffer + ';\n'
        self.write_typedef(s)
        self.clear_temp()
        self.write_and_push(signature)

    def visit_arg_template(self, node):
        self.write_and_push('(')
        for i in node.sub_nodes:
            i.accept(self)
        s = ')'
        for i in node.sub_nodes:
            self.pop_temp()
            if s != ')':
                s = self.temp.buffer + ', ' + s
            else:
                s = self.temp.buffer + s
        self.pop_temp()
        s = '(' + s
        self.clear_temp()
        self.write_and_push(s)

    def visit_func_type_arg(self, node):
        self.visit_arg_template(node)

    def visit_import(self, node):
        for i in node.sub_nodes:
            i.accept(self)
        self.clear_temp()
        self.write_and_push('')

    def visit_c_header(self, node):
        self.write_typedef('#include ' + node.tok.tok + '\n')

    def visit_c_definition(self, node):
        var = node.tok.tok
        decl = node.first()
        ty = decl.first()
        self.add_global_symbol(var, ty)

    def visit_c_type(self, node):
        self.add_symbol(node.tok.tok, PrimitiveType.make_primitive(node.tok.tok))

    def visit_implementation(self, node):
        self.gen_impl(node)

    def gen_impl(self, node, decorated_name=''):
        #  print(node)
        struct_name = node.tok.tok
        self.update_pos(node.tok)
        if decorated_name != '':
            struct_name = decorated_name
        struct = self.retrieve_symbol(struct_name)
        node.link()
        s = ''
        for i in node.sub_nodes:
            struct.add_method(VarInfo(i.tok.tok, i.call_signature()))
            i.class_name = struct_name
            i.accept(self)
            self.pop_temp()
            s += self.temp.buffer
            self.clear_temp()
        self.clear_temp()
        self.write_and_push(s)
        # print(struct.methods)

    def visit_method_def(self, node: MethodDef):
        parent = node.parent
        struct_name = node.class_name
        if parent.type() == 'Implementation':
            func_name = struct_name + '_' + node.tok.tok
        else:
            func_name = '_imp_' + parent.interface.tok + '_' + struct_name + '_' + node.tok.tok
        self.push_lexical_scope()
        node.second().accept(self)
        self.pop_temp()
        s = self.temp.buffer + ' '
        self.clear_temp()
        s += func_name
        node.first().accept(self)
        self.pop_temp()
        s += self.temp.buffer
        self.clear_temp()
        self.write_typedef(s + ';\n')
        s += '{\n'
        node.third().accept(self)
        self.pop_temp()
        s += self.temp.buffer + '}\n'
        self.clear_temp()
        self.write_and_push(s)
        self.pop_lexical_scope()

    def interface_method_table_decoration(self, interface: str) -> str:
        return '_tab_' + interface

    def interface_impl(self) -> str:
        return 'impl'

    def interface_method_table(self) -> str:
        return 'table'

    def visit_interface(self, node):
        n = node.tok.tok
        tab = self.interface_method_table_decoration(n)
        struct = VarInfo(n, PrimitiveType.make_primitive(n))
        self.write_typedef('struct ' + n + ';\ntypedef struct ' + n + ' ' + n + ";\n")
        self.write_typedef('struct ' + tab + ';\ntypedef struct ' + tab + ' ' + tab + ";\n")
        n = 'struct ' + n + '{\nvoid*' + self.interface_impl() + \
            ';\n' + tab + '*' + self.interface_method_table() + ';\n};\n'
        s = n
        s = s + 'struct ' + tab + '{\n\n'
        # hack here, this is rubbish code. don't do that again
        temp = self.symbol_table
        self.symbol_table = [dict()]
        for i in node.sub_nodes:
            field_name = i.tok.tok
            struct.add_field(VarInfo(field_name, i.first()))
            i.accept(self)
            self.pop_temp()
            s += self.temp.buffer + ';\n'
            self.clear_temp()
        s += '};\n'
        self.clear_temp()
        self.write_and_push(s)
        self.symbol_table = temp
        self.interfaces[struct.name] = struct

    def visit_impl_for(self, node):
        struct_name = node.interface.tok
        struct = self.interfaces[struct_name]
        node.link()
        s = ''
        for i in node.sub_nodes:
            struct.add_method(VarInfo(i.tok.tok, i.call_signature()))
            i.accept(self)
            self.pop_temp()
            s += self.temp.buffer
            self.clear_temp()
        self.clear_temp()
        self.write_and_push(s)

    def get_generic_impl_struct_name(self, node: Generic) -> str:
        return node.first().tok.tok

    def add_generic_impl(self, node: Generic):
        self.generic_impls.append(node)

    def is_instantialized(self, node: Generic):
        flag = True
        for i in node.real_type_list():
            if not self.is_good_type(i):
                flag = False
                break
        return flag

    def is_good_type(self, type: Node):
        return self.has_type(type.signature())

    def gen_generic_func_and_struct(self, node: Generic):
        if not self.is_instantialized(node):
            f = node.first()
            self.add_generic(f.tok.tok, node)
            self.clear_temp()
            self.write_and_push('')
        else:
            f = node.first()
            if f.type() == 'Struct':
                func_name = f.tok.tok
                self.update_pos(f.tok)
                decorated_name = self.generic_decorate(func_name, node.real_type_list())
                self.gen_struct(f, decorated_name)
            elif f.type() == 'FuncDef':
                func_name = f.tok.tok
                self.update_pos(f.tok)
                decorated_name = self.generic_decorate(func_name, node.real_type_list())
                self.gen_func_def(f, decorated_name)

    def gen_generic_impl(self, node: Generic, type_list):
        self.instantiate_generic(node, type_list)

    def process_generic_impl(self, node):
        if not self.is_instantialized(node):
            self.add_generic_impl(node)
            self.write_and_push('')
        else:
            f = node.first()
            func_name = f.tok.tok
            self.update_pos(f.tok)
            decorated_name = self.generic_decorate(func_name, node.real_type_list())
            self.gen_impl(node.first(), decorated_name)

    def visit_generic(self, node: Generic):
        f = node.first()
        for i in node.real_type_list():
            if i.type() == 'Generic':
                i.accept(self)
                self.pop_temp()
                self.clear_temp()
        if f.type() in ['FuncDef', 'Struct']:
            self.gen_generic_func_and_struct(node)
        elif f.type() == 'Implementation':
            self.process_generic_impl(node)
        elif f.type() == 'Identifier':
            if node.parent.type() == 'Call':
                func_name = f.tok.tok
                self.update_pos(f.tok)
                decorated_name = self.generic_decorate(func_name, node.real_type_list())
                # print(decorated_name)
                f = self.retrieve_symbol(decorated_name)
                if not f:  # not yet instantiated
                    generic = (self.retrieve_generic(func_name))
                    generic.parent = None
                    assert generic is not self.generics[func_name]
                    temp = self.type_stack
                    generic.link()
                    self.type_stack = []
                    if not generic:
                        self.error('attempt to call a non generic object')
                    self.instantiate_generic(generic, node.real_type_list())
                    self.pop_temp()
                    self.write_typedef(self.temp.buffer)
                    self.clear_temp()
                    f = self.retrieve_symbol(decorated_name)
                    if not f:
                        self.update_pos(node.first().tok)
                        self.error('failed to instantiate generic:\n\t{0}'.format(node))
                    self.type_stack = temp
                self.clear_temp()
                self.write_and_push(decorated_name)
                self.push_type(f.type)
                # print(f.type)
        elif f.type() == 'PrimitiveType':
            struct_name = f.tok.tok
            decorated_name = self.generic_decorate(struct_name, node.real_type_list())
            # print(decorated_name)
            f = self.retrieve_symbol(decorated_name)
            if not f:  # not yet instantiated
                generic = self.retrieve_generic(struct_name)
                if not generic:
                    self.error('attempt to instantiate a non generic object')
                generic.link()
                self.instantiate_generic(generic, node.real_type_list())
                self.pop_temp()
                self.write_typedef(self.temp.buffer)
                self.clear_temp()
                self.instantiate_generic_impl(struct_name, node.real_type_list())
                self.pop_temp()
                self.write_typedef(self.temp.buffer)
                self.clear_temp()
                f = self.retrieve_symbol(decorated_name)
            self.clear_temp()
            self.write_and_push(decorated_name)
        else:
            self.write_and_push('')

    def instantiate_generic_impl(self, name: str, type_list: List[Type]):
        for i in self.generic_impls:
            if self.get_generic_impl_struct_name(i) == name:
                self.gen_generic_impl(i, type_list)

    @staticmethod
    def generic_decorate(name: str, type_list: List[Type]) -> str:
        s = name + '_GG_'
        for i in type_list:
            s += i.signature() + '_GG_'
        return s

    def instantiate_generic(self, node: Generic, type_list: List[Type], mutate=False):
        generic_type_list = node.type_list
        type_map = dict()
        if len(generic_type_list) != len(type_list):
            self.error('{0} arguments expected in generic, {1} found'.format(
                len(generic_type_list), len(type_list)
            ))
        for i in range(0, len(generic_type_list)):
            type_map[generic_type_list[i].signature()] = type_list[i]
        instance = node.copy()

        n0 = instance.first().tok.tok
        g0 = self.generics[n0].copy()  # self.generics is been modified by unknown source
        assert instance.first() is not self.generics[n0].first()
        idx = instance.index
        p = instance.parent
        instance = self.instantiate_generic_with_type_map(instance, type_map)
        assert instance.first() is not self.generics[n0].first()
        # instance = p.sub_nodes[idx]
        instance.link()
        self.generics[n0] = g0  # why? why?
        temp = instance.sub_nodes[0].tok.tok
        if instance.type() != "Struct":
            assert instance.first() is not self.generics[n0].first()
            n = self.generic_decorate(n0, type_list)
            t = instance.first().tok

        # print(instance)
        instance.accept(self)

    def instantiate_generic_with_type_map(self, node: Node, type_map) -> Node:
        for i in range(0, len(node)):
            node.sub_nodes[i] = self.instantiate_generic_with_type_map(node.sub_nodes[i], type_map)
        if node.type() == 'PrimitiveType':
            ty = node.tok.tok
            if ty in type_map:
                node = type_map[ty].copy()
        if node.type() == 'Generic':
            replace = True
            for i in range(0, len(node.type_list)):
                sign = node.type_list[i].signature()
                if sign in type_map:
                    node.type_list[i] = type_map[sign]
                else:
                    replace = False
            node.link()
        return node

    def visit_type_inference(self, node):
        decl = node.first()
        expr = decl.first()
        expr.accept(self)
        ty = self.pop_type()
        ty.accept(self)
        self.add_symbol(decl.tok.tok, ty)
        self.pop_temp()
        s = self.temp.buffer + ' '
        s += decl.tok.tok
        self.pop_temp()
        s += '=' + self.temp.buffer
        self.clear_temp()
        self.write_and_push(s)
