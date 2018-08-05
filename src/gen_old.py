from node import *
from visitor import Visitor
from typing import Dict, List, Optional


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


includes = '''
#define null NULL
'''


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
        self.typedefs = includes
        self.type = None  # type : Type
        self.type_stack = []  # type:List[Type]
        self.symbol_table = []  # type: List[Dict[str,VarInfo]]
        self.interfaces = dict()  # type: Dict[str,VarInfo]

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

    def add_symbol(self, var: str, ty: Type):
        self.symbol_table[-1][var] = VarInfo(var, ty)

    def add_symbol_directly(self, v: VarInfo):
        self.symbol_table[-1][v.name] = v

    def push_lexical_scope(self):
        self.symbol_table.append(dict())

    def pop_lexical_scope(self):
        self.symbol_table.pop()

    def write_typedef(self, s: str):
        self.typedefs += s

    def push_temp(self):
        self.temp_stack.append(self.temp)
        self.temp = Buffer()

    def pop_temp(self):
        self.temp = self.temp_stack.pop()

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
        self.produced_source += \
            '''
            
            /// typedefs
            
            
                        '''
        self.produced_source += self.typedefs
        self.produced_source += '''

/// end of typedefs



'''

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
        return ty1 == ty2

    def check_binary_expr_type_int_only_op(self, op, ty1: Type, ty2: Type):
        if op not in [x for x in '& ^ | %' if x]:
            return False
        if not ty1.is_int() or not ty2.is_int():
            raise RuntimeError('integer expected')
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
        if ty1.is_pointer() or ty2.is_pointer():
            raise RuntimeError('attempt to perform pointer arithmetic')
        if ty1.is_reference() or ty2.is_reference():
            raise RuntimeError('operator overloading not implemented yet!')
        return False

    def check_type_field_and_method(self, op, ty1: Type, ty2: Type) -> bool:
        if op not in ('.', '->'):
            return False
        if not ty2.is_primitive():
            raise RuntimeError('identifier expected after \'.\'')
        struct_name = ty1.signature()
        field_name = ty2.signature()
        struct_type = self.retrieve_symbol(struct_name).type
        if struct_type.is_reference() or struct_type.is_pointer():
            struct_type = struct_type.first()
        if struct_type.is_reference() or struct_type.is_pointer():
            # still a reference
            raise RuntimeError("don't do that")
        struct = self.retrieve_symbol(struct_type.signature())
        if not struct:
            print(struct_type)
            raise RuntimeError('www')
        field = struct.retrieve(field_name)
        if not field:
            # print(struct)
            # print(self.type_stack,self.symbol_table)
            raise RuntimeError('no such field ' + field_name)
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
            raise RuntimeError('type boom!')

    def check_index_expr_type(self):
        ty2 = self.pop_type()
        ty1 = self.pop_type()
        if not ty2.is_int():
            raise RuntimeError('int expected in index expr')
        self.push_type(ty1.first())

    def visit_declaration(self, node):
        ty = node.first()
        self.push_type(ty)
        ty.accept(self)
        self.pop_temp()
        s = self.temp.buffer + ' '
        var = node.tok.tok
        s += var
        self.add_symbol(var, ty)
        self.clear_temp()
        self.write_and_push(s)

    def visit_identifier(self, node):
        self.write_and_push(node.tok.tok)
        parent = node.parent
        if parent.type() == 'BinaryExpr' and parent.tok.tok in ['.', '->']:
            # hack here, rubbish code again
            self.push_type(PrimitiveType.make_primitive(node.tok.tok))
        elif node.tok.tok in ['null', 'false', 'true']:
            self.push_type(PrimitiveType.make_primitive('int'))
        else:
            v = self.retrieve_symbol(node.tok.tok)
            if v:
                self.push_type(v.type)
            else:
                raise RuntimeError('unresolvable reference to ' + node.tok.tok)

    def visit_string(self, node):
        self.write_and_push(node.tok.tok)
        t = PrimitiveType(Token('char', '', 0, 0))
        a = ArrayType(-1)
        a.add(t)
        self.push_type(a)

    def get_type_hack(self):
        ty = self.type_stack[-1]
        ty = self.retrieve_symbol(ty.signature())
        if ty:
            return ty.type
        return None

    def visit_dot_expr(self, node: BinaryExpr) -> bool:
        left = node.first()
        right = node.second()
        if left.type() != 'Identifier' or right.type() != 'Identifier':
            return False
        struct_name = left.tok.tok
        method_name = right.tok.tok
        struct = self.retrieve_symbol(struct_name)
        struct_type = struct.type
        if struct_type.is_pointer() or struct_type.is_reference():
            struct_type = struct_type.first()
        struct = self.retrieve_symbol(struct_type.signature())
        method = struct.retrieve_method(method_name)
        if not method:
            return False
        ty = struct.type
        if ty.is_reference() or ty.is_pointer():
            ty = ty.first()
        if not ty.is_primitive():
            raise RuntimeError('illegal method call')
        s = ty.signature() + '_' + right.tok.tok
        self.clear_temp()
        self.write_and_push(s)
        self.push_type(method.type)
        return True

    def visit_binary_expr(self, node: BinaryExpr):
        if self.visit_dot_expr(node):
            return
        left = node.first()
        if left.precedence < node.precedence:
            self.write_and_push('(')
        else:
            self.write_and_push('')
        left.accept(self)
        s = ''
        if left.precedence < node.precedence:
            s = ')'
        #  s += node.tok.tok
        op = node.tok.tok
        ty = self.get_type_hack()
        if ty and (ty.is_reference() or ty.is_pointer()) and op == '.':
            op = '->'
        # end of converting . into ->
        s += op
        right = node.second()
        if right.precedence < node.precedence:
            s += '('
        self.write_and_push(s)
        right.accept(self)
        s = ''
        for i in range(0, 4):
            self.pop_temp()
            s = self.temp.buffer + s
        if right.precedence < node.precedence:
            s += ')'
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
        s = op + '('
        node.first().accept(self)
        self.pop_temp()
        s += self.temp.buffer + ')'
        self.clear_temp()
        self.write_and_push(s)
        ty = self.pop_type()
        if op == '&':
            t = RefType()
            t.add(ty)
            self.push_type(t)
        elif op == '*':
            if ty.is_pointer() or ty.is_reference():
                self.push_type(ty.first())
            else:
                raise RuntimeError("could not dereference\n {0}".format(ty))

    def method_call(self, node: Call) -> bool:
        method = node.first()
        if len(method) < 2:
            return False
        if not self.visit_dot_expr(method):
            return False
        arg = node.second()
        self.method_call_arg(arg, method.first())
        return True

    def method_call_arg(self, node: CallArg, caller: Node):
        f = self.pop_type()
        arg = f.first()
        self.write_and_push('(')
        caller.accept(self)
        ty = self.pop_type()
        ty = self.retrieve_symbol(ty.signature()).type
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
            ty_list.insert(0, self.pop_temp())

        if len(arg) - 1 != len(node):
            raise RuntimeError('{0} arguments expected, but have {1}'.format(len(arg) - 1, len(node)))
        for i in range(0, len(node)):
            if not self.is_equal_type(arg[i + 1], node.sub_nodes[i]):
                raise RuntimeError('incompatible type {0}\n and {1}'.format(arg[i + 1], node.sub_nodes[i]))
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
        if self.method_call(node):
            return
        node.first().accept(self)
        node.second().accept(self)
        s = ''
        for i in range(0, 2):
            self.pop_temp()
            s = self.temp.buffer + s
        self.clear_temp()
        self.write_and_push(s)

    def visit_call_arg(self, node: CallArg):
        self.visit_arg_template(node)
        ty = []
        for i in range(0, len(node.sub_nodes)):
            ty.append(self.pop_type())
        func = self.pop_type()
        arg = func.first().sub_nodes
        if len(arg) != len(ty):
            raise RuntimeError('{0} arguments expected, {1} provided'.format(len(arg), len(ty)))
        ty.reverse()
        for i in range(0, len(arg)):
            if not self.is_equal_type(arg[i], ty[i]):
                raise RuntimeError('incompatible type\n {0}\n and {1}'.format(arg[i], ty[i]))
        self.push_type(func.second())

    def visit_cast_expr(self, node):
        node.first().accept(self)
        self.pop_type()
        self.push_type(node.second())
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
        self.pop_lexical_scope()

    def visit_struct(self, node):
        s = node.tok.tok
        struct = VarInfo(s, PrimitiveType.make_primitive(s))
        self.write_typedef('struct ' + s + ';\ntypedef struct ' + s + ' ' + s + ";\n")
        s = 'struct ' + s + '{\n'
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
        self.add_symbol_directly(struct)

    def visit_func_def(self, node: FuncDef):
        self.add_symbol(node.tok.tok, node.call_signature())
        self.push_lexical_scope()
        node.second().accept(self)
        self.pop_temp()
        s = self.temp.buffer + ' '
        self.clear_temp()
        s += node.tok.tok
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

    def visit_func_def_arg(self, node):
        self.visit_arg_template(node)

    def visit_primitive_type(self, node):
        self.write_and_push(node.tok.tok)

    def visit_array_type(self, node: ArrayType):
        size = node.size
        node.first().accept(self)
        self.pop_temp()
        sign = node.signature()
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
        s = 'typedef ' + self.temp.buffer + ' * ' + sign + ';\n'
        self.clear_temp()
        self.write_typedef(s)
        self.write_and_push(sign)

    def visit_ref_type(self, node):
        node.first().accept(self)
        self.pop_temp()
        sign = node.signature()
        s = 'typedef ' + self.temp.buffer + ' * ' + sign + ';\n'
        self.clear_temp()
        self.write_typedef(s)
        self.write_and_push(sign)

    def visit_func_type(self, node: FuncType):
        signature = node.signature()
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
        self.add_symbol(var, ty)

    def visit_implementation(self, node):
        struct_name = node.tok.tok
        struct = self.retrieve_symbol(struct_name)
        s = ''
        for i in node.sub_nodes:
            struct.add_method(VarInfo(i.tok.tok, i.call_signature()))
            i.accept(self)
            self.pop_temp()
            s += self.temp.buffer
            self.clear_temp()
        self.clear_temp()
        self.write_and_push(s)
        # print(struct.methods)

    def visit_method_def(self, node):
        parent = node.parent
        struct_name = parent.tok.tok
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

    def visit_interface(self, node):
        n = node.tok.tok
        tab = '_tab_' + n
        struct = VarInfo(n, PrimitiveType.make_primitive(n))
        self.write_typedef('struct ' + n + ';\ntypedef struct ' + n + ' ' + n + ";\n")
        self.write_typedef('struct ' + tab + ';\ntypedef struct ' + tab + ' ' + tab + ";\n")
        n = 'struct ' + n + '{\nvoid* __impl__;\n' + tab + '*' + '__table__;\n};\n'
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
        s = ''
        for i in node.sub_nodes:
            struct.add_method(VarInfo(i.tok.tok, i.call_signature()))
            i.accept(self)
            self.pop_temp()
            s += self.temp.buffer
            self.clear_temp()
        self.clear_temp()
        self.write_and_push(s)
