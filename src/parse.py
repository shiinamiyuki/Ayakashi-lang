from lex import *
from node import *


class Parser:
    def __init__(self, lex: Lexer):
        self.token_stream = lex.token_stream
        self.idx = -1
        self.op_precedence = dict()
        self.precedence = 0
        self.op_assoc = dict()
        self.init_op_precedence_and_assoc()
        self.types = [['int', 'float', 'double', 'char']]  # type: List[List[str]]

    def push_type_scope(self):
        self.types.append([])

    def pop_type_scope(self):
        self.types.pop()

    def add_type(self, t: str, i:int= -1):
        self.types[i].append(t)

    def init_op_precedence_and_assoc(self):
        for i in [x for x in '+= -= = *= /= %= >>= <<= &= |='.split(' ') if x]:
            self.add_op(i, 0)
        self.inc_precedence()
        self.add_op('or', 1)
        self.add_op('||', 1)
        self.inc_precedence()
        self.add_op('and', 1)
        self.add_op('&&', 1)
        self.inc_precedence()
        self.add_op('|', 1)
        self.inc_precedence()
        self.add_op('^', 1)
        self.inc_precedence()
        self.add_op('&', 1)
        self.inc_precedence()
        self.add_op('==', 1)
        self.add_op('!=', 1)
        self.inc_precedence()
        self.add_op('>=', 1)
        self.add_op('<=', 1)
        self.add_op('>', 1)
        self.add_op('<', 1)
        self.inc_precedence()
        self.add_op('>>', 1)
        self.add_op('<<', 1)
        self.inc_precedence()
        self.add_op('+', 1)
        self.add_op('-', 1)
        self.inc_precedence()
        self.add_op('*', 1)
        self.add_op('/', 1)
        self.add_op('%', 1)
        self.inc_precedence()
        # self.add_op('.', 1)
        # self.add_op('->', 1)

    def add_op(self, op, assoc: int):
        self.op_assoc[op] = assoc
        self.op_precedence[op] = self.precedence

    def inc_precedence(self):
        self.precedence += 1

    def peek(self, i=1) -> Token:
        try:
            return self.token_stream[self.idx + i]
        except IndexError:
            return nil_token

    def cur(self) -> Token:
        return self.token_stream[self.idx]

    def next(self):
        self.idx += 1

    def expect(self, tok):
        if self.peek().tok != tok:
            self.error('{0} expected but have {1}'.format(tok, self.peek().tok))
        self.next()

    def has_next(self):
        return self.idx + 1 < len(self.token_stream)

    def has(self, tok):
        return self.peek().tok == tok

    def error(self, msg: str):
        raise RuntimeError("Parser Error: {0} {1}:{2}".format(
            msg,
            self.cur().line,
            self.cur().col))

    def parse_atom(self) -> Node:
        if self.peek().type == 'identifier':
            self.next()
            iden = Identifier(self.cur())
            if self.is_generic():
                g = self.parse_generic()
                g.add(iden)
                return g
            return iden
        elif self.peek().type == 'number':
            self.next()
            return Number(self.cur())
        elif self.peek().type == 'string':
            self.next()
            return String(self.cur())
        elif self.has('('):
            self.next()
            e = self.parse_binary_expr()
            self.expect(')')
            return e
        else:
            self.error('illegal token {0}'.format(self.peek().tok))

    def parse_cast_expr(self) -> Node:
        result = self.parse_unary_expr()
        if self.has('as'):
            self.next()
            cast = CastExpr()
            cast.add(result)
            cast.add(self.parse_type())
            result = cast
        return result

    def parse_unary_expr(self) -> Node:
        if self.peek().tok in ['-', '*', '&', 'not', '!']:
            self.next()
            e = UnaryExpr(self.cur())
            e.add(self.parse_unary_expr())
            return e
        if self.peek().tok == 'sizeof':
            self.next()
            e = UnaryExpr(self.cur())
            e.add(self.parse_type())
            return e
        else:
            return self.parse_postfix_expr()

    def parse_postfix_expr(self) -> Node:
        result = self.parse_atom()
        while self.has_next() and (self.has('[') or
                                   self.has('(') or
                                   self.has('.') or
                                   self.has('->')):
            if self.has('['):
                self.next()
                index = Index()
                index.add(result)
                index.add(self.parse_binary_expr())
                self.expect(']')
                result = index
            elif self.has('.') or self.has('->'):
                self.next()
                e = BinaryExpr(self.cur(), 20)
                e.add(result)
                e.add(self.parse_atom())
                result = e
            else:
                call = Call()
                call.add(result)
                call.add(self.parse_call_arg())
                result = call
        return result

    def parse_call_arg(self) -> Node:
        self.expect('(')
        result = CallArg()
        while self.has_next() and not self.has(')'):
            result.add(self.parse_binary_expr())
            if self.has(')'):
                break
            else:
                self.expect(',')
        self.expect(')')
        return result

    def parse_binary_expr(self, lev: int = 0) -> Node:
        """
        parses binary expressions using precedence climbing
        :param lev: operator precedence
        :return:
        """
        result = self.parse_cast_expr()
        while self.has_next():
            _next = self.peek()
            if _next.tok not in self.op_precedence:
                break
            else:
                if self.op_precedence[_next.tok] >= lev:
                    self.next()
                    rhs = self.parse_binary_expr(
                        self.op_assoc[_next.tok] + self.op_precedence[_next.tok])
                    op = BinaryExpr(_next, self.op_precedence[_next.tok])
                    op.add(result)
                    op.add(rhs)
                    result = op
                else:
                    break
        return result

    def parse_stmt(self) -> Node:
        if self.has('if'):
            return self.parse_if()
        elif self.has('import'):
            return self.parse_import()
        elif self.has('interface'):
            return self.parse_interface()
        elif self.has('let'):
            self.next()
            return self.parse_declaration()
        elif self.has('impl'):
            return self.parse_impl()
        elif self.has('while'):
            return self.parse_while()
        elif self.has('type'):
            return self.parse_struct()
        elif self.has('fn'):
            return self.parse_func_def()
        elif self.has('return'):
            return self.parse_return()
        else:
            return self.parse_binary_expr()

    def parse_var_def(self):
        self.expect('let')
        return self.parse_declaration()

    def parse_while(self):
        self.expect('while')
        result = WhileStmt()
        result.add(self.parse_binary_expr())
        result.add(self.parse_block())
        return result

    def parse_if(self):
        self.expect('if')
        result = IfStmt()
        result.add(self.parse_binary_expr())
        result.add(self.parse_block())
        while self.has_next() and self.has('else'):
            self.next()
            if self.has('if'):
                self.next()
                result.add(self.parse_binary_expr())
                result.add(self.parse_block())
            else:
                result.add(self.parse_block())
                break
        return result

    def parse_return(self):
        self.expect('return')
        result = Return()
        result.add(self.parse_binary_expr())
        return result

    def parse_block(self) -> Node:
        self.expect('{')
        block = Block()
        while self.has_next() and not self.has('}'):
            block.add(self.parse_stmt())
        self.expect('}')
        return block

    def parse_generic(self) -> Generic:
        self.expect('<')
        type_list = []
        while self.has_next() and not self.has('>'):
            if self.has('type'):
                self.next()
            t = self.parse_type()
            type_list.append(t)
            if t.is_primitive():
                self.add_type(t.signature())
            if self.has('>'):
                break
            self.expect(',')
        self.expect('>')
        return Generic(type_list)

    def is_next_token_type(self, i=1):
        try:
            tok = self.peek(i).tok
            for i in self.types:
                if tok in i:
                    return True
            return False
        except IndexError:
            return False

    def is_generic(self) -> bool:
        if not self.has('<'):
            return False
        if self.is_next_token_type(2):
            return True
        if self.peek(2).tok == 'type':
            return True
        return False

    def parse_func_def(self) -> FuncDef:
        self.expect('fn')
        func = FuncDef(self.peek())
        result = func
        self.next()
        self.push_type_scope()
        if self.has('<'):
            result = self.parse_generic()
            result.add(func)
        func.add(self.parse_func_def_arg())
        self.expect('->')
        func.add(self.parse_type())
        func.add(self.parse_block())
        self.pop_type_scope()
        return result

    def parse_func_def_arg(self):
        self.expect('(')
        arg = FuncDefArg()
        while self.has_next() and not self.has(')'):
            arg.add(self.parse_declaration())
            if self.has(')'):
                break
            self.expect(',')
        self.expect(')')
        return arg

    def parse_declaration(self):
        decl = Declaration(self.peek())
        self.next()
        if self.has(':'):
            self.expect(':')
            decl.add(self.parse_type())
            if self.has('='):
                self.next()
                decl.add(self.parse_binary_expr())
            return decl
        else:
            self.expect('=')
            decl.add(self.parse_binary_expr(0))
            infer = TypeInference()
            infer.add(decl)
            return infer

    def parse_interface(self):
        self.expect('interface')
        result = Interface(self.peek())
        self.add_type(result.tok.tok,0)
        self.next()
        self.expect('{')
        while self.has_next() and not self.has('}'):
            result.add(self.parse_declaration())
        self.expect('}')
        return result

    def parse_struct(self):
        self.expect('type')
        result = Struct(self.peek())
        struct = result
        self.push_type_scope()
        self.add_type(result.tok.tok,0)
        self.next()
        if self.has('<'):
            result = self.parse_generic()
            result.add(struct)
        self.expect('{')
        while self.has_next() and not self.has('}'):
            struct.add(self.parse_declaration())
        self.expect('}')
        self.pop_type_scope()
        return result

    def parse_primitive_type(self) -> Node:
        self.next()
        result = PrimitiveType(self.cur())
        if self.has('<'):
            t = result
            result = self.parse_generic()
            result.add(t)
        return result

    def parse_arr_type(self) -> Type:
        size = -1
        self.expect('[')
        if not self.has(']'):
            self.next()
            try:
                size = int(self.cur().tok)
            except ValueError:
                self.error('int constant expected')
        self.expect(']')
        result = ArrayType(size)
        result.add(self.parse_type())
        return result

    def parse_type(self) -> Type:
        if self.has('('):
            return self.parse_func_type()
        elif self.has('['):
            return self.parse_arr_type()
        elif self.has('&'):
            return self.parse_ref_type()
        elif self.has('*'):
            return self.parse_ptr_type()
        else:
            return self.parse_primitive_type()

    def parse_ptr_type(self):
        self.expect('*')
        ptr = PointerType()
        ptr.add(self.parse_type())
        return ptr

    def parse_ref_type(self):
        self.expect('&')
        ref = RefType()
        ref.add(self.parse_type())
        return ref

    def parse_func_type_arg(self) -> Type:
        arg = FuncTypeArg()
        self.expect('(')
        while self.has_next() and not self.has(')'):
            arg.add(self.parse_type())
            if self.has(')'):
                break
            else:
                self.expect(',')
        self.expect(')')
        return arg

    def parse_func_type(self) -> Type:
        arg = self.parse_func_type_arg()
        self.expect('->')
        ret_type = self.parse_type()
        result = FuncType()
        result.add(arg)
        result.add(ret_type)
        return result

    def parse_import(self) -> Import:
        self.expect('import')
        self.expect('{')
        i = Import()
        while self.has_next() and not self.has('}'):
            i.add(self.parse_import_stmt())
        self.expect('}')
        return i

    def parse_import_stmt(self):
        if self.has('cheader'):
            self.next()
            self.next()
            return CHeader(self.cur())
        elif self.has('cdef'):
            self.next()
            cdef = CDefinition(self.peek())
            cdef.add(self.parse_declaration())
            return cdef
        else:
            self.error('unknown import stmt ' + self.peek().tok)

    def parse(self):
        chunk = Chunk()
        while self.has_next():
            chunk.add(self.parse_stmt())
        return chunk

    def parse_impl(self):
        self.expect('impl')
        impl = Implementation(self.peek())
        struct_name = self.peek()
        result = impl
        self.next()
        generic = False
        if self.has('<'):
            result = self.parse_generic()
            result.add(impl)
            generic = True
        if self.has('for'):
            self.next()
            impl = ImplFor(impl.tok, self.peek())
            self.next()
        self.expect('{')
        # hack
        self_type = RefType()

        if generic:
            g = Generic(result.real_type_list())
            g.add(PrimitiveType.make_primitive(impl.tok.tok))
            self_type.add(g)
        else:
            self_type.add(PrimitiveType.make_primitive(impl.tok.tok))
        self_var = Declaration(Token('self', 'identifier', 0, 0))
        self_var.add(self_type)
        while self.has_next() and not self.has('}'):
            f = self.parse_func_def()
            m = MethodDef(f)
            m.first().sub_nodes.insert(0, self_var)
            impl.add(m)
        self.expect('}')
        return result
