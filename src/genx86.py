from node import *
from visitor import Visitor
from typing import Dict, List, Optional
from copy import deepcopy

template = r'''
    .file	"temp.c"
    .text
    .def	__main;	.scl	2;	.type	32;	.endef
    .section .rdata,"dr"
    .LC0:
    .ascii "%d\0"
    .text
    .globl	main
    .def	main;	.scl	2;	.type	32;	.endef
    .seh_proc	main
main:
    pushq	%rbp
    .seh_pushreg	%rbp
    movq	%rsp, %rbp
    .seh_setframe	%rbp, 0
    subq	$48, %rsp
    .seh_stackalloc	48
    .seh_endprologue
    call	__main
{0}
    movl    %eax, -4(%rbp)
    movl	-4(%rbp), %eax
    movl	%eax, %edx
    leaq	.LC0(%rip), %rcx
    call	printf
    movl	$0, %eax
    addq	$48, %rsp
    popq	%rbp
    ret
    .seh_endproc
    .ident	"GCC: (x86_64-posix-seh-rev0, Built by MinGW-W64 project) 8.1.0"
    .def	printf;	.scl	2;	.type	32;	.endef
'''

function_template = '''
    pushq	%rbp
    movq	%rsp, %rbp
    subq	$16, %rsp
{0}
    addq	$16, %rsp
    popq	%rbp
    ret
'''


class VarInfo:
    def __init__(self, tok: Token, ty: Type, addr: int):
        self.tok = tok
        self.type = ty
        self.addr = addr

    def gen_addr(self):
        if self.type == PrimitiveType.make_primitive('int'):
            if self.addr == 0:
                return '%ecx'
            elif self.addr == 1:
                return '%edx'
            elif self.addr == 2:
                return '%r8d'
            elif self.addr == 3:
                return '%r9d'
            else:
                return '-{0}(%rbp'.format(48 - (self.addr - 4) * 8)


class CodeGenx86_64(Visitor):
    def __init__(self):
        super().__init__()
        self.output = ''
        self.empty = True
        self.dict = []  # type: List[Dict[str,VarInfo]]

    def add_symbol(self, s:str, v:VarInfo):
        self.dict[-1][s] = v

    def add_global_symbol(self, s:str, v:VarInfo):
        self.dict[0][s] = v

    def push_scope(self):
        self.dict.append([])

    def pop_scope(self):
        self.dict.pop()

    def clear_stack(self):
        self.empty = True

    def write(self, s: str):
        self.output += '\t' + s + '\n'

    def write_label(self, s: str):
        self.output += s + ':\n'

    def visit_c_type(self, node):
        pass

    def visit_chunk(self, node: Chunk):
        for i in node.sub_nodes:
            i.accept(self)

    def visit_binary_expr(self, node):
        node.second().accept(self)
        node.first().accept(self)
        op_map = {
            '+': 'addl',
            '-': 'subl',
            '*': 'imul',
            '/': 'idiv',
            '>': 'setg',
            '>=': 'setge',
            '<': 'setl',
            '<=': 'setle',
            '==': 'sete',
            '!=': 'setne',
        }
        self.write('popq %rbx')
        if node.tok.tok in ['+', '-', '*', '/']:
            self.write('{0} %ebx, % eax'.format(op_map[node.tok.tok]))
        elif node.tok.tok in ['>', '<', '>=', '<=', '==', '!=']:
            self.write('cmpl %ebx, %eax')
            self.write('{0} %al'.format(op_map[node.tok.tok]))
            self.write('movzbl	%al, %eax')

    def visit_number(self, node):
        if not self.empty:
            self.write('pushq %rax')
        else:
            self.empty = False
        i = int(node.tok.tok)
        self.write('movl ${0}, %eax'.format(i))

    def visit_block(self, node):
        pass

    def visit_return(self, node):
        self.clear_stack()
        node.first().accept(self)
        self.write('ret')

    def refine(self):
        self.output = template.format(self.output)



