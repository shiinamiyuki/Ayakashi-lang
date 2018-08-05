from parse import *
from gen import *
import os
import sys
import genx86
from copy import deepcopy


class Compiler:
    def __init__(self):
        self.filename = ''

    def compilex86_64(self, s: str, filename='out'):
        try:
            self.filename = filename
            lex = Lexer(s)
            lex.parse()
            p = Parser(lex)
            ast = p.parse()
            ast.link()
            print(ast)
            gen = genx86.CodeGenx86_64()
            gen.filename = filename + '.aya'
            ast.accept(gen)
            gen.refine()
            file = open(self.filename + '.s', 'w')
            file.write(gen.output)
            file.close()
            self.call_as()
        except RuntimeError as e:
            print(e, file=sys.stderr)

    def compile(self, s: str, filename='out'):
        try:
            self.filename = filename
            lex = Lexer(s)
            lex.parse()
            p = Parser(lex)
            ast = p.parse()
            ast.link()
            gen = CodeGen()
            gen.filename = filename + '.aya'
            ast.accept(gen)
            gen.pop_temp()
            gen.write_typedefs_to_source()
            gen.write_temp_to_source_and_destroy_temp()
            file = open(self.filename + '.c', 'w')
            file.write(gen.produced_source)
            file.close()
            self.call_cc()
        except RuntimeError as e:
            print(e, file=sys.stderr)

    def call_as(self):
        cmd = (r'''D:\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin\gcc -o {0}.exe {0}.s''' + \
               ''r' -lglfw3 -lgdi32 -lopengl32''').format(self.filename)
        print('Running C compiler: ' + cmd)
        os.system(cmd)

    def call_cc(self):
        cmd = (r'''D:\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin\gcc -o {0}.exe {0}.c''' + \
               ''r' -lglfw3 -lgdi32 -lopengl32''').format(self.filename)
        print('Running C compiler: ' + cmd)
        os.system(cmd)
