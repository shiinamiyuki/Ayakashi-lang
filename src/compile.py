from parse import *
from gen import *
import os
import sys
import genx86
import time
from copy import deepcopy


class Compiler:
    def __init__(self):
        self.filename = ''

    def call_pp(self):
        cmd = r'gcc -E pp.c ' + \
            '-o pp.i'
        os.system(cmd)

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

        except RuntimeError as e:
            print(e, file=sys.stderr)

    def compile(self, s: str, filename='out'):
        gen = CodeGen()
        try:
            self.filename = filename
            pp = open('pp.c','w')
            pp.write(s)
            pp.close()
            self.call_pp()
            time_start = time.time()
            src = open('pp.i','r')
            s = src.read()
            lex = Lexer(s)
            lex.parse()
            p = Parser(lex)
            ast = p.parse()
            ast.link()
            file = open('ast.txt','w')
            file.write(str(ast))
            file.close()

            gen.filename = filename + '.aya'
            ast.accept(gen)
            gen.pop_temp()
            gen.write_typedefs_to_source()
            gen.write_temp_to_source_and_destroy_temp()
            file = open(self.filename + '.c', 'w')
            file.write(gen.produced_source)
            file.close()
            time_end = time.time() - time_start
            print('c source generated in {0}s speed: {1} lines/sec'.format(time_end,
                                                                           p.total_lines/time_end))
            self.call_cc()
        except RuntimeError as e:
            print(e, file=sys.stderr)
            gen.pop_temp()
            gen.write_typedefs_to_source()
            gen.write_temp_to_source_and_destroy_temp()
            file = open(self.filename + '.c', 'w')
            file.write(gen.produced_source)
            file.close()

    def call_as(self):
        cmd = (r'''gcc -o {0}.exe {0}.s''' + \
               ''r' -lglfw3 -lgdi32 -lopengl32''').format(self.filename)
        print('Running assembler: ' + cmd)
        os.system(cmd)

    def call_cc(self):
        cmd = (r'''gcc -o {0}.exe {0}.c''' + \
               ''r' -lglfw3 -lgdi32 -lopengl32''').format(self.filename)
        print('Running C compiler: ' + cmd)
        os.system(cmd)
