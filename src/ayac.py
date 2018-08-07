import compile
import sys


def test():
    s = '''
import {
    // C interop
    cheader "stdio.h"
    cheader "malloc.h"
    cdef printf:([]char,...)->int
    cdef malloc:(int)->*void
    cdef free:(...)->void
}
fn foo<T>(x:T,y:T)->T{
    return x + y
}
fn main()->int{
    foo<int>(1,2.0 as int)
}
    '''
    compiler = compile.Compiler()
    compiler.compile(s)


def replace_ext_with(s: str, ext: str, new_ext: str) -> str:
    s = s.replace(ext, new_ext)
    # print(s)
    return s


def remove_ext(s: str, ext: str) -> str:
    s = s.replace(ext, '')
    # print(s)
    return s


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) == 1:
        test()
    else:
        for i in argv[1:]:
            filename = i
            file = open(filename, 'r')
            compiler = compile.Compiler()
            compiler.compile(file.read(), filename=remove_ext(filename, '.aya'))
            file.close()
