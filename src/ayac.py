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
type Vector<T> {
    data: []T
    size: int 
    capacity:int
}
impl Vector<T> {
    fn init()->void{
        self.capacity = 10
        self.size = 0
        self.data = malloc(sizeof T * self.capacity ) as  []T
    }
    fn reserve(i:int)->void{
        if self.size + 1 >= self.capacity {
            let temp:[] T
            temp = malloc(sizeof T * self.capacity * 2) as  []T
            self.capacity *= 2
            let i = 0
            while i< self.size {
                temp[i] = self.data[i]
                i += 1
            }
            self.data = temp
            free(self.data)
            self.data = temp
        }
    }
    fn append(x:T)->void{
        self.reserve(self.size + 1)
        self.data[self.size] = x
        self.size += 1
    }
    fn get(idx:int)->T{
        return self.data[idx]
    }
    fn print()->void{
        let i = 0
        while i< self.size{
            printf("%d ", self.get(i))
            i += 1
        }
    }
}
fn main()->int{
 //   let v2:Vector<int>
 //   let v:Vector<Vector<int> >
    let v3:Vector<Vector<Vector<int>>>
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
