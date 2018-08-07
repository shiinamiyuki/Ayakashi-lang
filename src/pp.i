# 1 "pp.c"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "pp.c"
# 1 "example.aya" 1
import {

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
        self.data = malloc(sizeof T * self.capacity ) as []T
    }
    fn reserve(i:int)->void{
        if self.size + 1 >= self.capacity {
            let temp:[] T
            temp = malloc(sizeof T * self.capacity * 2) as []T
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
fn foo<T>(x:T, y:T)->void{
}
fn make_vec<T>()->Vector<T>{
    let v: Vector<T>
    v.init()
    return v
}
fn reverse_vec<T>(v:Vector<T>)->Vector<T>{
    let v2 = make_vec<T>()
    let i = v.size - 1
    while i >= 0 {
        v2.append(v.get(i))
        i -= 1
    }
    return v
}
type Boo<T>{
    x:T
}
fn make_Boo<T>(x:T)->Boo<T>{
    let b:Boo<T>
    b.x = x
    return b
}
fn blow<E>(i:int,x:E)->E{
    if i == 0 {
        return x
    }else{
        return blow<Boo<E> >(i - 1, make_Boo(x)).x
    }
}
type Pair<T,U>{
    first:T
    second:U
}
# 2 "pp.c" 2
fn main()->int{
    foo(2,2)
    foo(2, "a")
    let v = make_vec<int>()
    v.append(1)
    v.append(3)
    v.print()
    reverse_vec(v)
}
