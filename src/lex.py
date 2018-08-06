import re


class Token:
    def __init__(self, tok: str, type: str, line, col):
        self.tok = tok
        self.line = line
        self.col = col
        self.type = type

    def __repr__(self):
        return '<{0}> {1} {2} {3}'.format(self.tok, self.type, self.line, self.col)

    def copy(self):
        if self.tok:
            s = self.tok[0:-1] + self.tok[-1]
        else:
            s = ''
        n = Token(s,self.type,self.line,self.col)
        return n


nil_token = Token('', 'nil', 0, 0)


class Lexer:
    def __init__(self, src: str):
        self.line = 1
        self.col = 1
        self.line_comment = re.compile('//.+')
        self.token_stream = []
        self.src = src  # type : str
        self.number = re.compile('[0-9]+(\.[0-9]+)*f*')
        self.match = None
        self.token = None  # type : str
        self.symbol = '''& : , ( ) [ ] { } = + - * += ... ''' + \
                      '''! += -= *= /= %= >>= <<= /  % ^ > < >= <= == != -> . >> <<'''
        self.symbol = [x for x in self.symbol.split(' ') if x]
        self.symbol.sort(key=lambda x: -len(x))
        self.identifier = re.compile('[_A-Za-z]*([_A-Za-z0-9])+')

    def advance(self, i:int):
        while i > 0:
            if self.src[0] == '\n':
                self.line += 1
                self.col = 1
            elif self.src[0] == '\t':
                self.col += 4
            else:
                self.col += 1
            self.src = self.src[1:]
            i -= 1

    def get_match(self):
        if self.match:
            self.token = self.src[self.match.span()[0]:self.match.span()[1]]
            # self.src = self.src[len(self.token):]
            self.advance(len(self.token))
        return self.match

    def has_number(self):
        self.match = self.number.match(self.src)
        return self.get_match()

    def has_identifier(self):
        self.match = self.identifier.match(self.src)
        return self.get_match()

    def has_symbol(self):
        for i in self.symbol:
            if self.src.startswith(i):
                self.token = i
                #self.src = self.src[len(i):]
                self.advance(len(i))
                return True
        return False

    def has_char(self):
        if self.src.startswith("\'"):
            start = 0
            end = 1
            while self.src[end] != '\'':
                end += 1
            self.token = self.src[start:end + 1]
            # self.src = self.src[len(self.token):]
            self.advance(len(self.token))
            self.token = str(ord(self.token))
            return True
        else:
            return False

    def has_string(self):
        if self.src.startswith("\""):
            start = 0
            end = 1
            while self.src[end] != '\"':
                end += 1
            self.token = self.src[start:end + 1]
            # self.src = self.src[len(self.token):]
            self.advance(len(self.token))
            return True
        else:
            return False

    def skip_comment(self):
        match = self.line_comment.match(self.src)
        if match:
            self.advance(match.span()[1])

    def skip_space(self):
        try:
            self.skip_comment()
            while self.src[0].isspace():
                # self.src = self.src[1:]
                self.advance(1)
                self.skip_comment()
        except IndexError:
            pass

    def append_token(self, type):
        token = Token(self.token, type, self.line, self.col)
        self.token_stream.append(token)

    def next(self):
        self.skip_space()
        if self.has_number():
            self.append_token('number')
        elif self.has_identifier():
            self.append_token('identifier')
        elif self.has_symbol():
            self.append_token('symbol')
        elif self.has_string():
            self.append_token('string')
        elif self.has_char():
            self.append_token('number')
        else:
            raise RuntimeError('wtf ' + self.src)

    def parse(self):
        while self.src:
            self.next()
            self.skip_space()


def test():
    s = '1 + 1'
    lex = Lexer(s)
    lex.parse()
    print(lex.token_stream)


if __name__ == '__main__':
    test()
