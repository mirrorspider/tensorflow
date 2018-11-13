from random import randint
import numpy as np

class CardImage:

    # define colours
    WHITE_CLR = [255, 255, 255]
    RED_CLR = [255, 0, 0]
    BLACK_CLR = [0, 0, 0]


    def __init__(self):
        self.img = [[self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR],
                    [self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR],
                    [self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR],
                    [self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR],
                    [self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR],
                    [self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR],
                    [self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR]]
        self.rank = 0
                    
    def get_image(self):
        return np.array(self.img)
    
    def get_rank(self):
        # rank goes low to high slightly in defiance of the random choice approach
        return self.rank

class Heart(CardImage):
    def __init__(self):
        self.img = [[self.WHITE_CLR, self.RED_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.RED_CLR, self.WHITE_CLR],
                    [self.RED_CLR, self.RED_CLR, self.RED_CLR, self.WHITE_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR],
                    [self.RED_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR],
                    [self.RED_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR],
                    [self.WHITE_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.WHITE_CLR],
                    [self.WHITE_CLR, self.WHITE_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.WHITE_CLR, self.WHITE_CLR],
                    [self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.RED_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR]]
        self.rank = 1

class Diamond(CardImage):
    def __init__(self):
        self.img = [[self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.RED_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR],
                    [self.WHITE_CLR, self.WHITE_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.WHITE_CLR, self.WHITE_CLR],
                    [self.WHITE_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.WHITE_CLR],
                    [self.RED_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR],
                    [self.WHITE_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.WHITE_CLR],
                    [self.WHITE_CLR, self.WHITE_CLR, self.RED_CLR, self.RED_CLR, self.RED_CLR, self.WHITE_CLR, self.WHITE_CLR],
                    [self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.RED_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR]]
        self.rank = 2

class Club(CardImage):
    def __init__(self):
        self.img = [[self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.BLACK_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR],
                    [self.WHITE_CLR, self.WHITE_CLR, self.BLACK_CLR, self.BLACK_CLR, self.BLACK_CLR, self.WHITE_CLR, self.WHITE_CLR],
                    [self.WHITE_CLR, self.BLACK_CLR, self.WHITE_CLR, self.BLACK_CLR, self.WHITE_CLR, self.BLACK_CLR, self.WHITE_CLR],
                    [self.BLACK_CLR, self.BLACK_CLR, self.BLACK_CLR, self.BLACK_CLR, self.BLACK_CLR, self.BLACK_CLR, self.BLACK_CLR],
                    [self.WHITE_CLR, self.BLACK_CLR, self.WHITE_CLR, self.BLACK_CLR, self.WHITE_CLR, self.BLACK_CLR, self.WHITE_CLR],
                    [self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.BLACK_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR],
                    [self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.BLACK_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR]]
        self.rank = 3

class Spade(CardImage):
    def __init__(self):
        self.img = [[self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.BLACK_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR],
                    [self.WHITE_CLR, self.WHITE_CLR, self.BLACK_CLR, self.BLACK_CLR, self.BLACK_CLR, self.WHITE_CLR, self.WHITE_CLR],
                    [self.WHITE_CLR, self.BLACK_CLR, self.BLACK_CLR, self.BLACK_CLR, self.BLACK_CLR, self.BLACK_CLR, self.WHITE_CLR],
                    [self.BLACK_CLR, self.BLACK_CLR, self.BLACK_CLR, self.BLACK_CLR, self.BLACK_CLR, self.BLACK_CLR, self.BLACK_CLR],
                    [self.BLACK_CLR, self.BLACK_CLR, self.BLACK_CLR, self.BLACK_CLR, self.BLACK_CLR, self.BLACK_CLR, self.BLACK_CLR],
                    [self.BLACK_CLR, self.BLACK_CLR, self.WHITE_CLR, self.BLACK_CLR, self.WHITE_CLR, self.BLACK_CLR, self.BLACK_CLR],
                    [self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR, self.BLACK_CLR, self.WHITE_CLR, self.WHITE_CLR, self.WHITE_CLR]]
        self.rank = 0

class Scuffer:
    def halve(c):
        out = np.divide(c, 2)
        return out
        
    def flip_colour(c):
        out = c
        for i in range(7):
            for j in range(7):
                if (out[i,j]==CardImage.BLACK_CLR).all():
                    out[i,j] = CardImage.RED_CLR
                elif (out[i,j]==CardImage.RED_CLR).all():
                    out[i,j] = CardImage.BLACK_CLR
        return out
        
    def add_noise(c):
        out = c
        with np.nditer(out, op_flags=['readwrite']) as it:
            for x in it:
                u = int(randint(1,16))
                if u < 15:
                    x[...] = x
                elif u == 15:
                    x[...] = x + int(randint(1,255))
                else:
                    x[...] = x - int(randint(1,255))
        out = np.clip(out, 0, 255)
        return out
        
    def scuff(c):
        u = int(randint(1,10))
        if u < 8:
            out = c
        elif u == 8:
            out = Scuffer.halve(c)
        elif u == 9:
            out = Scuffer.flip_colour(c)
        else:
            out = Scuffer.add_noise(c)
            
        return out
        
class Pack:
    def get_suits(abbrev = False):
        sts = ["Spade", "Heart", "Diamond", "Club"]
        if abbrev:
            sts = ["S", "H", "D", "C"]
          
        return sts

    def get_card():
        x = int(randint(1,4))
        if x == 1:
            ci = Club()
        elif x == 2:
            ci = Diamond()
        elif x == 3:
            ci = Heart()
        else:
            ci = Spade()
        return ci
            
    def shuffle(n = 9, dirty = False):
        cards = np.empty([n, 7, 7, 3], dtype="int")
        ranks = np.empty(n, dtype="int")
        for i in range(0,n):
            c = Pack.get_card()
            if dirty:
                cards[i] = Scuffer.scuff(c.get_image())
            else:
                cards[i] = c.get_image()
            ranks[i] = c.get_rank()
        return [cards, ranks]

if __name__ == "__main__":
    i = Pack.get_card()
    print(i.get_image())