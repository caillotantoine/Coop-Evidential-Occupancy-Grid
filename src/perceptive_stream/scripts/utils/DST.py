#!/usr/bin/env python3
# coding: utf-8

class DST:
    def __init__(self, listOfmasses):
        self.masses = listOfmasses

    def get_masses(self):
        return self.masses

    def get_mass(self, set):
        for (key, val) in self.masses:
            if key == set:
                return val
        return None

    def sum(self, B, set = None, normalized = True, debug = False):
        Bmasses = B.get_masses()
        K = 0
        if normalized:
            for i in range(len(self.masses)):
                for j in range(len(Bmasses)):
                    if len(self.masses[i][0].intersection(Bmasses[j][0])) == 0:
                        if debug:
                            print("K+= ", self.masses[i], " * ", Bmasses[j])
                        K += self.masses[i][1] * Bmasses[j][1]

        if set != None:
            return 1.0 / (1.0 - K) * self.rawSum(B, set, debug=debug)
        else:
            out = []
            for (set, _) in self.masses:
                out.append((set, self.rawSum(B, set, debug=debug)))
            return DST(out)
                

    def rawSum(self, B, set, debug = False):
        Bmasses = B.get_masses()
        rawSum = 0
        for i in range(len(self.masses)):
            for j in range(len(Bmasses)):
                if self.masses[i][0].intersection(Bmasses[j][0]) == set:
                    if debug:
                        print("rawSum += ", self.masses[i], " * ", Bmasses[j])
                    rawSum += self.masses[i][1] * Bmasses[j][1]
        return rawSum
        

    def bel(self, set, debug = False):
        b = 0
        for i in range(len(self.masses)):
            if self.masses[i][0].issubset(set):
                if debug:
                    print("Add : ", self.masses[i])
                b += self.masses[i][1]
        return b

    def pl(self, set, debug = False):
        b = 0
        for i in range(len(self.masses)):
            if len(self.masses[i][0].intersection(set)) > 0:
                if debug:
                    print("Add : ", self.masses[i])
                b += self.masses[i][1]
        return b
        

if __name__ == '__main__':
    A = [({"O"}, 0.1), ({"F"}, 0.1), ({"O", "F"}, 0.8)]
    B = [({"O"}, 0.1), ({"F"}, 0.6), ({"O", "F"}, 0.3)]
    # twoPowUniv.add(""univ)
    Adst = DST(A)
    Bdst = DST(B)

    print("belA(O) = ", Adst.bel({"O"}, debug=True))
    print("plA(O) = ", Adst.pl({"O"}, debug=True))
    print("belA(F) = ", Adst.bel({"F"}, debug=True))
    print("plA(F) = ", Adst.pl({"F"}, debug=True))
    print("belB(O) = ", Bdst.bel({"O"}, debug=True))
    print("plB(O) = ", Bdst.pl({"O"}, debug=True))
    print("belB(F) = ", Bdst.bel({"F"}, debug=True))
    print("plB(F) = ", Bdst.pl({"F"}, debug=True))

    sumDST = Adst.sum(Bdst)
    print("Adst : ", Adst.get_masses(), "\nBdst : ", Bdst.get_masses(), "\nSum : ", sumDST.get_masses())
    print("bel_sum(O) = ", sumDST.bel({"O"}, debug=True))
    print("pl_sum(O) = ", sumDST.pl({"O"}, debug=True))
    print("bel_sum(F) = ", sumDST.bel({"F"}, debug=True))
    print("pl_sum(F) = ", sumDST.pl({"F"}, debug=True))