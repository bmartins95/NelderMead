import numpy as np

class NelderMead:
    def __init__(self, f_variables, alpha=1.0, gamma=2.0, beta=0.5, delta=0.5, use_shrink=False):
        if not (isinstance(f_variables, list) or isinstance(f_variables, np.ndarray)):
            raise TypeError("f_variables should be numpy.ndarray or list, i.e. mutable!")
        else:
            self.f_var = f_variables

        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.delta = delta
        self.use_shrink = use_shrink
        self.f_values = np.zeros(len(f_variables) + 1)
        self.gather_calls = 0
        self.current_phase = "gather"

        self.buildSimplexPoints()

    def run(self, f_value):
        if self.gather_calls < len(self.f_var):
            self.gatherRoutine(f_value)
        else:
            self.optimizeRoutine(f_value)

    def gatherRoutine(self, f_value):
        if self.gather_calls == 0:
            self.f_values[-1] = f_value
        else:
            self.f_values[self.gather_calls-1] = f_value

        for index,value in enumerate(self.simplex[self.gather_calls]):
            self.f_var[index] = value
        self.gather_calls += 1

    def optimizeRoutine(self, f_value):
        if self.gather_calls == len(self.f_var):
            self.f_values[self.gather_calls-1] = f_value
            self.sort()
            self.computeCentroid()
            self.gather_calls += 1
            self.current_phase = "pre_reflection"

        if self.current_phase == "pre_reflection":
            self.reflection()
        elif self.current_phase == "reflection":
            if f_value < self.f_values[-2] and f_value >= self.f_values[0]:
                self.simplex[-1,:] = self.x_r
                self.f_values[-1] = f_value
                self.sort()
                self.computeCentroid()
                self.reflection()
            elif (f_value < self.f_values[0]):
                self.f_reflection = f_value
                self.expansion()
            else:
                self.contraction(f_value)
        elif self.current_phase == "expansion":
            if f_value < self.f_reflection:
                self.simplex[-1,:] = self.x_e
                self.f_values[-1] = f_value
            else:
                self.simplex[-1,:] = self.x_r
                self.f_values[-1] = self.f_reflection
            self.sort()
            self.computeCentroid()
            self.reflection()
        elif self.current_phase == "contraction":
            if f_value < self.f_values[-1] or not self.use_shrink:
                self.simplex[-1,:] = self.x_c
                self.f_values[-1] = f_value
                self.sort()
                self.computeCentroid()
                self.reflection()
            else:
                self.shrink()

    def buildSimplexPoints(self):
        self.simplex = np.vstack([np.eye(len(self.f_var), dtype = float), self.f_var])
        for index, value in enumerate(self.f_var):
            h = 0.00025 if abs(value) < 1.0e-22 else 0.05
            self.simplex[index,:] = self.simplex[index,:] * h + value

    def sort(self):
        indexes = np.argsort(self.f_values)
        self.f_values = sorted(self.f_values)
        self.simplex = self.simplex[indexes,:]

    def computeCentroid(self):
        self.c = (self.simplex[:-1,:]).mean(axis=0)

    def reflection(self):
        self.current_phase = "reflection"
        self.x_r = self.c + self.alpha*(self.c - self.simplex[-1,:])
        for index, value in enumerate(self.x_r):
            self.f_var[index] = value

    def expansion(self):
        self.current_phase = "expansion"
        self.x_e = self.c + self.gamma*(self.x_r - self.c)
        for index,value in enumerate(self.x_e):
            self.f_var[index] = value

    def contraction(self, f_value):
        self.current_phase = "contraction"
        if f_value >= self.f_values[-1]:
            self.x_c = self.c + self.beta*(self.simplex[-1,:] - self.c)
        else:
            self.x_c = self.c + self.beta*(self.x_r - self.c)
        for index, value in enumerate(self.x_c):
            self.f_var[index] = value

    def shrink(self):
        self.current_phase = "gather"
        self.gather_calls = 1
        for index in range(1 , self.simplex.shape[1]+1):
            self.simplex[index,:] = self.simplex[0,:] + self.delta*(self.simplex[index,:] - self.simplex[0,:])

        self.simplex = np.flip(self.simplex, 0)
        self.f_values = np.flip(self.f_values)
        for index, value in enumerate(self.simplex[0]):
            self.f_var[index] = value
