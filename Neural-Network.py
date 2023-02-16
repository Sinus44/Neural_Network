import random as rnd

class Neuron():
    def __init__(self):
        self.sum = 0
        self.out = 0
        self.err = 0
        self.delta = 0

class NeuralNetwork:
    def __init__(self, inputCount, hidenLayers, outputs, learnRate, moment):
        self.learnRate = learnRate
        self.moment = moment

        # Инициализация входного слоя
        inputs = []

        for i in range(inputCount):
            inputs.append(Neuron())

        # Инициализация внутрених слоев
        self.hidenLayers = []
        for i in range(len(hidenLayers)):
            self.hidenLayers.append([])
            for j in range(hidenLayers[i]):
                self.hidenLayers[i].append(Neuron())
        
        # Инициализация выходного слоя
        output = []
        for i in range(outputs):
            output.append(Neuron())
        
        # Инициализация весов
        self.multiArr = [inputs] + self.hidenLayers + [output]

        self.w = []
        for i in range(len(self.multiArr) - 1):
            self.w.append([])
            for j in range(len(self.multiArr[i])):
                self.w[i].append([])
                for k in range(len(self.multiArr[i+1])):
                    self.w[i][j].append(rnd.random()*2-1)

        self.prevW = []
        for i in range(len(self.multiArr) - 1):
            self.prevW.append([])
            for j in range(len(self.multiArr[i])):
                self.prevW[i].append([])
                for k in range(len(self.multiArr[i+1])):
                    self.prevW[i][j].append(0)
    
    def predict(self, inputs):
        # Вставка входных данных
        for i in range(len(self.multiArr[0])):
            self.multiArr[0][i].out = inputs[i]

        # для каждого слоя кроме входного:
        for i in range(1, len(self.multiArr)):
            # для каждого нейрона слоя:
            for j in range(len(self.multiArr[i])):
                self.multiArr[i][j].sum = 0
                # для каждого нейрона пред.слоя:
                for k in range(len(self.multiArr[i-1])):
                    self.multiArr[i][j].sum += self.multiArr[i-1][k].out * self.w[i-1][k][j]
                self.multiArr[i][j].out = self.activation(self.multiArr[i][j].sum)

        res = []
        for neuron in self.multiArr[-1]:
            res.append(neuron.out)

        return res

    def activation(self, x):
        return 1 / (1 + (2.71 ** (-x)))
    
    def learn(self, data, out):
        res = self.predict(data)

        for i in range(len(self.multiArr[-1])):
            self.multiArr[-1][i].delta = (out[i] - res[i]) * self.derivativeActivation(res[i])
        
        for i in range(len(self.multiArr) - 2, -1, -1):
            for j in range(len(self.multiArr[i])):
                s = 0 
                for k in range(len(self.multiArr[i+1])):
                    s += self.w[i][j][k] * self.multiArr[i+1][k].delta

                delta = self.derivativeActivation(self.multiArr[i][j].out) * s
                self.multiArr[i][j].delta = delta

        for i in range(len(self.w)):
            for j in range(len(self.w[i])):
                for k in range(len(self.w[i][j])):
                    deltaW = self.learnRate * self.multiArr[i][j].out * self.multiArr[i+1][k].delta
                    self.w[i][j][k] += deltaW + self.prevW[i][j][k] * self.moment
                    self.prevW[i][j][k] = deltaW

    def exp(self, x):
        return NeuronNet.e ** x

    def derivativeActivation(self, x):
        return x * (1 - x)