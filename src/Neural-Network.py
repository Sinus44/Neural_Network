import random as rnd

class Neuron():
	def __init__(self):
		self.sum = 0
		self.out = 0
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
		for i, inp in enumerate(self.multiArr[0]):
			inp.out = inputs[i]

		i = 0
		# для каждого слоя кроме входного:
		for i1, layer in enumerate(self.multiArr[1:]):
			i = i1 + 1
			# для каждого нейрона слоя:
			for j, neuron in enumerate(layer):
				neuron.sum = 0
				# для каждого нейрона пред.слоя:
				for k, prevNeuron in enumerate(self.multiArr[i-1]):
					neuron.sum += prevNeuron.out * self.w[i-1][k][j]
				neuron.out = self.activation(self.multiArr[i][j].sum)

		res = []
		for neuron in self.multiArr[-1]:
			res.append(neuron.out)

		return res

	def activation(self, x):
		return 1 / (1 + (2 ** (-x)))
	
	def learn(self, data, out):
		res = self.predict(data)

		for i, mi in enumerate(self.multiArr[-1]):
			mi.delta = (out[i] - res[i]) * self.derivativeActivation(res[i])
		
		for i in range(len(self.multiArr) - 2, -1, -1):
			for j, maij in enumerate(self.multiArr[i]):
				s = 0
				for k, mik in enumerate(self.multiArr[i+1]):
					s += self.w[i][j][k] * mik.delta

				delta = self.derivativeActivation(maij.out) * s
				maij.delta = delta

		for i, wi in enumerate(self.w):
			for j, wij in enumerate(wi):
				for k, wijk in enumerate(wij):
					deltaW = self.learnRate * self.multiArr[i][j].out * self.multiArr[i+1][k].delta
					self.w[i][j][k] += deltaW + self.prevW[i][j][k] * self.moment
					self.prevW[i][j][k] = deltaW

	def exp(self, x):
		return NeuronNet.e ** x

	def derivativeActivation(self, x):
		return x * (1 - x)