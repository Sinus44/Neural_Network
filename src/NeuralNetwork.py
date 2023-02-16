# 1_NeuralNetwork_v1.0.0

import random as rnd

class Neuron():
	def __init__(self):
		self.sum = 0
		self.out = 0
		self.delta = 0

class NeuralNetwork:
	def __init__(self, inputCount, hidenLayers, outputs, learnRate):
		self.learnRate = learnRate

		# Инициализация внутрених слоев
		self.hidenLayers = []
		for i in range(len(hidenLayers)):
			self.hidenLayers.append([])
			for j in range(hidenLayers[i]):
				self.hidenLayers[i].append(Neuron())

		# Массив нейронов
		self.multiArr = [[Neuron() for i in range(inputCount)]] + self.hidenLayers + [[Neuron() for _ in range(outputs)]]

		# Инициализация весов
		self.w = []
		for i in range(len(self.multiArr) - 1):
			self.w.append([])
			for j in range(len(self.multiArr[i])):
				self.w[i].append([])
				for k in range(len(self.multiArr[i+1])):
					self.w[i][j].append(rnd.random() * 2 - 1)
	
	def predict(self, inputs):
		# Вставка входных данных
		for neurons, inp in zip(self.multiArr[0], inputs):
			neurons.out = inp

		# Расчет суммы и активация
		for i, layer in enumerate(self.multiArr[1:]):
			for j, neuron in enumerate(layer):
				neuron.sum = sum([lastNeuron.out * self.w[i][k][j] for k, lastNeuron in enumerate(self.multiArr[i])])
				neuron.out = self.activation(neuron.sum)

		# Возврат результата
		return [neuron.out for neuron in self.multiArr[-1]]
	
	def learn(self, data, out):
		res = self.predict(data)

		# Поиск дельт последнего слоя
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
					self.w[i][j][k] += deltaW

	def activation(self, x):
		return 1 / (1 + (2.7182 ** (-x)))

	def derivativeActivation(self, x):
		return x * (1 - x)