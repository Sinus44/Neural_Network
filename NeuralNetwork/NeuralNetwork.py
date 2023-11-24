# NeuralNetwork_v2.2.0 Stable

import random as rnd
import matplotlib.pyplot as plt
import os
import json

class Neuron:
	def __init__(self, is_bias=False):
		self.out = int(is_bias)
		self.delta = 0
		self.is_bias = is_bias

class NeuralNetwork:

	@staticmethod
	def create_by_structure(structure):
		layers = structure["l"]
		input_count = layers[0]
		hidden_layers = layers[1:-1]
		output_count = layers[-1]

		learn_rate = structure.get("r")
		use_bias = structure.get("b")
		moment = structure.get("m")

		return NeuralNetwork(input_count, hidden_layers, output_count, learn_rate, use_bias, moment)

	@staticmethod
	def load(name="nn.sw"):
		data = ''
		with open(name, "r") as file:
			data = file.read()

		data = json.loads(data)
		neural_network = NeuralNetwork.create_by_structure(data["s"])
		neural_network.w = data["w"]
		return neural_network 

	def __init__(self, input_count, hiden_layers, output_count, learn_rate=0.1, use_bias=True, moment=0.9):
		self.learn_rate = learn_rate
		self.moment = moment
		self.layers = [input_count] + hiden_layers + [output_count]
		self.use_bias = use_bias

		# Инициализация внутрених слоев
		self.hiden_layers = []
		for i in range(len(hiden_layers)):
			self.hiden_layers.append([])
			for j in range(hiden_layers[i]):
				self.hiden_layers[i].append(Neuron())
			
			if use_bias:
				self.hiden_layers[i].append(Neuron(True))

		# Массив нейронов
		self.array = [[Neuron() for i in range(input_count)] + ([Neuron(True)] if use_bias else [])] + self.hiden_layers + [[Neuron() for _ in range(output_count)]]
		self.reset_weights()

	def reset_weights(self):
		# Инициализация весов
		self.w = []
		for i in range(len(self.array) - 1):
			self.w.append([])
			for j in range(len(self.array[i])):
				self.w[i].append([])
				for k in range(len(self.array[i+1])):
					self.w[i][j].append(rnd.random() * 2 - 1)

		# Инициализация весов
		self.prev_w = []
		for i in range(len(self.array) - 1):
			self.prev_w.append([])
			for j in range(len(self.array[i])):
				self.prev_w[i].append([])
				for k in range(len(self.array[i+1])):
					self.prev_w[i][j].append(0)
	
	def predict(self, inputs):
		# Вставка входных данных
		for neurons, inp in zip(self.array[0], inputs):
			neurons.out = inp

		# Расчет суммы и активация
		for i, layer in enumerate(self.array[1:]):
			for j, neuron in enumerate(layer):
				if neuron.is_bias:
					continue

				s = sum([lastNeuron.out * self.w[i][k][j] for k, lastNeuron in enumerate(self.array[i])])
				neuron.out = self.activation(s)

		# Возврат результата
		return [neuron.out for neuron in self.array[-1]]
	
	def learn(self, data, out):
		res = self.predict(data)
		errors = []

		# Поиск градиентов последнего слоя
		for i, mi in enumerate(self.array[-1]):
			errors.append((out[i] - res[i]) ** 2)
			mi.delta = (out[i] - res[i]) * self.derivativeActivation(res[i])

		# Поиск градиентов других слоев
		for i in range(len(self.array) - 2, -1, -1):
			for j, maij in enumerate(self.array[i]):
				s = 0
				for k, mik in enumerate(self.array[i+1]):
					s += self.w[i][j][k] * mik.delta

				delta = self.derivativeActivation(maij.out) * s
				maij.delta = delta

		# Коррекция весов
		for i, wi in enumerate(self.w):
			for j, wij in enumerate(wi):
				for k, wijk in enumerate(wij):
					deltaW = self.learn_rate * self.array[i][j].out * self.array[i+1][k].delta + self.prev_w[i][j][k] * self.moment
					
					self.w[i][j][k] += deltaW
					self.prev_w[i][j][k] = deltaW

		return errors

	def activation(self, x):
		return 1 / (1 + (2.718281 ** (-x)))

	def derivativeActivation(self, x):
		return x * (1 - x)

	def learn_by_iterations(self, dataset, iterations, use_error_history=False):
		if use_error_history:
			errs = []
			for i in range(iterations):
				mse_list = []
				for one_set in dataset:
					mse_list += self.learn(one_set[0], one_set[1])

				err = (1 / len(mse_list)) * sum(mse_list)
				errs.append(err)
		
			return errs
		else:
			for i in range(iterations):
				for one_set in dataset:
					self.learn(one_set[0], one_set[1])

	def learn_by_error(self, dataset, max_error, max_iterations=100_000, use_error_history=False):
		err = float("inf")
		i = 0
		errs = []
		while err > max_error:
			mse_list = []

			for one_set in dataset:
				mse_list += self.learn(one_set[0], one_set[1])

			err = (1 / len(mse_list)) * sum(mse_list)

			if use_error_history:
				errs.append(err)

			if i > max_iterations:
				break

			i += 1
		else:
			return errs
		
		self.reset_weights()
		return self.learn_by_error(dataset, max_error, max_iterations, use_error_history)

	def save(self, name="nn.sw"):
		structure = {
			"l": self.layers,
			"r": self.learn_rate,
			"m": self.moment,
			"b": self.use_bias,
		}

		saving_data = {
			"s": structure,
			"w": self.w
		}

		with open(name, "w") as file:
			file.write(json.dumps(saving_data))

class Graph:
	def create(error_list):
		plt.plot(error_list, label='Ошибка')
		plt.switch_backend('TkAgg')

		plt.title('Ошибка')
		plt.xlabel('Итерация')
		plt.ylabel('Ошибка')

		return plt

	def show(error_list):
		Graph.create(error_list).show()

	def save(error_list):
		Graph.create(error_list).savefig("errors.png")

class DatasetLoader:
	def load(file_path):
		data = ""
		with open(file_path, "r") as file:
			data = file.read()

		data = json.loads(data)
		return data