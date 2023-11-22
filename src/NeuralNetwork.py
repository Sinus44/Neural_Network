# NeuralNetwork_v2.0.0 Stable

import random as rnd
import matplotlib.pyplot as plt
import os
import json

class Neuron:
	def __init__(self, is_bias=False):
		self.sum = 0
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

		learn_rate = structure["r"]
		use_bias = structure["b"]
		moment = structure["m"]

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
					self.w[i][j].append(rnd.random())

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

				neuron.sum = sum([lastNeuron.out * self.w[i][k][j] for k, lastNeuron in enumerate(self.array[i])])
				neuron.out = self.activation(neuron.sum)

		# Возврат результата
		return [neuron.out for neuron in self.array[-1]]
	
	def learn(self, data, out):
		res = self.predict(data)
		errors = []

		# Поиск дельт последнего слоя
		for i, mi in enumerate(self.array[-1]):
			errors.append((out[i] - res[i]) ** 2)
			mi.delta = (out[i] - res[i]) * self.derivativeActivation(res[i])
		
		for i in range(len(self.array) - 2, -1, -1):
			for j, maij in enumerate(self.array[i]):
				s = 0
				for k, mik in enumerate(self.array[i+1]):
					s += self.w[i][j][k] * mik.delta

				delta = self.derivativeActivation(maij.out) * s
				maij.delta = delta

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
		errs = []
		for i in range(iterations):
			mse_list = []
			for one_set in dataset:
				mse_list += nn.learn(one_set[0], one_set[1])

			err = (1 / len(mse_list)) * sum(mse_list)
			if use_error_history:
				errs.append(err)
		
		return err

	def learn_by_error(self, dataset, max_error, max_iterations=10_000, use_error_history=False):
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
			"b": self.use_bias
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

		plt.title('Ошибка')
		plt.xlabel('Итерация')
		plt.ylabel('Ошибка')

		return plt

	def show(error_list):
		Graph.save(error_list)
		os.system("errors.png")

	def save(error_list):
		Graph.create(error_list).savefig("errors.png")

if __name__ == "__main__":
	# Создание нейросети
	nn = NeuralNetwork(2, [2], 1, 1, True, 0.9)

	# Загрузка нейросети из файла
	nn = NeuralNetwork.load()

	# Создание датасета
	dataset = [
		[[0, 0], [0]],
		[[0, 1], [1]],
		[[1, 0], [1]],
		[[1, 1], [0]]
	]

	# Вывод значений до обучения
	print("+" * 10 + " BEFORE " + "+" * 10)
	for one_set in dataset:
		print(nn.predict(one_set[0]))

	# Обучение нейросети пока значение ошибки не будет ниже указанного
	e = nn.learn_by_error(dataset, 0.01, max_iterations=100_000, use_error_history=True)
	
	# Показ графика обучения
	Graph.show(e)

	# Вывод значений после обучения
	print("+" * 10 + " AFTER " + "+" * 10)
	for one_set in dataset:
		print(nn.predict(one_set[0]))

	# Сохранение нейросети в файл
	nn.save()