from NeuralNetwork import *

if __name__ == "__main__":
	loading = True

	if loading:
		n = NeuralNetwork.load("nn.sw")
	else:
		n = NeuralNetwork(2, [2], 1, 5, True, 0)

	ds = [
		[[0, 0], [0]],
		[[0, 1], [1]],
		[[1, 0], [1]],
		[[1, 1], [0]],
	]

	print("До обучения: ")
	for value in ds:
		print(n.predict(value[0]))

	n.learn_by_error(ds, 0.001)

	print("После обучения: ")
	for value in ds:
		print(n.predict(value[0]))

	if not loading:
		n.save()

	enable = True
	while enable:
		x1 = float(input("Введите значение 1 (1/0): "))
		x2 = float(input("Введите значение 2 (1/0): "))

		print(f"Нейросеть дала ответ: {n.predict([x1, x2])}")
		if input(f"Повторить запрос? (yes/no): ").upper() != "YES":
			enable = False
