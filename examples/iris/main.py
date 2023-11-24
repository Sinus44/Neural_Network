from NeuralNetwork import NeuralNetwork, DatasetLoader, Graph
import os

def create():
	print("Создание нейросети...")

	# Создание нейросети
	nn = NeuralNetwork(4, [10], 3, 0.1, True, 0)

	# Загрузка датасета
	dataset = DatasetLoader.load("iris.ds")

	# Вывод значений до обучения
	before_learning_values = [[round(elem, 1) for elem in nn.predict(one_set[0])] for one_set in dataset]

	print("Нейросеть обучается...")

	# Обучение нейросети по кол-ву итераций
	e = nn.learn_by_iterations(dataset, 10_000, use_error_history=True)

	# Вывод значений после обучения
	after_learning_values = [[round(elem, 1) for elem in nn.predict(one_set[0])] for one_set in dataset]

	# Вывод в консоль
	print("Вход                 | До обучения     | После обучения     | Ожидалось")
	print("---------------------+-----------------+--------------------+----------------")
	[print(f"{i[0]} | {b} | {a} | {i[1]}") for i, b, a in zip(dataset, before_learning_values, after_learning_values)]

	# Поиск несовпадений с ожидаемыми значениями
	miss_count = len([0 for x, y in zip(dataset, after_learning_values) if [round(e) for e in x[1]] != [round(e) for e in y]])
	if miss_count != 0:
		print(f"Несовпало с ожиданием {miss_count} из {len(dataset)}")
	
	# Показ графика обучения
	Graph.show(e)

	if input("Хотите сохранить нейросеть? (yes / no): ").upper() == "YES": 
		nn.save()

	return nn

def chat_get_file_path():
	file_path = input("Укажите путь к файлу (или \"no\" для создания): ")
	if file_path.upper() == "NO":
		return None

	if os.path.isfile(file_path):
		return file_path
	else:
		print("Файл не найден")
		return chat_get_file_path()

def chat_main_menu():
	if input("Хотите использовать нейросеть из файла? (y/n): ").upper() == "Y":
		file_path = chat_get_file_path()
		if file_path == None:
			return create()
		else:
			return NeuralNetwork.load(file_path)
	return create()

if __name__ == "__main__":
	neural_network = chat_main_menu()

	print("Нейросеть готова к использованию")

	enable = True
	while enable:
		inputs = [0, 0, 0, 0]
		
		inputs[0] = float(input(f"Введите длину чашелистника: "))
		inputs[1] = float(input(f"Введите ширину чашелистника: "))
		inputs[2] = float(input(f"Введите длину лепестка: "))
		inputs[3] = float(input(f"Введите ширину лепестка: "))

		result = neural_network.predict(inputs)
		result_sum = sum(result)

		setosa = result[0] / result_sum
		versicolor = result[1] / result_sum
		verginica = result[2] / result_sum

		print(f"Результат нейросети:\nSetosa: {setosa}\nVersicolor: {versicolor}\nVerginica: {verginica}")

		enable = input("Повторить (yes) / выйти (no): ").upper() == "YES"