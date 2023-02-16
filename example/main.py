from Engine import *

from NeuralNetwork import NeuralNetwork

filenames = ["dataSet_v1.txt"]
datastrings = []

for filename in filenames:
	file = open(filename, "r")
	datastrings += file.read().split("\n")
	file.close()

data = []

for i in range(10):
	data.append([])

for i in datastrings:
	if i != "":
		num = int(i[0]) # Цифра для обучения
		#print(f"Цифро: {num}")
		data[num].append(list(map(int, list(i[1:]))))

nn = NeuralNetwork(200, [20, 20], 10, 0.4)

epochs = 100

for epoch in range(epochs):
	print(f"Epoch: {epoch} / {epochs}")
	
	for num, examples in enumerate(data):
		out = [0 for _ in range(10)]
		out[num] = 1
				
		for _, example in enumerate(examples):
			#print(f"Time: {Performance.function(lambda: nn.learn(example, out))[0]}")
			#print(f"Нейронка учит цифру {num}")
			
			nn.learn(example, out)

W = 20
H = 10

Output.init()
Output.title("Paint Console")
Output.resize(W, H)

Input.init(
	useHotkey=True,
	mouseEvents=True,
	extended=True
)

screen = Window(W, H)
style = Style()

frame = Frame(screen, style)
frame.draw()
prev = None

def write(num):
	file = open("data.txt", "a")
	file.write(f"{num} {''.join([''.join(['1' if (screen.buffer[i][j] == '*') else '0' for j in range(len(screen.buffer[i]))]) for i in range(len(screen.buffer))])}\n")
	file.close()



def predict():

	data = []
	for i in range(len(screen.buffer)):
		for j in range(len(screen.buffer[i])):
			data.append(1 if (screen.buffer[i][j] == '*') else 0)

	out = nn.predict(data)

	maxVal = 0
	maxNum = -1
	for i in range(len(out)):
		if out[i] > maxVal:
			maxVal = out[i]
			maxNum = i

	Output.title(f"{maxNum} ({maxVal})")

while True:
	Input.tick()

	if Input.eventType == Input.Types.Mouse:
		if Input.mouseKey == Input.Mouse.LEFT:
			if prev != None:
				screen.line(Input.mouseX, Input.mouseY, prev[0], prev[1])
	
	prev = [Input.mouseX, Input.mouseY]

	if Input.keyboardCode == Input.Keyboard.Keys.SPACE:
		frame.draw()

	if Input.eventType == Input.Types.Keyboard:
		if Input.keyboardState == Input.Keyboard.DOWN and not Input.prevKeyboardState:
			if Input.keyboardChar == "0":
				write('0')
			if Input.keyboardChar == "1":
				write('1')
			if Input.keyboardChar == "2":
				write('2')
			if Input.keyboardChar == "3":
				write('3')
			if Input.keyboardChar == "4":
				write('4')
			if Input.keyboardChar == "5":
				write('5')
			if Input.keyboardChar == "6":
				write('6')
			if Input.keyboardChar == "7":
				write('7')
			if Input.keyboardChar == "8":
				write('8')
			if Input.keyboardChar == "9":
				write('9')

	predict()

	screen.draw()