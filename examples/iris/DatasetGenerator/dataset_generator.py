import sys

replace_list = (
	("setosa", "[1.0, 0.0, 0.0]"),
	("versicolor", "[0.0, 1.0, 0.0]"),
	("virginica", "[0.0, 0.0, 1.0]")
)

save_as_json = True
save_as_python = False

file_path = ""

if len(sys.argv) > 1:
	file_path = sys.argv[1]

else:
	file_path = "data.txt"

file_name = ".".join(file_path.split(".")[:-1]) 
output_file_path = file_name + ".ds"

raw_data = ""
with open(file_path, "r") as file:
	raw_data = file.read()

print(raw_data)

for replace_rule in replace_list:
	raw_data = raw_data.replace(replace_rule[0], replace_rule[1])

print(raw_data)

result_data = ""
for line in raw_data.split("\n"):
	values = line.split("\t")
	#print(values)
	result_data += f"\t{[[float(element) for element in values[:-1]], eval(values[-1])]},\n"

result_array = result_data[:-2]
result_array = "[\n" + result_array + "\n]"

if save_as_json:
	result_data = result_array

elif save_as_python:
	result_data = f"data_set = {result_array}" 

with open(output_file_path, "w") as file:
	file.write(result_data)

print(result_data)