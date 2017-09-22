This is an example of using bidirection lstm to sort an array.

Firstly, generate data by:

	python gen_data.py

Move generated txt files to data directory

	mkdir data
	mv *.txt data

Then, train the model by:

    python lstm_sort.py

At last, test model by:

    python infer_sort.py 234 189 785 763 231

and will output sorted seq

    189
	231
	234
	763
	785


