This is an example of using bidirection lstm to sort an array.

Run the training script by doing the following:

```
python lstm_sort.py --start-range 100 --end-range 1000 --cpu
```
You can provide the start-range and end-range for the numbers and whether to train on the cpu or not.
By default the script tries to train on the GPU. The default start-range is 100 and end-range is 1000.

At last, test model by doing the following:

```
python infer_sort.py 234 189 785 763 231
```

This should output the sorted seq like the following:
```
189
231
234
763
785
```
