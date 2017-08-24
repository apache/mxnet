import random

vocab = [str(x) for x in range(100, 1000)]
sw_train = open("sort.train.txt", "w")
sw_test = open("sort.test.txt", "w")
sw_valid = open("sort.valid.txt", "w")

for i in range(1000000):
    seq = " ".join([vocab[random.randint(0, len(vocab) - 1)] for j in range(5)])
    k = i % 50
    if k == 0:
        sw_test.write(seq + "\n")
    elif k == 1:
        sw_valid.write(seq + "\n")
    else:
        sw_train.write(seq + "\n")

sw_train.close()
sw_test.close()
sw_valid.close()
