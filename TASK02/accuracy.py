def accuracy(expected, real):
    total = 0
    correct = 0
    for i in range(len(real)):
        total += 1
        if (round(real[i]) == expected[i]):
            correct += 1
    return (correct / total) * 100
