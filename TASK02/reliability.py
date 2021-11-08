def reliability(vector):
    epsilon = 0.3
    total = 0
    good = 0
    for i in range(len(vector)):
        total += 1
        if vector[i] < epsilon or vector[i] > (1 - epsilon):
            good += 1
    return (good / total) * 100