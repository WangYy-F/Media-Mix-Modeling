def adstock(x, l=0.5):
    result = []
    prev = 0
    for xi in x:
        prev = xi + l * prev
        result.append(prev)
    return result

def saturation(x, alpha=1.0, beta=0.5):
    return alpha * (1 - np.exp(-beta * np.array(x)))
