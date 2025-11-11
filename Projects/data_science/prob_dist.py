import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_vectors = 50   # how many vectors you want
vector_dim = 3     # dimension of each vector

xi = 10
# Generate random vectors (from standard normal distribution)
random_vectors = np.random.randn(num_vectors, vector_dim)

sigma = 0.5
sigma_n = sigma**2
v = []
H = 0

sum = 0
for xj in random_vectors:
    b = np.sum((xj - random_vectors[xi])**2)
    sum += np.exp((-b/(2*(sigma_n**2))))

for xj in random_vectors:
    if xj[0] == random_vectors[xi][0]:
        p = 0
        v.append(p)
        continue
    b = np.sum((xj - random_vectors[xi])**2)
    #print(b)
    s = sum - b
    p = np.exp((-b/(2*(sigma_n**2))))/s
    #print(f'{p=}')
    v.append(p)
    H += -(p * np.log2(p))
    print(f'{H=}')

r = np.arange(0, num_vectors)
plt.plot(r, v)
plt.show()
plt.scatter(random_vectors[:,0], random_vectors[:,1] )
plt.scatter(random_vectors[xi][0], random_vectors[xi][1], color='c')
plt.show()


