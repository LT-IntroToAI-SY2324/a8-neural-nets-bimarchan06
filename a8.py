from neural import *
print("<<<<<<<<<<<<<< XOR 2 hidden nodes >>>>>>>>>>>>>>\n")
xor_data = [
    ([0, 0], [0]),
    ([1, 0], [1]),
    ([0, 1], [1]),
    ([0, 0], [0]),
]

print(xor_data)

xor_nn = NeuralNet(2, 2, 1)
xor_nn.train(xor_data, iters=10000, print_interval=1000) 

print(xor_nn.test_with_expected(xor_data))
print(xor_nn.evaluate[(1,1)])

print("<<<<<<<<<<<<<< XOR 8 hidden nodes >>>>>>>>>>>>>>\n")
xor_data = [
    ([0, 0], [0]),
    ([1, 0], [1]),
    ([0, 1], [1]),
    ([1, 1], [0]),
]

print(xor_data)

xor_nn = NeuralNet(2, 2, 1)
xor_nn.train(xor_data, iters=10000, print_interval=1000) 

print(xor_nn.test_with_expected(xor_data))
print(xor_nn.evaluate[(1,1)])

# print("<<<<<<<<<<<<<< XOR 1 hidden nodes >>>>>>>>>>>>>>\n")
# xor_data = [
#     ([0, 0], [0]),
#     ([1, 0], [1]),
#     ([0, 1], [1]),
#     ([1, 1], [0]),
# ]

# print(xor_data)

# xor_nn = NeuralNet(2, 1, 1)
# xor_nn.train(xor_data, iters=10000, print_interval=1000) 

# print(xor_nn.test_with_expected(xor_data))
# print(xor_nn.evaluate[(1,1)])
print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")




