'''学习自《Python神经网络编程》一书'''
'''使用MNIST数据集进行训练'''
import numpy as np
import scipy.special
import matplotlib.pyplot
import scipy.misc
import imageio


# scipy.special for the sigmoid function expit
# sxpit()即为sigmoid激活函数

# neural network class definition  神经网络类的定义
class neuralNetwork:

    # initialise the  neural network  初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input,hidden,outputlayer
        # 设置输入层、隐藏层、输出层节点数量
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices,wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # learning rate  学习率
        self.lr = learningrate

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # train the neural network     训练神经网络
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target-actual)
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors,split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs *
                                      (1.0 - final_outputs)), np.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

        pass

    # query the neural network   查询神经网络
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


# 尝试创建每层三个节点，学习率为0.5的小型神经网络对象
input_nodes=3
hidden_nodes=3
output_nodes=3

# learning rate is 0.5
learning_rate=0.5

#create instance of neural network
n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

# number of input , hidden and output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# learning rate is 0.3
learning_rate = 0.3

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network

# go through all records in the training data set
for record in training_data_list:
    # split the record by the ' , ' ,commas
    all_values = record.split(',')

    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # create the target output values (all 0.01 ,
    # except the desired label which is 0.99)
    targets = np.zeros(output_nodes) + 0.01

    # all_values[0] is the target label for this record
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass

## 测试网络
#load the mnist test data CSV file into a list
test_data_file = open("mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#get the first test record
#手动进行第一次测试
all_values = test_data_list[0].split(',')
#print the label
print(all_values[0])


image_array = np.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap="Greys", interpolation ='None')

n.query(np.asfarray(all_values[1:]) / 255.0 *0.99) + 0.01

# 对神经网络的性能进行评估

# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is the first value
    correct_label = int(all_values[0])
    print("正确的标签应该是", correct_label)
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    print("神经网络给出的答案是", label)
    print("-------------------")

    # append correct or incorrect to list
    if (label == correct_label):
        # network answer matches correct answer ,add 1 to scorecard
        scorecard.append(1)
    else:
        # 答案不匹配
        # network answer does not match correct answer ,add 0 to scorecard
        scorecard.append(0)
        pass
    pass

print(scorecard)

#calculate the performance score , the fraction of correct answers
scorecard_array = np.asarray(scorecard)
print("performance =",scorecard_array.sum() / scorecard_array.size)


img_array = imgio.imread ("Nine.png", flatten=True)
img_data = 255.0 - img_array . reshape (784)
img_data = (img_data / 255.0 * 0.99) + 0.01

inputs = (np.asfarray(img_data) / 255.0 *0.99) + 0.01
#query the network
outputs = n.query(inputs)
#the index of the highest value corresponds to the label
label = np.argmax(outputs)
print("神经网络给出的答案是",label)
print("-------------------")