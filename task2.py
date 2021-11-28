import sys
from math import e as e
import pandas as pd
import numpy as np
from math import log
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class NeuralNet:
    def __init__(self,batch_size = 32, lr = 0.01, layers = [37,32,32,1]):
        self.batch_size = batch_size
        self.lr = lr
        self.layers = layers
        self.HiddenNeurons = 32
        self.OutNeurons = 1
        self.epochs = 200
        self.train_set = None
        self.test_set = None
        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None
        self.weights = {}
        self.bias = {}
        self.lay = {}
        self.Z = {}
        self.dA = {}
        self.dZ = {}
        self.dW = {}
        self.db = {}
    
    def relu(self,z):
        R = np.maximum(0,z)
        return R

    def relu_derivative(self,z):
        z[z >= 0] = 1
        z[z < 0]  = 0
        return z
        
    def sigmoid(self, z):
        x = 1/(1+e**(-z))
        return x
    
    def sigmoid_derivative(self,z):
        SD = self.sigmoid(z) * (1 - self.sigmoid(z))
        return SD
        
    '''def calculate_cost(self, y_hat):
        num_train_examples = y_hat.shape
        length = len(self.layers)-1
        cost = np.sum(- np.dot(y_hat, np.log((self.lay[length]))) - np.dot(1 - y_hat, np.log(1 - self.lay[length]))) / num_train_examples
        cost = np.squeeze(cost)
        return cost'''
    
    def display(self):
        print("Learning rate : {0}\nbatch = {1}\nlayers = {2}".format(self.lr,self.batch_size,self.layers))
        print("-----------------------------------------------------Train data Information----------------------------------------------------------")
        print(self.train_X)
        print("-----------------------------------------------------Test data Information----------------------------------------------------------")
        print(self.train_Y)
        
    def preprocess(self,data):
    #data.info()
        obj_df = data.select_dtypes(include=['object']).copy()
        obj_df["songtitle"] = obj_df["songtitle"].astype('category')
        obj_df["artistname"] = obj_df["artistname"].astype('category')
        obj_df["songID"] = obj_df["songID"].astype('category')
        obj_df["artistID"] = obj_df["artistID"].astype('category')

        data["songtitle"] = obj_df["songtitle"].cat.codes
        data["artistname"] = obj_df["artistname"].cat.codes
        data["songID"] = obj_df["songID"].cat.codes
        data["artistID"] = obj_df["artistID"].cat.codes
        shape=data.shape
        for column in data:
            if column=='Top10':
                continue
            if data.dtypes[column]!="float64":
                data[column] = data[column].astype('float')
            mean=data.mean()[column]
            std=data.std()[column]
            for i in range(shape[0]):
                data.at[i,column]=(data.at[i,column]-mean)/std

        shuffle_data = data.sample(frac=1)
        train_size = int(0.8 * len(data))
        # Split your dataset 
        self.train_set = shuffle_data[:train_size]
        self.test_set = shuffle_data[train_size:]
        self.train_X = self.train_set.iloc[:, :-1].values
        self.train_Y = self.train_set.iloc[:,-1:]
        self.test_X = self.test_set.iloc[:, :-1]
        self.test_X=self.test_X.drop(self.test_X.columns[-1],axis=1)
        self.test_Y = self.test_set.iloc[:,-1:]

    def create_miniBatches(self):
        mini_batches = []
        data = self.train_X
        batchSize = self.batch_size
        n_minibatches = data.shape[0] // batchSize
        i = 0
        for i in range(n_minibatches + 1): 
            mini_batch = data[i * batchSize:(i + 1)*batchSize, :] 
            X_mini = mini_batch[:, :-1] 
            Y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((X_mini, Y_mini)) 
        if data.shape[0] % batchSize != 0: 
            mini_batch = data[i * batchSize:data.shape[0]]
            X_mini = mini_batch[:, :-1] 
            Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
            mini_batches.append((X_mini, Y_mini))
        return mini_batches
    
    def error_sum(self,error):
        sum=[]
        for i in range(len(error)):
            sum.append(np.sum(error[i]))
        return sum
        
    def init_weights(self):
        np.random.seed(42)
        length = len(self.layers)
        #noofiterations = length-1
        for i in range(1,length):
            self.weights[i] = np.random.randn(self.layers[i-1], self.layers[i]) * np.sqrt(2. / self.layers[i - 1])
            self.bias[i] = np.random.randn(1,self.layers[i])
        
    def forward_pass(self,data):
        length = len(self.layers)
        self.lay[0] = data
        #print("--------------{}-------------".format(data.shape))
        
        for i in range(1,length):
            #print("At i={} weight:-{}-".format(i,self.weights[i].shape))
            #print("At i={} bias:-{}-".format(i,self.bias[i].shape))
            #print("At i={} lay:-{}-".format(i,self.lay[i-1].shape))
            self.Z[i] = np.dot(self.lay[i - 1], self.weights[i]) + self.bias[i]
            #print(self.Z[i])
            
            if i == length-1:
                output = self.lay[i] = self.sigmoid(self.Z[i])
                #print(self.lay[i])
            else:
                self.lay[i] = self.relu(self.Z[i])
            #print("At i={} Z:-{}-".format(i,self.Z[i].shape))
            #print()
        return output
     
    def backpropagation(self,y):
        length = len(self.layers)
        output_error={}
        for i in range(length-1,0,-1):
            #print("---------------{}--------------".format(i))
            if i==length-1:
                output_error[i] = np.subtract(self.lay[i],y)
                #print(self.lay[i-1].shape)
                #print(output_error[i].shape)
                delta_weight=np.dot(self.lay[i-1].T,output_error[i])
                #print(delta_weight.shape)
                delta_bias=np.sum(output_error[i])
            else:
                #print(np.dot(output_error[i+1],self.weights[i+1].T).shape)
                #print(self.relu_derivative(self.lay[i]).shape)
                output_error[i]=np.multiply(np.dot(output_error[i+1],self.weights[i+1].T),self.relu_derivative(self.lay[i]))
                delta_weight=np.dot(self.lay[i-1].T,output_error[i])
                delta_bias=self.error_sum(output_error[i].T)
                
            #print("-------------delta weights------------")
            delta_weight = delta_weight/(np.max(delta_weight)+1)
            #print(delta_weight)
            
            #print("-------------delta bias------------")
            delta_bias = delta_bias/(np.max(delta_bias)+1)
            #print(delta_bias)
            
            self.weights[i]-=self.lr*delta_weight
            self.bias[i]-=self.lr*delta_bias
            '''print("-------------updated weights------------")
            print(self.weights[i])
            print("-------------updated bias--------------")
            print(self.bias[i])'''
        
    def plot_cost(self,cost,flag):
        plt.plot(cost.keys(), cost.values())
#         if flag==0:
#             plt.title("Training Accuracy")
        if flag==1:
            plt.title("Accuracy vs Batch Size")
            plt.xlabel("Batch Size")
            plt.ylabel("Accuracy")
            plt.grid(True)
            plt.xticks(list(range(1,128,1)))
        if flag==2:
            plt.title("Accuracy vs Learning Rate")
            plt.xlabel("Learning Rate")
            plt.ylabel("Accuracy")
            plt.grid(True)
            #plt.xticks(list(range(0.01, 1.0,0.05)))  # Setting up the x ticks

        if flag==3:
            plt.title("Accuracy vs Number of hidden layers")
            plt.xlabel("Number of hidden layers")
            plt.ylabel("Accuracy")
            plt.grid(True)
            plt.xticks(list(range(1,8,1)))  # Setting up the x ticks

        plt.show()
    
    def training_accruracy(self):
        correct=0
        train_X = self.train_set.iloc[:, :-1]
        train_X=train_X.drop(train_X.columns[-1],axis=1)
        train_Y = self.train_set.iloc[:,-1:]
        output=self.forward_pass(train_X)
        i=0
        for index,row in train_Y.iterrows():
            if output[i]>0.5:
                check=1
            else:
                check=0
            if check==row['Top10']:
                correct+=1
            i+=1
        return correct/len(output)*100
        
    def training(self):
        accuracy_dict={}
        test_acc_dict={}
        c=0
        best_acc=0
        best_test=0
        best_epoch=-1
        for i in range(self.epochs):
            j=0
            batches = self.create_miniBatches()
            for batch in batches:
                x_batch, y_batch = batch
                output = self.forward_pass(x_batch)
                self.backpropagation(y_batch)
            if (i+1)%10==0:
                accuracy_dict[i+1]=self.training_accruracy()
                test_acc_dict[i+1]=self.test()
                if accuracy_dict[i+1]>best_acc:
                    best_acc=accuracy_dict[i+1]
                    best_epoch=i+1
                if test_acc_dict[i+1]>best_test:
                    best_test=test_acc_dict[i+1]
                    
    qwe=0        
    def test(self):
        correct=0
        output=self.forward_pass(self.test_X)
        i=0
        for index,row in self.test_Y.iterrows():
            if output[i]>0.5:
                check=1
            else:
                check=0
            if check==row['Top10']:
                correct+=1
            i+=1
            #print(check)
        return correct/len(output)*100
        
data = pd.read_csv("music_data.csv")
pd.set_option("display.max_rows", None, "display.max_columns", None)

batchSize = int(sys.argv[1])
hidden = int(sys.argv[2])
lRate = float(sys.argv[3])

l = [37]
for neurons in range(hidden):
    l.append(32)
l.append(1)

ann = NeuralNet(batchSize,lRate,l)
ann.preprocess(data)
ann.init_weights()
ann.training()
accuracy = ann.test()
print("The Test Accuracy is:"+str(accuracy))

#for graphs
batches = [1,2,4,8,16,32,64,128]
hidden_layers= [1,2,4,6,8]
learningRate= [0.01,0.05,0.1,0.2,0.4,0.8]
testaccuracies_batches = {}
for batchsize in batches:
    data = pd.read_csv("music_data.csv")
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    nn = NeuralNet(batchsize,0.01,[37,32,32,1])
    nn.preprocess(data)
    nn.init_weights()
    nn.training()
    accuracy = nn.test()
    print(accuracy)
    testaccuracies_batches[batchsize] = accuracy
nn.plot_cost(testaccuracies_batches,1)

testaccuracies_learningRate = {}
for lr in learningRate:
    data = pd.read_csv("music_data.csv")
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    nn = NeuralNet(32,lr,[37,32,32,1])
    nn.preprocess(data)
    nn.init_weights()
    nn.training()
    accuracy = nn.test()
    print(accuracy)
    testaccuracies_learningRate[lr] = accuracy
nn.plot_cost(testaccuracies_learningRate,2)

testaccuracies_layers = {}
for hidden_layer in hidden_layers:
    data = pd.read_csv("music_data.csv")
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    layers = [37]
    for neurons in range(hidden_layer):
        layers.append(32)
    layers.append(1)
    nn = NeuralNet(32,0.01,layers)
    nn.preprocess(data)
    nn.init_weights()
    nn.training()
    accuracy = nn.test()
    print(accuracy)
    testaccuracies_layers[hidden_layer] = accuracy
nn.plot_cost(testaccuracies_layers,3)