from sklearn import datasets, model_selection, svm, metrics
import numpy as np
mnist = datasets.fetch_openml("mnist_784", version=1)

mnist_data = mnist.data / 255
mnist_label = mnist.target

data_train, data_test, label_train, label_test =\
model_selection.train_test_split(mnist_data, mnist_label, test_size=0.2, random_state=42)


def white(X_train, X_test, epsilon=1e-5):
    n, p = X_train.shape
    u, v = np.linalg.eig(np.dot(X_train.T, X_train)/n)
    Z_train = np.dot(X_train, np.dot(v, np.diag(1/(np.sqrt(u) + epsilon))))
    Z_test = np.dot(X_test, np.dot(v, np.diag(1/(np.sqrt(u) + epsilon))))
    return (Z_train, Z_test)
    
data_train_white, data_test_white = white(data_train, data_test)

num_class = np.unique(label_train).size
dimension = data_train_white.shape[1]
num_train_sample = data_train_white.shape[0]
num_test_sample = data_test_white.shape[0]
np.random.seed(42)

num_samples_train_eachclass = [np.count_nonzero(label_train == i) for i in range(num_class)]
num_samples_test_eachclass = [np.count_nonzero(label_test == i) for i in range(num_class)]

data_train_vertical = data_train_white[: , : , np.newaxis]
data_test_vertical = data_test_white[: , : , np.newaxis]

data_train_eachclass_list = [data_train_vertical[label_train == i] for i in range(num_class)]

mu_eachclass_vertical = [np.mean(data_train_eachclass_list[i], axis = 0)for i in range(num_class)]

def sigma_eachclass(each_class_data_list, class_label):
    count = 0
    global sum_each
    sum_each = 0

    while count < num_samples_train_eachclass[class_label]:
        sum_each += np.dot((each_class_data_list[class_label][count] - mu_eachclass_vertical[class_label]), 
                    (each_class_data_list[class_label][count] - mu_eachclass_vertical[class_label]).T)
        count += 1
    return sum_each


def calculate_sigma(each_class_data_list, num_class):
    count = 0
    global sum_all
    sum_all = 0

    while count < num_class:
        sum_all += sigma_eachclass(each_class_data_list, count)
        count += 1
    return sum_all
     
sigma = calculate_sigma(data_train_eachclass_list, num_class)

inverse_sigma = np.linalg.pinv(sigma)

def calculate_log_likelihood_each_class(input_data, inverse_sigma, mu_each_class, num_samples_eachclass):
    log_likelihood_each_class = np.dot(np.dot(input_data.T, inverse_sigma), mu_each_class)
    - np.dot(np.dot(mu_each_class.T, inverse_sigma), mu_each_class)/2 + np.log(num_samples_eachclass)
             
    return log_likelihood_each_class

def predict_label():
    predicted_label_list = []
    count = 0
    while count < num_test_sample:
        predicted_label = np.argmax([calculate_log_likelihood_each_class(data_test_vertical[count], inverse_sigma, 
                                                               mu_eachclass_vertical[i],
                                                               num_samples_train_eachclass[i])
                                     for i in range(num_class)])
        predicted_label_list.append(predicted_label)
        count += 1
    return predicted_label_list

predicted_label = predict_label()

match = [predicted_label == label_test]
u, counts = np.unique(match, return_counts=True)
print(u)
print('-------------------')
print(counts)
print('accuracy is', counts[1]/num_test_sample)