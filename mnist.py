from sklearn import datasets, model_selection, svm, metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import itertools
np.random.seed(42)


def augument_data(len_shift, data_list):
    shift_length = range(len_shift)
    shift_position = list(itertools.product(shift_length, shift_length))
    return np.array([data[y: y+24, x: x+24] for data in data_list for y, x in shift_position])

def augument_label(len_shift, label):
    num_repeat = len_shift*len_shift
    return np.repeat(label, num_repeat)

def whitening(X_train, X_test, epsilon = 1e-5):
    mean_x = np.mean(X_train)
    X_train = X_train - mean_x
    X_test = X_test - mean_x
    n, p = X_train.shape
    u, v = np.linalg.eig(np.dot(X_train.T, X_train) / n)
    return(np.dot(X_train, np.dot(v, calculating_diag(u, epsilon))),  (np.dot(np.dot(calculating_diag(u, epsilon), v.T) , X_test.T)).T)

def calculating_diag(u, epsilon):
    return np.diag(1 / (np.sqrt(u) + epsilon))

def calculate_log_likelihood_each_class(input_data, inverse_sigma, mu_each_class, num_samples_eachclass):
    return np.dot(np.dot((input_data.T - mu_each_class.T)/2, inverse_sigma), mu_each_class) + np.log(num_samples_eachclass)

def predict_label(data, inverse_sigma, mu_each_class_list, num_train_sample_per_class, num_class):
    return str(np.argmax([calculate_log_likelihood_each_class(data, inverse_sigma, mu_each_class_list[i], 
    num_train_sample_per_class[i]) for i in range(num_class)]))

def make_predicted_label_list(test_data, inverse_sigma, mu_each_class_list, num_train_sample_per_class, num_class):
    return np.array([predict_label(data, inverse_sigma, mu_each_class_list, num_train_sample_per_class, num_class) for data in test_data])

def main():

    mnist = datasets.fetch_openml("mnist_784", version=1)
    mnist_data = mnist.data / 255
    mnist_label = mnist.target

    train_data, test_data, label_train, label_test =\
    model_selection.train_test_split(mnist_data, mnist_label, test_size=0.2, random_state=42)

    train_data = train_data.reshape(56000, 28, 28)
    train_data = augument_data(5, train_data)
    label_train = augument_label(5, label_train)
    train_data = train_data.reshape(1400000, 576)

    test_data = test_data.reshape(14000, 28, 28)
    test_data = np.array([data[2: 26, 2: 26] for data in test_data])
    test_data = test_data.reshape(14000, 576)

    num_class = np.unique(label_train).size
    print('num_class:', num_class)
    num_train_sample = train_data.shape[0]
    print('num_train_sample:', num_train_sample)
    num_test_sample = test_data.shape[0]
    print('num_test_sample:', num_test_sample)

    train_data, test_data = whitening(train_data, test_data)
    print('finish whitening')
    num_train_sample_per_class = np.array([np.count_nonzero(label_train == str(i)) for i in range(num_class)])
    print('num_train_sample_per_class.shape:', num_train_sample_per_class.shape)

    train_data_eachclass_list = np.array([train_data[label_train == str(i)] for i in range(num_class)])
    print('train_data_eachclass_list.shape:', train_data_eachclass_list.shape)

    sum_sigma = 0
    for sigma, num_samples in zip([np.cov(data_list.T) for data_list in train_data_eachclass_list], num_train_sample_per_class):
        sum_sigma += sigma * num_samples
    print('sum_sigma:', sum_sigma)

    sigma = sum_sigma / num_train_sample
    inverse_sigma = np.linalg.pinv(sigma)
    print('inverse_sigma:', inverse_sigma)

    predicted_label = make_predicted_label_list(test_data, inverse_sigma, np.array([np.mean(train_data[label_train == str(i)], axis = 0) for i in range(num_class)]), 
    num_train_sample_per_class, num_class)

    print('accuracuy_score:', accuracy_score(label_test, predicted_label))
    print('precision_score:', precision_score(label_test, predicted_label, average = 'weighted'))
    print('recall_score:', recall_score(label_test, predicted_label, average = 'weighted'))
    print('F_score:', f1_score(label_test, predicted_label, average = 'weighted'))


main()