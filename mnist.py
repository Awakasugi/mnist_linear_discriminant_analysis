from sklearn import datasets, model_selection, svm, metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import numpy as np
import itertools
np.random.seed(42)


def augument_data(len_shift, data_list, data_width):
    shift_length = range(len_shift)
    shift_position = list(itertools.product(shift_length, shift_length))
    return np.array([data[y: y + data_width, x: x + data_width] for data in data_list for y, x in shift_position])

def augument_label(len_shift, label):
    repeat_times = len_shift * len_shift
    return np.repeat(label, repeat_times)
    # num_repeat = len_shift * len_shift
    # return [num_repeat for _i in range(9)]

def whitening(X_train, X_test, epsilon = 1e-5):
    mean_x = np.mean(X_train)
    X_train = X_train - mean_x
    X_test = X_test - mean_x
    n, p = X_train.shape
    eigenvalue, eigenvector = np.linalg.eig(np.dot(X_train.T, X_train) / n)
    return whitening_calculate(X_train, eigenvector, eigenvalue, epsilon), whitening_calculate(X_test, eigenvector, eigenvalue, epsilon)

def whitening_calculate(data, eigenvector, eigenvalue, epsilon):
    return np.dot(data, np.dot(eigenvector, calculating_diag(eigenvalue, epsilon)))

def calculating_diag(eigenvalue, epsilon):
    return np.diag(1 / (np.sqrt(eigenvalue) + epsilon))

def calculate_log_likelihood_each_class(input_data, inverse_sigma, mu_each_class, num_samples_eachclass):
    return np.dot(np.dot((input_data.T - mu_each_class.T) / 2, inverse_sigma), mu_each_class) + np.log(num_samples_eachclass)

def predict_label(data, inverse_sigma, mu_each_class_list, num_train_sample_per_class, num_class):
    return str(np.argmax([calculate_log_likelihood_each_class(data, inverse_sigma, mu_each_class_list[i], num_train_sample_per_class[i]) for i in range(num_class)]))

def make_predicted_label_list(test_data, inverse_sigma, mu_each_class_list, num_train_sample_per_class, num_class):
    return np.array([predict_label(data, inverse_sigma, mu_each_class_list, num_train_sample_per_class, num_class) for data in test_data])

def main():
    mnist = datasets.fetch_openml("mnist_784", version=1)
    mnist_data = mnist.data / 255
    mnist_label = mnist.target

    train_data, test_data, label_train, label_test =\
    model_selection.train_test_split(mnist_data, mnist_label, test_size = 0.2, random_state = 42)

    num_class = np.unique(label_train).size
    num_train_sample = train_data.shape[0]
    num_test_sample = test_data.shape[0]

    data_width, shift_width = 28, 1
    shift_length = shift_width * 2 + 1
    train_data = train_data.reshape(num_train_sample, data_width, data_width)
    test_data = test_data.reshape(num_test_sample, data_width, data_width)
    test_data = np.array([data[shift_width: data_width - shift_width, shift_width: data_width - shift_width] for data in test_data])

    data_width = 26
    train_data = augument_data(shift_length, train_data, data_width)
    label_train = augument_label(shift_length, label_train)

    num_train_sample = num_train_sample * shift_length * shift_length
    train_data = train_data.reshape(num_train_sample, data_width * data_width)
    test_data = test_data.reshape(num_test_sample, data_width * data_width)

    train_data, test_data = whitening(train_data, test_data)
    print('finish whitening')
    num_train_sample_per_class = np.array([np.count_nonzero(label_train == str(i)) for i in range(num_class)])

    sigma = np.cov(train_data.T)

    inverse_sigma = np.linalg.pinv(sigma)

    predicted_label = make_predicted_label_list(test_data, inverse_sigma, np.array([np.mean(train_data[label_train == str(i)], axis = 0) for i in range(num_class)]), 
    num_train_sample_per_class, num_class)

    print('accuracuy_score:', accuracy_score(label_test, predicted_label))
    print('precision_score:', precision_score(label_test, predicted_label, average = 'weighted'))
    print('recall_score:', recall_score(label_test, predicted_label, average = 'weighted'))
    print('F_score:', f1_score(label_test, predicted_label, average = 'weighted'))


main()