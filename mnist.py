from sklearn import datasets, model_selection, svm, metrics
import numpy as np
np.random.seed(42)


def whitening(X_train, X_test, epsilon = 1e-5):
    mean_x = np.mean(X_train)
    X_train = X_train - mean_x
    X_test = X_test - mean_x
    n, p = X_train.shape
    u, v = np.linalg.eig(np.dot(X_train.T, X_train) / n)
    Z_train = np.dot(X_train, np.dot(v, np.diag(1 / (np.sqrt(u) + epsilon))))
    Z_test = (np.dot(np.dot( np.diag(1 / (np.sqrt(u) + epsilon)), v.T) , X_test.T)).T
    return (Z_train, Z_test)

def calculate_log_likelihood_each_class(input_data, inverse_sigma, mu_each_class, num_samples_eachclass):
    log_likelihood_each_class = np.dot(np.dot((input_data.T - mu_each_class.T)/2, inverse_sigma), mu_each_class)
    + np.log(num_samples_eachclass)
             
    return log_likelihood_each_class

def predict_label(data, inverse_sigma, mu_each_class_list, num_train_samples_per_class, num_class):

    predicted_label = np.argmax([calculate_log_likelihood_each_class(data, inverse_sigma, mu_each_class_list[i], 
    num_train_samples_per_class[i]) for i in range(num_class)])

    return str(predicted_label)

def make_predicted_label_list(test_data, inverse_sigma, mu_each_class_list, num_train_samples_per_class, num_class):
    predicted_label_list = [predict_label(data, inverse_sigma, mu_each_class_list, num_train_samples_per_class, num_class)
    for data in test_data]

    return predicted_label_list


def main():
    mnist = datasets.fetch_openml("mnist_784", version=1)
    mnist_data = mnist.data / 255
    mnist_label = mnist.target

    train_data, test_data, label_train, label_test =\
    model_selection.train_test_split(mnist_data, mnist_label, test_size=0.2, random_state=42)

    num_class = np.unique(label_train).size
    num_train_sample = train_data.shape[0]
    num_test_sample = test_data.shape[0]

    train_data, test_data = whitening(train_data, test_data)

    num_train_samples_per_class = [np.count_nonzero(label_train == str(i)) for i in range(num_class)]
    num_test_samples_per_class = [np.count_nonzero(label_test == str(i)) for i in range(num_class)]

    train_data_eachclass_list = [train_data[label_train == str(i)] for i in range(num_class)]
    mu_each_class_list = [np.mean(train_data[label_train == str(i)], axis = 0) for i in range(num_class)]
    sigma_each_class_list = [np.cov(data_list.T) for data_list in train_data_eachclass_list]

    sum_sigma = 0
    for sigma, num_samples in zip(sigma_each_class_list, num_train_samples_per_class):
        sum_sigma += sigma*num_samples
    sigma = sum_sigma/num_train_sample
    inverse_sigma = np.linalg.pinv(sigma)

    predicted_label = make_predicted_label_list(test_data, inverse_sigma, mu_each_class_list, num_train_samples_per_class, num_class)
    match = [predicted_label == label_test]
    u, counts = np.unique(match, return_counts=True)
    print(u)
    print('-------------------')
    print(counts)
    print('accuracy is', counts[1]/num_test_sample)

main()