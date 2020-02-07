from sklearn import datasets, model_selection, svm, metrics
import numpy as np
np.random.seed(42)

# test change retry

def whitening(X_train, X_test, epsilon=1e-5):
    n, p = X_train.shape
    u, v = np.linalg.eig(np.dot(X_train.T, X_train)/n)
    Z_train = np.dot(X_train, np.dot(v, np.diag(1/(np.sqrt(u) + epsilon))))
    Z_test = np.dot(X_test, np.dot(v, np.diag(1/(np.sqrt(u) + epsilon))))
    return (Z_train, Z_test)

def calculate_deviation_from_mu(x, mu):
    deviation_from_mu = np.dot((x - mu), (x - mu).T)
    return deviation_from_mu

def calculate_sigma(data_eachclass_list, mu_eachclass_list):
    for element_data_list, element_mu in zip(data_eachclass_list, mu_eachclass_list):
        list_eachclass_sigma.append(sum([calculate_deviation_from_mu(e, element_mu) for e in element_data_list, axis=1)])
    return list_eachclass_sigma 

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


def main():
    mnist = datasets.fetch_openml("mnist_784", version=1)
    mnist_data = mnist.data / 255
    mnist_label = mnist.target

    data_train, data_test, label_train, label_test =\
    model_selection.train_test_split(mnist_data, mnist_label, test_size=0.2, random_state=42)

    num_class = np.unique(label_train).size
    dimension = data_train.shape[1]
    num_train_sample = data_train.shape[0]
    num_test_sample = data_test.shape[0]

    num_samples_train_eachclass = [np.count_nonzero(label_train == i) for i in range(num_class)]
    num_samples_test_eachclass = [np.count_nonzero(label_test == i) for i in range(num_class)]

    data_train_white, data_test_white = whitening(data_train, data_test)
    data_train_vertical = data_train_white[: , : , np.newaxis]
    data_test_vertical = data_test_white[: , : , np.newaxis]

    data_train_eachclass_list = [data_train_vertical[label_train == i] for i in range(num_class)]
    mu_eachclass_list = [np.mean(data_train_eachclass_list[i], axis = 0) for i in range(num_class)]

    sigma = calculate_sigma(data_train_eachclass_list, mu_eachclass_list)
    print(sigma)
    # inverse_sigma = np.linalg.pinv(sigma)

    # predicted_label = predict_label()
    # match = [predicted_label == label_test]
    # u, counts = np.unique(match, return_counts=True)
    # print(u)
    # print('-------------------')
    # print(counts)
    # print('accuracy is', counts[1]/num_test_sample)



main()