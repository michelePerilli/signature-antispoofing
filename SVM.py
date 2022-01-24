from pandas import read_csv
from sklearn import svm
import numpy as np
from pandas import DataFrame
import pickle
import os


# Classifier class
class OneClassSVM:
    def __init__(self):
        self.classifier = None
        pass

    def train(self, training_data, kernel, nu, gamma):
        self.classifier = svm.OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

        self.classifier.fit(training_data)
        pass

    def test(self, test_data):
        x_genuine = test_data

        y_genuine = self.classifier.predict(x_genuine)

        return y_genuine


SIGNATURES = 24  # Genuine signatures used for training


# SVM parameters optimization
def fitting():
    print("%5s %20s %10s %10s %10s" % ('User', 'Genuine accepted', 'Kernel', 'nu', 'gamma'))
    for user in os.listdir(csv_path):
        output = []
        user_folder = csv_path + user + '\\'
        cls = OneClassSVM()

        data = read_csv(user_folder + 'features.csv').drop('Unnamed: 0', axis=1)
        result = []
        for kernel in ('rbf', 'poly'):
            for nu in np.linspace(0.002, 0.2, 10):
                for gamma in np.linspace(0.002, 0.2, 10):
                    cls.train(data, kernel, nu, gamma)
                    accepted = cls.test(data)
                    result.append(accepted[accepted == 1].size)
                    output.append([user, accepted[accepted == 1].size, kernel, nu, gamma])

        a = np.argmax(result)

        pickle.dump(output, open(user_folder + 'optimization.bin', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        print("%5s %10s / %d %14s %12.3f %9.3f" % (user,
                                                   output[int(a)][1], SIGNATURES,
                                                   output[int(a)][2],
                                                   output[int(a)][3],
                                                   output[int(a)][4]))


# Test a single user using csv files
def test_user(user):
    optimization_data = DataFrame(pickle.load(open('.csv\\' + user + '\\optimization.bin', 'rb')))

    optimal_parameter_index = np.argmax(np.asarray(DataFrame(optimization_data).get(1)))
    a, b, kernel, nu, gamma = np.asarray(optimization_data.iloc[optimal_parameter_index])

    train = read_csv('.csv\\' + user + '\\features.csv')
    test = read_csv('.csv_test\\' + user + '\\features.csv')

    cls = OneClassSVM()
    cls.train(train, kernel, nu, gamma)
    return cls.test(test)


# Generate 'testResult/test_difesa.txt' file
def antispoof_filter(user, svm_result, test_atk, test_def):
    dataset = np.asarray(read_csv(test_atk))
    print(dataset)
    file = open(test_def, "a")

    index = 0 + int(user) * 12 - 12
    svm_index = 0
    for classe in range(2):
        for sign in range(6):

            if dataset[index][2] == 'F':
                if svm_result[svm_index] == -1:
                    dataset[index][3] = 0
                svm_index += 1

            print(dataset[index])
            row = ('%s,%s,%s,%s\n' % (dataset[index][0], dataset[index][1], dataset[index][2], dataset[index][3]))
            file.write(row)
            index += 1
    file.close()


csv_path = '.csv\\'
csv_path_test = '.csv_test\\'
# fitting()  #  Generate optimization.bin file containing the optimal parameters for the svm of each user
test_atk = 'testResult\\test_attacco.txt'
test_def = 'testResult\\test_difesa.txt'
f = open(test_def, "w")
f.close()
for user in os.listdir(csv_path):
    result = test_user(user)
    antispoof_filter(user, result, test_atk, test_def)
