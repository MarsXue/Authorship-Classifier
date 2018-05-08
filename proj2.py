import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

# HELPER LIST
raw_type_list = ['User_ID', 'Gender', 'Age', 'Occupation', 'Star_Sign', 'Date', 'Text']
top_word_list = ['anyways', 'cuz', 'digest', 'diva', 'evermean', 'fox', 'gonna', 'greg', 'haha', 'jayel',
                 'kinda', 'levengals', 'literacy', 'lol', 'melissa', 'nan', 'nat', 'postcount', 'ppl', 'rick',
                 'school', 'shep', 'sherry', 'spanners', 'teri', 'u', 'ur', 'urllink', 'wanna', 'work']
top_type_list = ['Instance_ID'] + top_word_list + ['Class']
raw_file_dict = {'train': 'train_raw.csv', 'dev': 'dev_raw.csv', 'test': 'test_raw.csv'}
top_file_dict = {'train': 'train_top10.csv', 'dev': 'dev_top10.csv', 'test': 'test_top10.csv'}


def read_data(file_name):
    if 'raw' in file_name:
        input_data = pd.read_csv(file_name, names=raw_type_list)
    else:
        input_data = pd.read_csv(file_name, names=top_type_list)
    # input_data = input_data[input_data.Class != "?"]
    return input_data


def read_id(file_name):
    f = open(file_name, 'r', encoding='utf-8')
    id_lst = []

    for line in f.readlines():
        row = line.rstrip().split(',')
        id_lst.append(row[0])
    f.close()
    return id_lst


def preprocess(id_lst, X_lst, y_lst):
    X_dict = {x: [0 for i in range(len(X_lst))] for x in id_lst}
    y_dict = {}

    for i in range(len(id_lst)):
        X_dict[id_lst[i]] += X_lst[i]
        y_dict[id_lst[i]] = y_lst[i]

    id_set = [k for k in X_dict.keys()]
    X = [v.tolist() for v in X_dict.values()]
    y = [v for v in y_dict.values()]

    return id_set, X, y


def convert_back(y_pred, id_lst, id_set):
    pred_dict = {}

    for i in range(len(id_set)):
        pred_dict[id_set[i]] = y_pred[i]

    y_new_pred = []
    for i in range(len(id_lst)):
        y_new_pred.append(pred_dict[id_lst[i]])

    return y_new_pred


def prediction(index, y_pred, file_name):
    df = pd.DataFrame()
    df['Id'] = index
    df['Prediction'] = y_pred
    df.to_csv(file_name, sep=',', index=False)


# --- train data - train_top10.csv
train_top = top_file_dict['train']
train_raw = raw_file_dict['train']
train_id = read_id(train_raw)
train_data = read_data(train_top).as_matrix()
X_train = train_data[:, 1:-1].astype(int)
y_train = train_data[:, -1]

train_id_set, X_new_train, y_new_train = preprocess(train_id, X_train, y_train)

# --- dev data - dev_top10.csv
dev_top = top_file_dict['dev']
dev_raw = raw_file_dict['dev']
dev_id = read_id(dev_raw)
dev_data = read_data(dev_top).as_matrix()
X_dev = dev_data[:, 1:-1].astype(int)
y_dev = dev_data[:, -1]

dev_id_set, X_new_dev, y_new_dev = preprocess(dev_id, X_dev, y_dev)

# --- test data - test_top10.csv
test_top = top_file_dict['test']
test_raw = raw_file_dict['test']
test_id = read_id(test_raw)
test_data = np.genfromtxt(test_top, delimiter=',', dtype='str')
X_test = test_data[:, 1:-1].astype(int)
y_test = test_data[:, -1]
test_index = test_data[:, 0].tolist()
test_id_set, X_new_test, y_new_test = preprocess(test_id, X_test, y_test)

# --- Decision Tree --- Dev
dt = DecisionTreeClassifier(max_depth=None)
dt.fit(X_new_train, y_new_train)
y_dev_pred = dt.predict(X_new_dev)
y_new_dev_pred = convert_back(y_dev_pred, dev_id, dev_id_set)
print("Decision Tree (Dev):", accuracy_score(y_dev, y_new_dev_pred))

# --- Logistic Regression --- Dev
clf = LogisticRegression()
clf.fit(X_new_train, y_new_train)
y_dev_pred = clf.predict(X_new_dev)
y_new_dev_pred = convert_back(y_dev_pred, dev_id, dev_id_set)
print("Logistic Regression (Dev):", accuracy_score(y_dev, y_new_dev_pred))

# --- Logistic Regression --- Test
# y_test_pred = clf.predict(X_new_test)
# y_new_test_pred = convert_back(y_test_pred, test_id, test_id_set)
# print("Logistic Regression (Test):", accuracy_score(y_test, y_new_test_pred))

# --- MLP Classifier --- Dev
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 2), random_state=1)
clf.fit(X_new_train, y_new_train)
y_dev_pred = clf.predict(X_new_dev)
y_new_dev_pred = convert_back(y_dev_pred, dev_id, dev_id_set)
print("MLP Classifier (Dev):", accuracy_score(y_dev, y_new_dev_pred))

# --- MLP Classifier --- Test
# y_test_pred = clf.predict(X_new_test)
# y_new_test_pred = convert_back(y_test_pred, test_id, test_id_set)
# print("MLP Classifier (Test):", accuracy_score(y_test, y_new_test_pred))

# prediction(test_index, y_new_test_pred, 'out-MLP.csv')


############################################

# HELPER FUNCTION

def feature_selection():
    # Build a classification task using 3 informative features
    # Create the RFE object and compute a cross-validated score.
    clf = LogisticRegression(class_weight='balanced')
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(2), scoring='accuracy')
    rfecv.fit(X_test, y_test)

    # 22 features for dev_top10.csv
    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()


def word_distribution():
    # word_dict = {word: {'14-16': 0, '24-26': 0, '34-36': 0, '44-46': 0} for word in top_word_list}
    # for index, row in data_train.iterrows():
    #     for word in top_word_list:
    #         if row[word] != 0:
    #             word_dict[word][row['Class']] += 1
    #         word_dict[word][row['Class']] += row[word]

    word_dict = {'anyways': {'14-16': 4944, '24-26': 2441, '34-36': 168, '44-46': 13},
                 'cuz': {'14-16': 4819, '24-26': 1381, '34-36': 233, '44-46': 8},
                 'digest': {'14-16': 36, '24-26': 102, '34-36': 369, '44-46': 4},
                 'diva': {'14-16': 31, '24-26': 125, '34-36': 453, '44-46': 0},
                 'evermean': {'14-16': 0, '24-26': 0, '34-36': 105, '44-46': 0},
                 'fox': {'14-16': 187, '24-26': 508, '34-36': 111, '44-46': 417},
                 'gonna': {'14-16': 10597, '24-26': 4874, '34-36': 547, '44-46': 207},
                 'greg': {'14-16': 151, '24-26': 464, '34-36': 112, '44-46': 317},
                 'haha': {'14-16': 6416, '24-26': 1364, '34-36': 70, '44-46': 5},
                 'jayel': {'14-16': 0, '24-26': 0, '34-36': 0, '44-46': 157},
                 'kinda': {'14-16': 6598, '24-26': 3635, '34-36': 336, '44-46': 110},
                 'levengals': {'14-16': 0, '24-26': 0, '34-36': 0, '44-46': 137},
                 'literacy': {'14-16': 11, '24-26': 64, '34-36': 171, '44-46': 6},
                 'lol': {'14-16': 8422, '24-26': 1557, '34-36': 408, '44-46': 444},
                 'melissa': {'14-16': 250, '24-26': 275, '34-36': 21, '44-46': 225},
                 'nan': {'14-16': 88, '24-26': 56, '34-36': 6, '44-46': 145},
                 'nat': {'14-16': 195, '24-26': 78, '34-36': 2, '44-46': 204},
                 'postcount': {'14-16': 0, '24-26': 356, '34-36': 389, '44-46': 0},
                 'ppl': {'14-16': 3343, '24-26': 531, '34-36': 36, '44-46': 0},
                 'rick': {'14-16': 125, '24-26': 350, '34-36': 148, '44-46': 519},
                 'school': {'14-16': 13610, '24-26': 9365, '34-36': 1853, '44-46': 414},
                 'shep': {'14-16': 3, '24-26': 0, '34-36': 2, '44-46': 190},
                 'sherry': {'14-16': 11, '24-26': 48, '34-36': 13, '44-46': 180},
                 'spanners': {'14-16': 0, '24-26': 1, '34-36': 99, '44-46': 0},
                 'teri': {'14-16': 6, '24-26': 12, '34-36': 3, '44-46': 102},
                 'u': {'14-16': 9139, '24-26': 2547, '34-36': 142, '44-46': 28},
                 'ur': {'14-16': 2909, '24-26': 537, '34-36': 14, '44-46': 1},
                 'urllink': {'14-16': 14835, '24-26': 46685, '34-36': 13404, '44-46': 2139},
                 'wanna': {'14-16': 5750, '24-26': 2196, '34-36': 210, '44-46': 46},
                 'work': {'14-16': 9719, '24-26': 26484, '34-36': 5452, '44-46': 1211}}

    # count = {'14-16': 0, '24-26': 0, '34-36': 0, '44-46': 0}
    # total = 0
    # for k in y_train:
    #     count[k] += 1
    # for k in ['14-16', '24-26', '34-36', '44-46']:
    #     total += count[k]
    #
    # dic = {word: {'14-16': 0, '24-26': 0, '34-36': 0, '44-46': 0} for word in top_word_list}
    #
    # for index, row in X_train.iterrows():
    #     for word in top_word_list:
    #         if row[word] > 0:
    #             dic[word][y_train[index]] += 1
    #
    # for word in top_word_list:
    #     _sum = 0
    #     for j in ['14-16', '24-26', '34-36', '44-46']:
    #         dic[word][j] = total * dic[word][j] / count[j]
    #         _sum += dic[word][j]
    #     for j in ['14-16', '24-26', '34-36', '44-46']:
    #         dic[word][j] = dic[word][j] / _sum

    word_dist = {'anyways': {'14-16': 0.6691182379567854, '24-26': 0.230508118020287, '34-36': 0.07376515984507058,
                             '44-46': 0.026608484177856995},
                 'cuz': {'14-16': 0.7236296537282069, '24-26': 0.14469287565639952, '34-36': 0.11350969350726677,
                         '44-46': 0.018167777108126944},
                 'digest': {'14-16': 0.02637747439922867, '24-26': 0.05214647080230209, '34-36': 0.8771516331479886,
                            '44-46': 0.04432442165048062},
                 'diva': {'14-16': 0.01952296013693118, '24-26': 0.05492727159495668, '34-36': 0.9255497682681122,
                          '44-46': 0.0},
                 'evermean': {'14-16': 0.0, '24-26': 0.0, '34-36': 1.0, '44-46': 0.0},
                 'fox': {'14-16': 0.02594315436305799, '24-26': 0.04917438369831134, '34-36': 0.04995992212047086,
                         '44-46': 0.8749225398181598},
                 'gonna': {'14-16': 0.5605997007713579, '24-26': 0.17990759264086534, '34-36': 0.09388038335212125,
                           '44-46': 0.16561232323565556},
                 'greg': {'14-16': 0.0268098387650139, '24-26': 0.05748165655831117, '34-36': 0.06451383783777703,
                          '44-46': 0.8511946668388979},
                 'haha': {'14-16': 0.8364584267053635, '24-26': 0.12407618049904351, '34-36': 0.029607084703633556,
                          '44-46': 0.009858308091959265},
                 'jayel': {'14-16': 0.0, '24-26': 0.0, '34-36': 0.0, '44-46': 1.0},
                 'kinda': {'14-16': 0.5550158468463889, '24-26': 0.2133494499066458, '34-36': 0.09169590690744266,
                           '44-46': 0.13993879633952247},
                 'levengals': {'14-16': 0.0, '24-26': 0.0, '34-36': 0.0, '44-46': 1.0},
                 'literacy': {'14-16': 0.015688123162793954, '24-26': 0.0636872239841547, '34-36': 0.7912104527801802,
                              '44-46': 0.12941420007287105},
                 'lol': {'14-16': 0.47997133335565567, '24-26': 0.06191312358681809, '34-36': 0.07543585660864577,
                         '44-46': 0.38267968644888045},
                 'melissa': {'14-16': 0.06389292209731687, '24-26': 0.049038759548887055, '34-36': 0.017412039252058385,
                             '44-46': 0.8696562791017377},
                 'nan': {'14-16': 0.03761572779208674, '24-26': 0.01670201479263184, '34-36': 0.008320619225598399,
                         '44-46': 0.937361638189683},
                 'nat': {'14-16': 0.05836389321183739, '24-26': 0.01628914486415194, '34-36': 0.0019420358341603447,
                         '44-46': 0.9234049260898503},
                 'postcount': {'14-16': 0.0, '24-26': 0.16445487302236883, '34-36': 0.8355451269776312, '44-46': 0.0},
                 'ppl': {'14-16': 0.8727789149847253, '24-26': 0.09672894040889149, '34-36': 0.03049214460638329,
                         '44-46': 0.0},
                 'rick': {'14-16': 0.014370360527610612, '24-26': 0.02807499809416495, '34-36': 0.055199731731514955,
                          '44-46': 0.9023549096467094},
                 'school': {'14-16': 0.4198399891669903, '24-26': 0.20157074400041788, '34-36': 0.18544652384760632,
                            '44-46': 0.19314274298498552},
                 'shep': {'14-16': 0.0010405993344425428, '24-26': 0.0, '34-36': 0.002250659963603346,
                          '44-46': 0.9967087407019541},
                 'sherry': {'14-16': 0.003916127184328978, '24-26': 0.011923379865968885, '34-36': 0.015014990295512294,
                            '44-46': 0.9691455026541897},
                 'spanners': {'14-16': 0.0, '24-26': 0.0021676981334452596, '34-36': 0.9978323018665548, '44-46': 0.0},
                 'teri': {'14-16': 0.0038296985477289107, '24-26': 0.0053442728883391286, '34-36': 0.006212296780869644,
                          '44-46': 0.9846137317830624},
                 'u': {'14-16': 0.7744724558628764, '24-26': 0.15060184225208706, '34-36': 0.039040308824444994,
                       '44-46': 0.035885393060591624},
                 'ur': {'14-16': 0.8698564164613034, '24-26': 0.11203976157633738, '34-36': 0.01358155468197766,
                        '44-46': 0.004522267280381567},
                 'urllink': {'14-16': 0.12037053097683702, '24-26': 0.2643041783638263, '34-36': 0.35284555075250373,
                             '44-46': 0.26247973990683304},
                 'wanna': {'14-16': 0.6640323036842943, '24-26': 0.17694884758888418, '34-36': 0.07867889612843924,
                           '44-46': 0.0803399525983822},
                 'work': {'14-16': 0.15138556142890766, '24-26': 0.287832939550357, '34-36': 0.2755093369403132,
                          '44-46': 0.28527216208042216}}

    age_dict = {'14-16': {}, '24-26': {}, '34-36': {}, '44-46': {}}
    new_age_dict = {'14-16': {}, '24-26': {}, '34-36': {}, '44-46': {}}

    for k, v in word_dist.items():
        for ks, vs in age_dict.items():
            age_dict[ks][k] = word_dist[k][ks]

    for k, v in age_dict.items():
        new_age_dict[k] = list(reversed(sorted(v.items(), key=operator.itemgetter(1))))

    for k, v in new_age_dict.items():
        print(k, v)

    # for k, v in word_dict.items():
    #     print(v)
    #     plt.bar(range(len(v)), list(v.values()), align='center')
    #     plt.xticks(range(len(v)), list(v.keys()))
    #     plt.title(k)
    #     plt.show()

    # 14 - 16 [('urllink', 14835), ('school', 13610), ('gonna', 10597), ('work', 9719), ('u', 9139),
    #          ('lol', 8422), ('kinda', 6598), ('haha', 6416), ('wanna', 5750), ('anyways', 4944)]
    # 24 - 26[('urllink', 46685), ('work', 26484), ('school', 9365), ('gonna', 4874), ('kinda', 3635),
    #         ('u', 2547), ('anyways', 2441), ('wanna', 2196), ('lol', 1557), ('cuz', 1381)]
    # 34 - 36[('urllink', 13404), ('work', 5452), ('school', 1853), ('gonna', 547), ('diva', 453),
    #         ('lol', 408), ('postcount', 389), ('digest', 369), ('kinda', 336), ('cuz', 233)]
    # 44 - 46[('urllink', 2139), ('work', 1211), ('rick', 519), ('lol', 444), ('fox', 417),
    #         ('school', 414), ('greg', 317), ('melissa', 225), ('gonna', 207), ('nat', 204)]

    # '14-16': [('ppl', 0.8727789149847253), ('ur', 0.8698564164613034), ('haha', 0.8364584267053635),
    #           ('u', 0.7744724558628764), ('cuz', 0.7236296537282069), ('anyways', 0.6691182379567854),
    #           ('wanna', 0.6640323036842943), ('gonna', 0.5605997007713579), ('kinda', 0.5550158468463889),
    #           ('lol', 0.47997133335565567)]
    # '24-26': [('work', 0.287832939550357), ('urllink', 0.2643041783638263), ('anyways', 0.230508118020287),
    #           ('kinda', 0.2133494499066458), ('school', 0.20157074400041788), ('gonna', 0.17990759264086534),
    #           ('wanna', 0.17694884758888418), ('postcount', 0.16445487302236883), ('u', 0.15060184225208706),
    #           ('cuz', 0.14469287565639952)]
    # '34-36': [('evermean', 1.0), ('spanners', 0.9978323018665548), ('diva', 0.9255497682681122),
    #           ('digest', 0.8771516331479886), ('postcount', 0.8355451269776312), ('literacy', 0.7912104527801802),
    #           ('urllink', 0.35284555075250373), ('work', 0.2755093369403132), ('school', 0.18544652384760632),
    #           ('cuz', 0.11350969350726677)]
    # '44-46': [('levengals', 1.0), ('jayel', 1.0), ('shep', 0.9967087407019541), ('teri', 0.9846137317830624),
    #           ('sherry', 0.9691455026541897), ('nan', 0.937361638189683), ('nat', 0.9234049260898503),
    #           ('rick', 0.9023549096467094), ('fox', 0.8749225398181598), ('melissa', 0.8696562791017377)]
