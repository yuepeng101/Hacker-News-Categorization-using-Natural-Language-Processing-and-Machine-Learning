

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
# nltk.download()


# Global variables
path = 'hns_2018_2019.csv'
cleaned_vocabulary = []         # list of cleaned vocabulary
labels = []                     # list of class types
smoothed_value = float(0.5)     # smooth value
cls_cond_prob_log_dict = {}     # log value of conditional prob list of each class
cls_sample_nums = {}            # the number of samples of each class


def read_and_divide_data():
    """
    read data from .csv file
    :return: None
    """
    dataset = pd.read_csv(path)

    # Attract features and labels
    attr_features = dataset['Title']
    attr_labels = dataset['Post Type']

    # Divide features and labels to training set and testing set
    X_train = attr_features[(dataset['Created At'] > '2018-01-01') &
                            (dataset['Created At'] < '2019-01-01')]
    y_train = attr_labels[(dataset['Created At'] > '2018-01-01') &
                          (dataset['Created At'] < '2019-01-01')]
    X_test = attr_features[(dataset['Created At'] >= '2019-01-01')]
    y_test = attr_labels[(dataset['Created At'] >= '2019-01-01')]

    return X_train, y_train, X_test, y_test


def proc_sent(_sent):
    """
    tokenize a sentence
    :param _sent: string
    :return: list of processed tokens
    """
    _sent = _sent.replace("'", "").replace('-', '').replace('_', '').replace('/', ' ') \
            .replace('.', ' ').replace('–', ' ').replace('+', '').replace('2', ' ')
    # _sent = _sent.replace("'", "").replace('-', '')
    _tokens = nltk.word_tokenize(_sent)
    _tokens = [w.lower() for w in _tokens]
    return _tokens


def proc_text(text_data):
    """
    clean the text data
    :param text_data: list of string text
    :return:
    """
    vocab_set = set()
    for sent in text_data:
        tokens_lst = proc_sent(sent)
        vocab_set = vocab_set | set(tokens_lst)

    all_vocab_lst = list(vocab_set)
    all_vocab_lst.sort()

    # Remove not useful for classification words // optional
    # cleaned_vocab_lst = all_vocab_lst
    remove_lst = []
    cleaned_vocab_lst = []
    for w in all_vocab_lst:
        if w.isalpha():
            cleaned_vocab_lst.append(w)
        else:
            remove_lst.append(w)

    return all_vocab_lst, cleaned_vocab_lst, remove_lst


def output_txt_file(f_name, lst):
    """
    output a text file of the lst
    :param f_name: string of the file name
    :param lst: list of words
    :return: None
    """
    output_file = open('{}.txt'.format(f_name), "w", encoding="utf-8")
    output_file.write('\n'.join(str(word) for word in lst))
    output_file.close()
    return None


def diff_exp_proc_text(exp, new_proc_vocab_lst, new_remove_lst):
    """
    process text based on stopwords and words length
    :param exp: int of experiment number
    :param new_proc_vocab_lst: list of vocabulary
    :param new_remove_lst: list of remove words
    :return: list of cleaned vocabulary and remove list
    """
    # Experiment 1: stopwords filtering
    if exp == 1:
        print('\nExperiment 1: stopwords filtering.')
        # Read and process stopwords file
        with open("Stopwords.txt", "r", encoding="utf-8") as f:
            stopwords = f.read().splitlines()
        # Clean stopwords
        stopwords = list(map(lambda ele: ele.replace("'", "").replace('-', '').replace('_', ''), stopwords))

        # Update vocabulary list and remove list
        for w in new_proc_vocab_lst:
            if w in stopwords:
                new_proc_vocab_lst.remove(w)
                # new_remove_lst.append(w)

    # Experiment 2: word length filtering
    if exp == 2:
        print('\nExperiment 2: word length filtering.')
        # Update vocabulary list and remove list
        for w in new_proc_vocab_lst:
            if len(w) >= 9 or len(w) <= 3:
                new_proc_vocab_lst.remove(w)
                # new_remove_lst.append(w)
    return new_proc_vocab_lst, new_remove_lst


def get_cls_dataset(x_tr, y_tr):
    """
    divided dataset based on class type
    :param x_tr: list
    :param y_tr: list
    :return: samples sets and the number of samples of each class
    """
    cls_data = {}
    cls_data_nums = {}
    for cls in labels:
        samples_in_cls = x_tr[y_tr == cls]
        samples_in_cls.tolist()
        cls_data[cls] = samples_in_cls
        cls_data_nums[cls] = len(samples_in_cls)
    return cls_data, cls_data_nums


def get_sent_freq_vec(_vocab_lst, sent):
    """
    get frequency vector of a sentence based on vocabulary
    :param _vocab_lst: list of vocabulary
    :param sent: string of a sentence
    :return: list of frequency
    """
    freq_vec = [0] * len(_vocab_lst)
    tokens = proc_sent(sent)
    for word in tokens:
        if word in _vocab_lst:
            freq_vec[_vocab_lst.index(word)] += 1
    return freq_vec


def get_cls_freq_vec(vocab_lst, cls_samples):
    """
    get frequency vector of each class
    :param vocab_lst: list
    :param cls_samples: dictionary
    :return: dictionary
    """
    cls_freq_vec_dict = {k: list(np.sum([get_sent_freq_vec(vocab_lst, sent) for sent in v], axis=0))
                         for k, v in cls_samples.items()}
    return cls_freq_vec_dict


def get_cond_prob(freq, freq_vec):
    """
    calculate conditional probability
    :param freq: int of frequency of each word in class
    :param freq_vec: list of frequency of the class
    :return:
    """
    return float(freq + smoothed_value) / (sum(freq_vec) + len(cleaned_vocabulary) * smoothed_value)


def get_log_of_cond_prob(cond_prob_lst):
    """
    calculate the log value of a list of conditional probability
    :param cond_prob_lst: list of conditional probability
    :return: list of the log value
    """
    return list(map(lambda ele: math.log(ele, 10), cond_prob_lst))


def build_model(file_name, _freq_vec_dict, _cond_prob_dict):
    """
    model building
    :param file_name: string
    :param _freq_vec_dict: dictionary of frequency
    :param _cond_prob_dict: dictionary of conditional probability
    :return: dataFrame of the model
    """
    # Create the dataFrame of the model
    counter = [i + 1 for i in range(len(cleaned_vocabulary))]
    df_model_dict = {'counter': counter, 'word': cleaned_vocabulary}
    for cls in labels:
        df_model_dict.update({'{}_freq'.format(cls): _freq_vec_dict[cls],
                              '{}_cond_prob'.format(cls): _cond_prob_dict[cls]})
    df_result = pd.DataFrame(df_model_dict)
    # Output model to text file
    output_file = open('{}.txt'.format(file_name), "w", encoding="utf-8")
    for i, r in df_result.iterrows():
        line_str = [str(i) for i in r.tolist()]
        line_str = '  '.join(line_str)
        output_file.write(line_str + '\r\n')
    output_file.close()
    # df_result.to_csv('{}.txt'.format(file_name), sep=' ', index=False, header=False)
    return df_result


def get_scores_dict(freq_vec):
    """
    calculate the scores of each class
    :param freq_vec: list of the words frequency
    :return: dictionary of scores of classes
    """
    score_dict = {}
    for cls in labels:
        # calculate scores
        score = sum([freq_vec[i] * cls_cond_prob_log_dict[cls][i] for i in range(len(freq_vec)) if freq_vec[i] != 0])
        score += math.log(cls_sample_nums[cls] / sum(list(cls_sample_nums.values())), 10)
        score_dict[cls] = score
    return score_dict


def predict(_score_dict):
    """
    predict the label by choosing the highest score value
    :param _score_dict: dictionary of scores of each class
    :return: string of class type
    """
    score_sorted_lst = sorted(_score_dict.items(), key=lambda item: item[1], reverse=True)
    return score_sorted_lst[0][0]


def cal_accuracy(_x_test, _y_test):
    """
    calculate accuracy
    :param _x_test: list of test dataset
    :param _y_test: list of labels of test
    :return: float of accuracy
    """
    predict_lst = []
    test_data_lst = _x_test.tolist()
    true_label_lst = _y_test.tolist()

    for title in test_data_lst:
        title_freq_vec = get_sent_freq_vec(cleaned_vocabulary, title)
        title_score_dict = get_scores_dict(title_freq_vec)
        cls_predict = predict(title_score_dict)
        predict_lst.append(cls_predict)

    n_total = len(true_label_lst)
    acc = sum([predict_lst[i] == true_label_lst[i] for i in range(n_total)]) / n_total
    return acc


def naive_bayes_classifier(result_file_name, _x_test, _y_test):
    """
    implement Naive Bayes classifier and classify test dataset
    :param result_file_name: string name of the output result
    :param _x_test: list of the test dataset
    :param _y_test: list of the labels of the test dataset
    :return: DataFrame of the result model and the accuracy
    """
    # Output the result
    baseline_file = open('{}.txt'.format(result_file_name), "w", encoding="utf-8")
    counter = [i + 1 for i in range(len(_x_test))]
    scores_lst = []
    predicts_lst = []
    correct_lst = _y_test.tolist()
    label_lst = []
    scores_val_lst = []
    test_samples_lst = _x_test.tolist()

    # Calculate the score of each title and output the result
    for i in range(len(test_samples_lst)):
        # get the vocabulary vector of the title
        title_freq_vec = get_sent_freq_vec(cleaned_vocabulary, test_samples_lst[i])
        # calculate scores
        title_score_dict = get_scores_dict(title_freq_vec)
        # save the scores to list
        scores_val_lst += [title_score_dict]
        # join score string for output
        scores_str = '  '
        for c in labels:
            scores_str += str(title_score_dict[c]) + '  '
        scores_lst.append(title_score_dict)
        # predict the class type
        cls_predict = predict(title_score_dict)
        predicts_lst.append(cls_predict)
        # verify the predict right or wrong
        label_lst.append('right') if cls_predict == correct_lst[i] else label_lst.append('wrong')

        # Output the result file
        line_str = ''
        line_str = line_str + str(counter[i]) + '  ' + str(test_samples_lst[i]) + '  ' + str(
            predicts_lst[i]) + scores_str + str(correct_lst[i]) + '  ' + str(label_lst[i])
        baseline_file.write(line_str + '\r\n')
    baseline_file.close()

    # Calculate accuracy
    accuracy = pd.value_counts(label_lst)['right']/len(label_lst)

    # Create the DataFrame of the result
    df_result_dict_l = {'counter': counter, 'title': test_samples_lst, 'predict_cls': predicts_lst}
    df_result_dict_r = {'real_cls': correct_lst, 'is_label_correct': label_lst}
    df_l = pd.DataFrame(df_result_dict_l)
    df_r = pd.DataFrame(df_result_dict_r)
    df_scores = pd.DataFrame(scores_val_lst)
    df_result = pd.concat([df_l, df_scores, df_r], axis=1)

    return df_result, accuracy


def run_main():
    """
    main function
    :return: None
    """
    global cls_cond_prob_log_dict
    # ----- Task 1: Extract the data and build the model -----
    # 1. Read the data and attract features and labels
    print('\n---- Task 1: Extract the data and build the model ----\n')
    X_train, y_train, X_test, y_test = read_and_divide_data()

    # 2. Divided dataset
    print('1. Dividing dataset.')
    # Get labels
    global labels
    labels = list(set(y_train.tolist() + y_test.tolist()))

    # Get samples sets and the number of samples in each class
    global cls_sample_nums
    cls_samples, cls_sample_nums = get_cls_dataset(X_train, y_train)

    # 3. Process and clean dataset
    # Prompt user to input the experiment number
    while True:
        exp = input('\nPlease enter the experiment number (1/2/3), or press enter get baseline experiment: ')
        if exp == '':
            exp = 0
            break
        else:
            try:
                exp = int(exp)
            except ValueError:
                print('Invalid input!')
                continue

        if exp <= 0 or exp > 3:
            print('Input invalid, please input again!')
        else:
            break

    print('\n2. Processing and cleaning dataset ... ')
    global cleaned_vocabulary
    vocabulary, cleaned_vocabulary, remove_word = proc_text(X_train)

    if exp == 1 or exp == 2:
        cleaned_vocabulary, remove_word = diff_exp_proc_text(exp, cleaned_vocabulary, remove_word)
    elif exp == 3:
        print('\nExperiment 3: Infrequent Word Filtering.')
        frequency_lst = [1, 5, 10, 15, 20]
        top_n_lst = [0.05, 0.1, 0.15, 0.2, 0.25]
        freq_accuracy_lst = []
        top_n_accuracy_lst = []

        # Calculate each word's frequency of vocabulary
        cls_freq_vec_dict = get_cls_freq_vec(cleaned_vocabulary, cls_samples)
        vocab_freq_vec = list(np.sum([i for i in cls_freq_vec_dict.values()], axis=0))
        vocab_freq_tup_lst = list(zip(vocab_freq_vec, cleaned_vocabulary))
        vocab_freq_tup_lst.sort(reverse=True)

        # Get the index of each case
        freq_index_lst = []
        for val in frequency_lst:
            for tup in vocab_freq_tup_lst:
                if tup[0] <= val:
                    freq_index_lst.append(vocab_freq_tup_lst.index(tup))
                    break
        num_words_left_freq = freq_index_lst

        top_n_index_lst = [int(i * len(vocab_freq_tup_lst)) for i in top_n_lst]
        num_words_left_top_n = [len(vocab_freq_tup_lst) - i for i in top_n_index_lst]

        # Calculate accuracy
        sorted_vocabulary = [tup[1] for tup in vocab_freq_tup_lst]
        con_index_lst = freq_index_lst + top_n_index_lst

        for i in range(len(con_index_lst)):
            if i < len(freq_index_lst):
                cleaned_vocabulary = sorted_vocabulary[:con_index_lst[i]]
                # remove_word += sorted_vocabulary[con_index_lst[i]:]
            else:
                cleaned_vocabulary = sorted_vocabulary[con_index_lst[i]:]

            # Get the frequency vector of each class
            cls_freq_vec_dict = get_cls_freq_vec(cleaned_vocabulary, cls_samples)
            # Get words' conditional probability of each class
            cls_cond_prob_dict = {k: [get_cond_prob(i, v) for i in v] for k, v in cls_freq_vec_dict.items()}
            cls_cond_prob_log_dict = {k: get_log_of_cond_prob(v) for k, v in cls_cond_prob_dict.items()}
            # Calculate accuracy
            acc = cal_accuracy(X_test, y_test)
            freq_accuracy_lst.append(acc) if i < len(freq_index_lst) else top_n_accuracy_lst.append(acc)

        # Draw performance graphs
        plt.figure(1)

        plt.subplot(211)
        plt.xlabel('Number Of Words left')
        plt.ylabel('Accuracy')
        plt.title("EXPERIMENT 3 RESULTS")
        plt.plot(freq_index_lst, freq_accuracy_lst, linestyle='--', marker='o', color='b')

        plt.subplot(212)
        plt.xlabel('Number Of Words left')
        plt.ylabel('Accuracy')
        plt.plot(num_words_left_top_n, top_n_accuracy_lst, linestyle='--', marker='o', color='g')

        plt.show()

        print('\n======== Program terminated! ========\n')
        return None

    print('\n... Dataset has been cleaned!\n')

    # Output vocabulary list and removed word list
    output_txt_file('vocabulary', vocabulary)
    output_txt_file('remove_word', remove_word)

    # 4. Calculate the frequency and the conditional probability
    print('3. Calculating the frequency and the conditional probability.')
    # Get the frequency vector of each class
    cls_freq_vec_dict = {k: list(np.sum([get_sent_freq_vec(cleaned_vocabulary, sent) for sent in v], axis=0))
                         for k, v in cls_samples.items()}

    # Get words' conditional probability of each class
    cls_cond_prob_dict = {k: [get_cond_prob(i, v) for i in v] for k, v in cls_freq_vec_dict.items()}
    cls_cond_prob_log_dict = {k: get_log_of_cond_prob(v) for k, v in cls_cond_prob_dict.items()}

    # 5. Build model
    # Build model based on different experiment
    print('\n4. Building model.')
    model_name = 'model-2018'
    result_name = 'baseline-result'
    if exp == 1:
        model_name = 'stopword-model'
        result_name = 'stopword-result'
    elif exp == 2:
        model_name = 'wordlength-model'
        result_name = 'wordlength-result'
    nb_model = build_model(model_name, cls_freq_vec_dict, cls_cond_prob_dict)

    # ----- Task 2: Use ML Classifier to test dataset -----
    print('\n---- Task 2: Using Naïve Bays Classifier to test and predict dataset ----\n')
    nb_result_model, accuracy = naive_bayes_classifier(result_name, X_test, y_test)
    print('The accuracy is ', accuracy)

    # ----- Task 3: Experiments with the classifier -----
    print('\n---- Task 3: Experiments with the classifier ----\n')
    # New models have been output above based on the user input

    print('\n======== Program finished! ========.')


if __name__ == '__main__':
    run_main()








