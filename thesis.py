import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind


def re_draw_satire_paper():
    p = [0.573, 0.616, 0.634, 0.989]
    d = [0.486, 0.746, 0.461, 0.598]
    name = ['Psycholinguistic\nFeature', 'Readability\nFeature', 'Writing Stylistic\nFeature', 'Structural\nFeature']
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(name)) - width / 2, p, 0.2, label='Importance at the paragraph level', color='b')
    ax.bar(np.arange(len(name)) + width / 2, d, 0.2, label='Importance at the document level', color='r')
    ax.set_ylabel('Importance score')
    ax.set_xticks(np.arange(len(name)))
    ax.set_xticklabels(name)
    ax.legend()
    plt.show()


def ver2ret_significant():
    mrr_r_5 = [85.17, 85.3764, 85.0241, 85.2324, 84.9632]
    mrr_rj_5 = [85.4398, 85.4079, 85.6286, 85.60, 85.6533, 85.8101]
    mrr_rvj_5 = [85.98, 86.0141, 85.8174, 85.7481, 85.4897]

    map_r_5 = [84.07, 84.2788, 83.9239, 84.0586, 83.8272]
    map_rj_5 = [84.3387, 84.3241, 84.5064, 84.69, 84.4884, 84.6920]
    map_rvj_5 = [84.86, 84.8975, 84.6501, 84.5425, 84.3809]

    mrr_r_a = [85.63, 85.7765, 85.4238, 85.5919, 85.3454]
    mrr_rj_a = [85.7726, 85.7584, 85.9850, 85.82, 85.9111, 86.1464]
    mrr_rvj_a = [86.31, 86.3593, 86.1556, 86.0731, 85.8687]

    map_r_a = [82.49, 82.7915, 82.2958, 82.5664, 82.2746]
    map_rj_a = [82.8894, 82.7562, 82.9002, 82.83, 82.9220, 83.1151]
    map_rvj_a = [83.29, 83.4660, 83.1968, 83.0700, 82.9841]

    print('\nmrr rvj-r 5', np.mean(mrr_r_5), np.std(mrr_r_5),
          np.mean(mrr_rvj_5), np.std(mrr_rvj_5),
          ttest_ind(mrr_r_5, mrr_rvj_5, equal_var=False))

    print('\nmap rvj-r 5', np.mean(map_r_5), np.std(map_r_5),
          np.mean(map_rvj_5), np.std(map_rvj_5),
          ttest_ind(map_r_5, map_rvj_5, equal_var=False))

    print('\nmrr rvj-r a', np.mean(mrr_r_a), np.std(mrr_r_a),
          np.mean(mrr_rvj_a), np.std(mrr_rvj_a),
          ttest_ind(mrr_r_a, mrr_rvj_a, equal_var=False))

    print('\nmap rvj-r a', np.mean(map_r_a), np.std(map_r_a),
          np.mean(map_rvj_a), np.std(map_rvj_a),
          ttest_ind(map_r_a, map_rvj_a, equal_var=False))

    """ SPLIT """
    print('\nmrr rj-rvj 5', np.mean(mrr_rj_5), np.std(mrr_rj_5),
          np.mean(mrr_rvj_5), np.std(mrr_rvj_5),
          ttest_ind(mrr_rj_5, mrr_rvj_5, equal_var=False))

    print('\nmap rj-rvj 5', np.mean(map_rj_5), np.std(map_rj_5),
          np.mean(map_rvj_5), np.std(map_rvj_5),
          ttest_ind(map_rj_5, map_rvj_5, equal_var=False))

    print('\nmrr rj-rvj a', np.mean(mrr_rj_a), np.std(mrr_rj_a),
          np.mean(mrr_rvj_a), np.std(mrr_rvj_a),
          ttest_ind(mrr_rj_a, mrr_rvj_a, equal_var=False))

    print('\nmap rj-rvj a', np.mean(map_rj_a), np.std(map_rj_a),
          np.mean(map_rvj_a), np.std(map_rvj_a),
          ttest_ind(map_rj_a, map_rvj_a, equal_var=False))

    """ SPLIT """
    print('\nmrr rj-r 5', np.mean(mrr_r_5), np.std(mrr_r_5),
          np.mean(mrr_rj_5), np.std(mrr_rj_5),
          ttest_ind(mrr_r_5, mrr_rj_5, equal_var=False))

    print('\nmap rj-r 5', np.mean(map_r_5), np.std(map_r_5),
          np.mean(map_rj_5), np.std(map_rj_5),
          ttest_ind(map_r_5, map_rj_5, equal_var=False))

    print('\nmrr rj-r a', np.mean(mrr_r_a), np.std(mrr_r_a),
          np.mean(mrr_rj_a), np.std(mrr_rj_a),
          ttest_ind(mrr_r_a, mrr_rj_a, equal_var=False))

    print('\nmap rj-r a', np.mean(map_r_a), np.std(map_r_a),
          np.mean(map_rj_a), np.std(map_rj_a),
          ttest_ind(map_r_a, map_rj_a, equal_var=False))


def paper_count():
    y_ = {2015: {'Fake news': 14, 'Disinformation': 41,
                 'Misinformation': 154, 'Hoax': 78,
                 'News satire': 14, 'Rumor': 269,
                 'Click-bait': 14},
          2016: {'Fake news': 391, 'Disinformation': 62,
                 'Misinformation': 145, 'Hoax': 74,
                 'News satire': 18, 'Rumor': 256,
                 'Click-bait': 41},
          2017: {'Fake news': 1410, 'Disinformation': 133,
                 'Misinformation': 219, 'Hoax': 194,
                 'News satire': 22, 'Rumor': 280,
                 'Click-bait': 52},
          2018: {'Fake news': 1720, 'Disinformation': 214,
                 'Misinformation': 318, 'Hoax': 274,
                 'News satire': 21, 'Rumor': 261,
                 'Click-bait': 65},
          2019: {'Fake news': 1610, 'Disinformation': 317,
                 'Misinformation': 414, 'Hoax': 327,
                 'News satire': 17, 'Rumor': 294,
                 'Click-bait': 56}}
    t_ = {'Fake news': {2015: 14,
                        2016: 391, 2017: 1410,
                        2018: 1720, 2019: 1610},
          'Disinformation': {2015: 41,
                             2016: 62, 2017: 133,
                             2018: 214, 2019: 317},
          'Misinformation': {2015: 154,
                             2016: 145, 2017: 219,
                             2018: 318, 2019: 414},
          'Hoax': {2015: 78,
                   2016: 74, 2017: 194,
                   2018: 274, 2019: 327},
          'News satire': {2015: 14,
                          2016: 18, 2017: 22,
                          2018: 21, 2019: 17},
          'Rumor': {2015: 269,
                    2016: 256, 2017: 280,
                    2018: 261, 2019: 294},
          'Click-bait': {2015: 14,
                         2016: 41, 2017: 52,
                         2018: 65, 2019: 56},
          'Click-bait & News satire': {2015: 28, 2016: 59,
                                       2017: 74, 2018: 86, 2019: 73}}

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 4),
                           constrained_layout=True)
    ax[0, 0].bar([1, 2, 3, 4, 5], t_['Fake news'].values())
    ax[0, 0].set_ylabel('Number of Papers')
    ax[0, 0].set_title('Fake News')
    ax[0, 0].set_xticks([1, 2, 3, 4, 5])
    ax[0, 0].set_xticklabels([2015, '', 2017, '', 2019])

    ax[0, 1].bar([1, 2, 3, 4, 5], t_['Disinformation'].values())
    ax[0, 1].set_title('Disinformation')
    ax[0, 1].set_xticks([1, 2, 3, 4, 5])
    ax[0, 1].set_xticklabels([2015, '', 2017, '', 2019])

    ax[0, 2].bar([1, 2, 3, 4, 5], t_['Misinformation'].values())
    ax[0, 2].set_title('Misinformation')
    ax[0, 2].set_xticks([1, 2, 3, 4, 5])
    ax[0, 2].set_xticklabels([2015, '', 2017, '', 2019])

    ax[1, 0].bar([1, 2, 3, 4, 5], t_['Hoax'].values())
    ax[1, 0].set_ylabel('Number of Paper')
    ax[1, 0].set_xticks([1, 2, 3, 4, 5])
    ax[1, 0].set_xticklabels([2015, '', 2017, '', 2019])
    ax[1, 0].set_title('Hoax')

    ax[1, 1].bar([1, 2, 3, 4, 5], t_['Rumor'].values())
    ax[1, 1].set_title('Rumor or Rumour')
    ax[1, 1].set_xticks([1, 2, 3, 4, 5])
    ax[1, 1].set_xticklabels([2015, '', 2017, '', 2019])

    ax[1, 2].bar([1, 2, 3, 4, 5], t_['Click-bait & News satire'].values())
    ax[1, 2].set_title('Click-bait or News Satire')
    ax[1, 2].set_xticks([1, 2, 3, 4, 5])
    ax[1, 2].set_xticklabels([2015, '', 2017, '', 2019])

    plt.show()


if __name__ == '__main__':
    # ver2ret_significant()
    paper_count()
