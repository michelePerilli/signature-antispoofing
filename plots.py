import numpy as np
import matplotlib.pyplot as plt


def determine_err(scores):
    far_array = np.array([])
    frr_array = np.array([])

    scores = sorted(scores)  # min->max
    sort_score = np.array(scores)
    minIndex = np.iinfo(np.int16).max
    minDis = np.iinfo(np.int16).max
    minTh = np.iinfo(np.int16).max
    alltrue = sort_score.sum(0)[1]
    allfalse = len(scores) - alltrue
    eer = np.iinfo(np.int16).max
    far = np.iinfo(np.int16).max
    frr = np.iinfo(np.int16).max
    fa = allfalse
    miss = 0

    for i in range(0, len(scores)):
        # min -> max
        if sort_score[i, 1] == 1:
            miss += 1
        else:
            fa -= 1

        fa_rate = float(fa) / allfalse
        miss_rate = float(miss) / alltrue
        far_array = np.append(far_array, fa_rate)
        frr_array = np.append(frr_array, miss_rate)

        if abs(fa_rate - miss_rate) < minDis:
            minDis = abs(fa_rate - miss_rate)
            far = fa_rate
            frr = miss_rate
            eer = max(fa_rate, miss_rate)
            minIndex = i
            minTh = sort_score[i, 0]

    print('FAR:', far, 'FRR:', frr, 'ERR:', eer, 'Minimum threshold:', minTh)
    return far_array, frr_array, minIndex


def get_stats(result_dir):
    stats = dict()
    ds = open(result_dir).readlines()
    ds.pop(0)  # Remove first row user for csv column name
    ds = [s.strip().split(',') for s in ds]
    mScores = [[float(s[3]), 1 if s[2] == 'G' else 0] for s in ds]
    stats['far'], stats['frr'], stats['eer_idx'] = determine_err(mScores)
    stats['thresholds'] = [float(s[3]) for s in ds]
    stats['thresholds'].sort()

    return stats


def plot_eer(title, stats_base, stats_gan, stats_def):
    plt.title(title)
    plt.xlabel('Threshold')

    plt.plot(stats_base['thresholds'], stats_base['far'], label='FAR base', color='green', linestyle=':')
    plt.plot(stats_gan['thresholds'], stats_gan['far'], label='FAR attacco', color='red', linestyle=':')
    plt.plot(stats_def['thresholds'], stats_def['far'], label='FAR difesa', color='blue', linestyle='-')

    plt.plot(stats_base['thresholds'], stats_base['frr'], label='FRR')

    eer_idx = stats_base['eer_idx']
    plt.plot(stats_base['thresholds'][eer_idx], stats_base['far'][eer_idx], marker='o', color='green', label='EER base')

    eer_idx = stats_gan['eer_idx']
    plt.plot(stats_gan['thresholds'][eer_idx], stats_gan['far'][eer_idx], marker='o', color='red', label='EER attacco')

    eer_idx = stats_def['eer_idx']
    plt.plot(stats_def['thresholds'][eer_idx], stats_def['far'][eer_idx], marker='o', color='blue', label='EER difesa')

    plt.legend()
    plt.show()


def plot_three(title, test1, test2, test3, marker):
    plt.title(title)

    plt.plot(test1['far'], test1['frr'], label='Test 1')
    plt.plot(test1['far'][test1['eer_idx']], test1['frr'][test1['eer_idx']], marker=marker,
             color='blue', ls='', label='EER')

    plt.plot(test2['far'], test2['frr'], label='Test 2')
    plt.plot(test2['far'][test2['eer_idx']], test2['frr'][test2['eer_idx']], marker=marker,
             color='orange', ls='', label='EER')

    plt.plot(test3['far'], test3['frr'], label='Test 3')
    plt.plot(test3['far'][test3['eer_idx']], test3['frr'][test3['eer_idx']], marker=marker,
             color='green', ls='', label='EER')

    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.legend()
    plt.show()


test_base = get_stats('testResult\\test_base.txt')
test_atk = get_stats('testResult\\test_attacco.txt')
test_def = get_stats('testResult\\test_difesa.txt')

plot_eer('', test_base, test_atk, test_def)


