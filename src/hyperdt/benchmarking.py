import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from hyperdt.product_space_DT import ProductSpace, ProductSpaceDT
from sklearn.tree import DecisionTreeClassifier
from hyperdt.forest import ProductSpaceRF
from sklearn.ensemble import RandomForestClassifier


"""
Methods for computing and plotting accuracy scores across signatures
"""
def compute_scores(signature, n, num_classes, seed=None, cov_scale=0.3, max_depth=3):
    # Generate data
    ps = ProductSpace(signature, seed=seed)
    ps.sample_clusters(n, num_classes, cov_scale=cov_scale)
    ps.split_data()

    # Fit ProductSpaceDT
    psdt = ProductSpaceDT(signature, max_depth=max_depth)
    psdt.fit(ps.X_train, ps.y_train)
    psdt_score = psdt.score(ps.X_test, ps.y_test)

    # Fit ProductSpaceRF
    psrf = ProductSpaceRF(signature, max_depth=max_depth, n_estimators=12)
    psrf.fit(ps.X_train, ps.y_train)
    psrf_score = psrf.score(ps.X_test, ps.y_test)

    # Fit sklearn's decision tree classifier
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(ps.X_train, ps.y_train)
    dt_score = dt.score(ps.X_test, ps.y_test)

    # Fit sklearn's random forest classifier
    rf = RandomForestClassifier(n_estimators=12, max_depth=max_depth)
    rf.fit(ps.X_train, ps.y_train)
    rf_score = rf.score(ps.X_test, ps.y_test)

    return psdt_score, psrf_score, dt_score, rf_score


def compute_scores_by_signature(signatures, n, num_classes, seed=None,
                                cov_scale=0.3, max_depth=3, n_seeds=10):
    rng = np.random.default_rng(seed)
    rnd_seeds = rng.integers(0, 100000, n_seeds)

    psdt_scores_by_signature = []
    psrf_scores_by_signature = []
    dt_scores_by_signature = []
    rf_scores_by_signature = []
    
    my_tqdm = tqdm(total=len(signatures) * n_seeds)
    for signature in signatures:
        psdt_scores = []
        psrf_scores = []
        dt_scores = []
        rf_scores = []

        for rnd_seed in rnd_seeds:
            score_tuple = compute_scores(signature, n, num_classes, seed=rnd_seed,
                                         cov_scale=cov_scale, max_depth=max_depth)
            psdt_scores.append(score_tuple[0])
            psrf_scores.append(score_tuple[1])
            dt_scores.append(score_tuple[2])
            rf_scores.append(score_tuple[3])
            my_tqdm.update(1)
        
        psdt_scores_by_signature.append(psdt_scores)
        psrf_scores_by_signature.append(psrf_scores)
        dt_scores_by_signature.append(dt_scores)
        rf_scores_by_signature.append(rf_scores)

    return (rnd_seeds, psdt_scores_by_signature, psrf_scores_by_signature,
            dt_scores_by_signature, rf_scores_by_signature)


def compute_avg_scores(psdt_scores_by_signature, dt_scores_by_signature):
    avg_psdt_scores = [np.mean(scores) for scores in psdt_scores_by_signature]
    avg_dt_scores = [np.mean(scores) for scores in dt_scores_by_signature]
    return avg_psdt_scores, avg_dt_scores


def sig_as_str(sig):
    result = ""
    for i, space in enumerate(sig):
        if space[1] < 0:
            result += f"H{space[0]}(K={space[1]})"
        elif space[1] > 0:
            result += f"S{space[0]}(K={space[1]})"
        else:
            result += f"E{space[0]}"
        if i < len(sig) - 1:
            result += " x "
    return result


def plot_avg_scores(signatures, avg_psdt_scores, avg_dt_scores):
    plt.figure(figsize=(10, 6))
    plt.plot([sig_as_str(sig) for sig in signatures], avg_psdt_scores, label='PSDT')
    plt.plot([sig_as_str(sig) for sig in signatures], avg_dt_scores, label='DT')
    plt.xlabel('Signature')
    plt.ylabel('Average Score')
    plt.xticks(rotation=45)
    plt.legend()
    plt.title('Average scores for different signatures')
    plt.show()


def plot_boxplots(signatures, psdt_scores_by_signature, dt_scores_by_signature):
    plt.figure(figsize=(10, 5))
    boxprops_hyperdt = dict(color='blue', linewidth=2)
    boxprops_sklearn = dict(color='red', linewidth=2)
    bp1 = plt.boxplot(psdt_scores_by_signature, positions=np.arange(len(signatures)) - 0.2, widths=0.4, boxprops=boxprops_hyperdt)
    bp2 = plt.boxplot(dt_scores_by_signature, positions=np.arange(len(signatures)) + 0.2, widths=0.4, boxprops=boxprops_sklearn)
    plt.xticks(range(len(signatures)), [sig_as_str(sig) for sig in signatures], rotation=45)
    plt.xlabel('Signature')
    plt.ylabel('Accuracy')
    plt.title('HyperDT vs DT accuracy by signature')
    plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['HyperDT', 'Sklearn'])
    plt.show()

