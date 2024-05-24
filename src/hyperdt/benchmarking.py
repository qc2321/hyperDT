import numpy as np
import matplotlib.pyplot as plt
from hyperdt.product_space_DT import ProductSpace, ProductSpaceDT
from sklearn.tree import DecisionTreeClassifier

"""
Methods for computing and plotting accuracy scores across signatures
"""
def compute_scores(signature, n, num_classes, seed=None, cov_scale=0.3, max_depth=3):
    # Generate data
    ps = ProductSpace(signature, seed=seed)
    ps.sample_clusters(n, num_classes, cov_scale=cov_scale)
    ps.split_data()

    # Fit hyperspace decision tree classifier
    psdt = ProductSpaceDT(product_space=ps, max_depth=max_depth)
    psdt.fit()
    psdt_score = psdt.score(ps.X_test, ps.y_test)

    # Fit sklearn's decision tree classifier
    X_train, X_test, y_train, y_test = ps.X_train, ps.X_test, ps.y_train, ps.y_test
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)
    dt_score = dt.score(X_test, y_test)

    return psdt_score, dt_score

def compute_scores_by_signature(signatures, n, num_classes, seed=None,
                                cov_scale=0.3, max_depth=3):
    rng = np.random.default_rng(seed)
    rnd_seeds = rng.integers(0, 1000, 10)
    # rnd_seeds = np.array([0, 1, 2, 10, 42, 123, 137, 1234, 12345])

    psdt_scores_by_signature = []
    dt_scores_by_signature = []
    for signature in signatures:
        psdt_scores = []
        dt_scores = []
        for rnd_seed in rnd_seeds:
            psdt_score, dt_score = compute_scores(signature, n, num_classes, seed=rnd_seed,
                                                  cov_scale=cov_scale, max_depth=max_depth)
            psdt_scores.append(psdt_score)
            dt_scores.append(dt_score)
        psdt_scores_by_signature.append(psdt_scores)
        dt_scores_by_signature.append(dt_scores)

    return rnd_seeds, psdt_scores_by_signature, dt_scores_by_signature

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


"""
Methods for plotting difference in accuracy scores for different signatures across random seeds
"""
def sort_scores(rnd_seeds, psdt_scores, dt_scores):
    sorted_idx = np.argsort(rnd_seeds)
    sorted_rnd_seeds = rnd_seeds[sorted_idx]
    sorted_psdt_scores = np.array(psdt_scores)[sorted_idx]
    sorted_dt_scores = np.array(dt_scores)[sorted_idx]
    return sorted_rnd_seeds, sorted_psdt_scores, sorted_dt_scores

def get_sorted_scores_by_signature(rnd_seeds, psdt_scores_by_signature, dt_scores_by_signature):
    sorted_psdt_scores_by_signature = []
    sorted_dt_scores_by_signature = []
    sorted_rnd_seeds = None
    for psdt_scores, dt_scores in zip(psdt_scores_by_signature, dt_scores_by_signature):
        sorted_rnd_seeds, sorted_psdt_scores, sorted_dt_scores = sort_scores(rnd_seeds, psdt_scores, dt_scores)
        sorted_psdt_scores_by_signature.append(sorted_psdt_scores)
        sorted_dt_scores_by_signature.append(sorted_dt_scores)
    return sorted_rnd_seeds, sorted_psdt_scores_by_signature, sorted_dt_scores_by_signature

def compute_diff_scores(sorted_psdt_scores_by_signature, sorted_dt_scores_by_signature):
    return [psdt_score - dt_score for psdt_score, dt_score in
            zip(sorted_psdt_scores_by_signature, sorted_dt_scores_by_signature)]

def plot_diff_scores(sorted_rnd_seeds, diff_scores, signatures):
    plt.figure(figsize=(10, 6))
    for i, diff_score in enumerate(diff_scores):
        plt.plot(sorted_rnd_seeds, diff_score, label=f'{sig_as_str(signatures[i])}')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Random Seed')
    plt.ylabel('PSDT Score - DT Score')
    plt.legend()
    plt.title('Difference between PSDT and DT scores for different signatures')
    plt.show()
