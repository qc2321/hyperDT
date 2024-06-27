import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from .product_space_DT import ProductSpace, ProductSpaceDT
from .forest import ProductSpaceRF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from .product_space_perceptron import mix_curv_perceptron
from .product_space_svm import mix_curv_svm
from sklearn.metrics import f1_score
from numpy.linalg import norm


"""
Methods for computing and plotting accuracy or F1 scores across signatures
"""
def compute_scores(signature, n, num_classes, seed=None, cov_scale=0.3, max_depth=3,
                   metric='accuracy'):
    """Compute scores for a given signature"""
    # Generate data
    ps = ProductSpace(signature, seed=seed)
    ps.sample_clusters(n, num_classes, cov_scale=cov_scale)
    ps.split_data()

    # Fit ProductSpaceDT
    psdt = ProductSpaceDT(signature, max_depth=max_depth)
    psdt.fit(ps.X_train, ps.y_train)
    if metric == 'accuracy':
        psdt_score = psdt.score(ps.X_test, ps.y_test)
    elif metric == 'f1':
        psdt_score = f1_score(ps.y_test, psdt.predict(ps.X_test), average='macro')

    # Fit ProductSpaceRF
    psrf = ProductSpaceRF(signature, max_depth=max_depth, n_estimators=12)
    psrf.fit(ps.X_train, ps.y_train)
    if metric == 'accuracy':
        psrf_score = psrf.score(ps.X_test, ps.y_test)
    elif metric == 'f1':
        psrf_score = f1_score(ps.y_test, psrf.predict(ps.X_test), average='macro')

    # Fit sklearn's decision tree classifier
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(ps.X_train, ps.y_train)
    if metric == 'accuracy':
        dt_score = dt.score(ps.X_test, ps.y_test)
    elif metric == 'f1':
        dt_score = f1_score(ps.y_test, dt.predict(ps.X_test), average='macro')

    # Fit sklearn's random forest classifier
    rf = RandomForestClassifier(n_estimators=12, max_depth=max_depth)
    rf.fit(ps.X_train, ps.y_train)
    if metric == 'accuracy':
        rf_score = rf.score(ps.X_test, ps.y_test)
    elif metric == 'f1':
        rf_score = f1_score(ps.y_test, rf.predict(ps.X_test), average='macro')

    # Fit product space perceptron (F1 score only)
    mix_component = sig_to_mix_component(signature)
    embed_data = make_embed_data(ps.X, ps.X_train, ps.X_test, ps.y_train, ps.y_test, signature)
    ps_perc = mix_curv_perceptron(mix_component, embed_data, multiclass=True, max_round=100, max_update=1000)
    ps_perc_score = ps_perc.process_data()

    # Fit product space SVM (F1 score only)
    ps_svm = mix_curv_svm(mix_component, embed_data)
    ps_svm_score = ps_svm.process_data()

    return psdt_score, psrf_score, dt_score, rf_score, ps_perc_score, ps_svm_score


def compute_scores_by_signature(signatures, n, num_classes, seed=None, cov_scale=0.3,
                                max_depth=3, n_seeds=10, metric='accuracy'):
    """Compute scores for each signature across multiple random seeds"""
    rng = np.random.default_rng(seed)
    rnd_seeds = rng.integers(0, 100000, n_seeds)

    psdt_scores_by_signature = []
    psrf_scores_by_signature = []
    dt_scores_by_signature = []
    rf_scores_by_signature = []
    ps_perc_scores_by_signature = []
    ps_svm_scores_by_signature = []
    
    my_tqdm = tqdm(total=len(signatures) * n_seeds)
    for signature in signatures:
        psdt_scores = []
        psrf_scores = []
        dt_scores = []
        rf_scores = []
        ps_perc_scores = []
        ps_svm_scores = []

        for rnd_seed in rnd_seeds:
            score_tuple = compute_scores(signature, n, num_classes, seed=rnd_seed,
                                         cov_scale=cov_scale, max_depth=max_depth,
                                         metric=metric)
            psdt_scores.append(score_tuple[0])
            psrf_scores.append(score_tuple[1])
            dt_scores.append(score_tuple[2])
            rf_scores.append(score_tuple[3])
            ps_perc_scores.append(score_tuple[4])
            ps_svm_scores.append(score_tuple[5])
            my_tqdm.update(1)
        
        psdt_scores_by_signature.append(psdt_scores)
        psrf_scores_by_signature.append(psrf_scores)
        dt_scores_by_signature.append(dt_scores)
        rf_scores_by_signature.append(rf_scores)
        ps_perc_scores_by_signature.append(ps_perc_scores)
        ps_svm_scores_by_signature.append(ps_svm_scores)

    return (rnd_seeds, psdt_scores_by_signature, psrf_scores_by_signature,
            dt_scores_by_signature, rf_scores_by_signature, ps_perc_scores_by_signature,
            ps_svm_scores_by_signature)


def compute_avg_scores(psdt_scores_by_signature, dt_scores_by_signature):
    """Compute average scores for each signature across seeds"""
    avg_psdt_scores = [np.mean(scores) for scores in psdt_scores_by_signature]
    avg_dt_scores = [np.mean(scores) for scores in dt_scores_by_signature]
    return avg_psdt_scores, avg_dt_scores


def sig_as_str(sig):
    """Convert a signature to a string representation"""
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


def sig_to_mix_component(sig):
    """Convert a signature to mix_component for perceptron and SVM"""
    result = []
    for space in sig:
        if space[1] < 0:
            result.append(f"h{space[0]}")
        elif space[1] > 0:
            result.append(f"s{space[0]}")
        else:
            result.append(f"e{space[0]}")
    return ",".join(result)


def make_embed_data(X, X_train, X_test, y_train, y_test, sig):
    """Create a dictionary of embedding data for perceptron and SVM"""
    embed_data = {}
    embed_data['X_train'] = X_train
    embed_data['X_test'] = X_test
    embed_data['y_train'] = y_train
    embed_data['y_test'] = y_test
    embed_data['curv_value'] = [abs(space[1]) for space in sig]
    max_norm = []
    for i in range(len(sig)):
        component_data = get_component_data(X, sig, i)
        max_norm.append(norm(component_data, axis=1).max())
    embed_data['max_norm'] = max_norm
    return embed_data


def get_component_data(X, sig, idx):
    """Get data for a component of the product space"""
    start_idx = sum([space[0] + 1 for space in sig[:idx]])
    end_idx = sum([space[0] + 1 for space in sig[:idx+1]])
    return X[:, start_idx:end_idx]


def plot_avg_scores(signatures, avg_psdt_scores, avg_dt_scores):
    """Plot average scores for different signatures"""
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
    """Plot boxplots for scores by signature"""
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

