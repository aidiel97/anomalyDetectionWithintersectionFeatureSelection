from scipy.stats import kendalltau
from sklearn.feature_selection import SelectKBest, f_classif

def pearsonCorrelation(X_train, y_train, k_value):
    ctx = "| Pearson Correlation Feature Selection |"
    # pearson corr (FS)
    # Pilih 20 (dibuat flexible) fitur terbaik berdasarkan Pearson
    selector = SelectKBest(score_func=f_classif, k=k_value)
    X_train_new = selector.fit_transform(X_train, y_train)

    # Fitur terpilih
    pearson_feature_indices = selector.get_support(indices=True)
    pearson_features = X_train.columns[pearson_feature_indices]

    print(f"\n{ctx}\n Fitur terpilih:", pearson_features.tolist())
    return pearson_features.tolist()

def kendallCorrelation(pd, X_train, y_train, k_value):
    ctx = "| Kendall Correlation Feature Selection |"
    kendall_corr = {col: kendalltau(X_train[col], y_train)[0] for col in X_train.columns}
    kendall_corr_df = pd.DataFrame(kendall_corr.items(), columns=["Feature", "Kendall Tau"])

    # Urutkan berdasarkan nilai absolut tertinggi
    kendall_corr_df["abs_tau"] = kendall_corr_df["Kendall Tau"].abs()
    kendall_corr_df = kendall_corr_df.sort_values(by="abs_tau", ascending=False)
    kendall_features = kendall_corr_df["Feature"].head(k_value).tolist()

    print(f"\n{ctx}\n Fitur terpilih:", kendall_features)
    return kendall_features

def intersection(pearson_features, kendall_features):
    ctx = "| Intersection Correlation Feature Selection |"
    intersection = list(set(pearson_features) & set(kendall_features))
    print(f"\n{ctx}\n Fitur yang beririsan:",intersection)

    return intersection