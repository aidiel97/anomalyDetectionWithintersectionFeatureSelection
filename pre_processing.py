from sklearn.preprocessing import LabelEncoder

categorical_cols = ["proto", "service", "state"]

def cleansing(train, test):
    ctx = "| Data Cleansing |"
    # data cleansing -> tidak perlu karena tidak ada data null
    # print(train.isnull().sum()) # tidak ada null
    # print(test.isnull().sum()) # tidak ada null

    print(f"\n{ctx} Tidak perlu dilakukan karena tidak ada data null")
    return train, test

def normalization(train, test):
    ctx = "| Data Normalization |"
    label_encoders = {} # Dictionary untuk menyimpan encoder setiap kolom

    for col in categorical_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col])  # Encode train data
        label_encoders[col] = le  # Simpan encoder untuk digunakan di test

    for col in categorical_cols:
        test[col] = test[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else 0)

    train.drop(columns=["attack_cat"], inplace=True)
    test.drop(columns=["attack_cat"], inplace=True)
    train.drop(columns=["id"], inplace=True)
    test.drop(columns=["id"], inplace=True)

    X_train = train.drop(columns=["label"])
    y_train = train["label"]

    X_test = test.drop(columns=["label"])
    y_test = test["label"]

    
    print(f"\n{ctx} Done!")
    return X_train, X_test, y_train, y_test