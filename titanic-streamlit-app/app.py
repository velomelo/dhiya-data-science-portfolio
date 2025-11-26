
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def load_model():
    df = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    for d in [df, test]:
        d['FamilySize'] = d['SibSp'] + d['Parch'] + 1
        d['IsAlone'] = (d['FamilySize'] == 1).astype(int)
        d['Title'] = d['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        d['Title'] = d['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
        d['Title'] = d['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
        d['Age'].fillna(28, inplace=True)
        d['Fare'].fillna(14, inplace=True)
        d['Embarked'].fillna('S', inplace=True)
    features = ['Pclass','Sex','Age','Fare','Embarked','FamilySize','IsAlone','Title']
    X = pd.get_dummies(df[features])
    X_test = pd.get_dummies(test[features])
    X_test = X_test.reindex(columns=X.columns, fill_value=0)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, df['Survived'])
    return model, X.columns

model, cols = load_model()

st.title("Titanic Survival Predictor")
st.write("Isi data → klik Predict")

c1, c2 = st.columns(2)
with c1:
    pclass = st.selectbox("Kelas", [1,2,3])
    sex = st.selectbox("Sex", ["male","female"])
    age = st.slider("Umur", 0, 80, 30)
    fare = st.number_input("Fare", 0, 512, 32)
with c2:
    emb = st.selectbox("Embarked", ["S","C","Q"])
    sib = st.number_input("Saudara/Pasangan", 0, 8, 0)
    parch = st.number_input("Ortu/Anak", 0, 6, 0)

if st.button("Predict"):
    fam = sib + parch + 1
    alone = 1 if fam == 1 else 0
    title = "Mr" if sex == "male" else "Miss"
    inp = pd.DataFrame({'Pclass':[pclass],'Sex':[sex],'Age':[age],'Fare':[fare],
                        'Embarked':[emb],'FamilySize':[fam],'IsAlone':[alone],'Title':[title]})
    X_inp = pd.get_dummies(inp).reindex(columns=cols, fill_value=0)
    pred = model.predict(X_inp)[0]
    prob = model.predict_proba(X_inp)[0]
    if pred==1:
        st.success(f"SELAMAT! (Kemungkinan hidup: {prob[1]:.1%})")
        st.balloons()
    else:
        st.error(f"GUGUR (Kemungkinan mati: {prob[0]:.1%})")

model, cols = load_model()

st.title("Titanic Survival Predictor")
st.write("Isi data → klik Predict")

c1, c2 = st.columns(2)
with c1:
    pclass = st.selectbox("Kelas", [1,2,3])
    sex = st.selectbox("Sex", ["male","female"])
    age = st.slider("Umur", 0, 80, 30)
    fare = st.number_input("Fare", 0, 512, 32)
with c2:
    emb = st.selectbox("Embarked", ["S","C","Q"])
    sib = st.number_input("Saudara/Pasangan", 0, 8, 0)
    parch = st.number_input("Ortu/Anak", 0, 6, 0)

if st.button("Predict"):
    fam = sib + parch + 1
    alone = 1 if fam == 1 else 0
    title = "Mr" if sex == "male" else "Miss"
    inp = pd.DataFrame({'Pclass':[pclass],'Sex':[sex],'Age':[age],'Fare':[fare],
                        'Embarked':[emb],'FamilySize':[fam],'IsAlone':[alone],'Title':[title]})
    X_inp = pd.get_dummies(inp).reindex(columns=cols, fill_value=0)
    pred = model.predict(X_inp)[0]
    prob = model.predict_proba(X_inp)[0]
    if pred==1:
        st.success(f"SELAMAT! (Kemungkinan hidup: {prob[1]:.1%})")
        st.balloons()
    else:
        st.error(f"GUGUR (Kemungkinan mati: {prob[0]:.1%})")
