import streamlit as st 
import pandas as pd
import plotly.express as px
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Funções de pré-processamento
def fill_missing_values(df):
    numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    df['Cabin'] = df['Cabin'].fillna('Unknown')
    df['Name'] = df['Name'].fillna('Unknown')
    return df

def engineer_features(df):
    df['Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0] if x != 'Unknown' else 'Unknown')
    df['Side'] = df['Cabin'].apply(lambda x: x.split('/')[-1] if x != 'Unknown' else 'Unknown')
    df['TotalSpend'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)
    return df

# Função para prever passageiro
def predict_passenger(model, passenger_data, feature_columns, le):
    passenger_df = pd.DataFrame([passenger_data])
    passenger_df = fill_missing_values(passenger_df)
    passenger_df = engineer_features(passenger_df)
    passenger_df = pd.get_dummies(passenger_df, columns=['HomePlanet', 'Destination', 'Deck', 'Side'], dummy_na=False)
    passenger_df, _ = passenger_df.align(feature_columns, join='right', axis=1, fill_value=0)
    for col in ['CryoSleep', 'VIP']:
        passenger_df[col] = le.fit_transform(passenger_df[col])
    passenger_df = passenger_df.drop(columns=['PassengerId', 'Cabin', 'Name', 'Group'], errors='ignore')
    
    prediction = model.predict(passenger_df)
    probability = model.predict_proba(passenger_df)[0]
    return prediction[0], probability


# Carregar modelo e dados
@st.cache_data
def load_model():
    return joblib.load('spaceship_titanic_model_v2.pkl')
model = load_model()
#model = joblib.load('spaceship_titanic_model_v2.pkl')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# Pré-processar dados
train = fill_missing_values(train)
test = fill_missing_values(test)
train = engineer_features(train)
test = engineer_features(test)

# Codificação
categorical_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']
train = pd.get_dummies(train, columns=categorical_cols, dummy_na=False)
test = pd.get_dummies(test, columns=categorical_cols, dummy_na=False)
train, test = train.align(test, join='left', axis=1, fill_value=0)
le = LabelEncoder()
for col in ['CryoSleep', 'VIP']:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# Remover colunas desnecessárias
drop_cols = ['PassengerId', 'Cabin', 'Name', 'Group']
train = train.drop(columns=drop_cols)
test_ids = test['PassengerId']
test = test.drop(columns=drop_cols)

# Dividir dados
X = train.drop('Transported', axis=1)
y = train['Transported']
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Previsões para avaliação
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)
class_report = classification_report(y_val, y_pred, output_dict=True)

# Streamlit App
st.title("Spaceship Titanic: Previsão de Passageiros Transportados")
st.write("""
Bem-vindo à aplicação interativa do projeto Spaceship Titanic!
""")

# Seção: Sobre o Projeto
st.header("Sobre o Projeto")
st.write("""
- **Objetivo**: Prever se um passageiro foi transportado (True) ou não (False) com base em características como idade, gastos, sono criogênico, etc.
- **Dataset**: Dados de ~13.000 passageiros (train.csv e test.csv) do Kaggle.
- **Modelos**: XGBoost.
- **Features**: Inclui Age, CryoSleep, TotalSpend, GroupSize, IsAlone, e mais.
- **Pré-processamento**: Tratamento de valores ausentes, codificação categórica, engenharia de features (ex.: GroupSize, IsAlone).
""")

# Seção: Resultados do Modelo
st.header("Resultados do Modelo")
st.write(f"**Acurácia no conjunto de validação**: {accuracy:.4f}")
st.write("**Matriz de Confusão**:")
# Criar e exibir a matriz de confusão como heatmap interativo
fig = px.imshow(
    conf_matrix,
    text_auto=True,
    labels=dict(x="Previsão", y="Real", color="Contagem"),
    x=["Não Transportado", "Transportado"],
    y=["Não Transportado", "Transportado"],
    color_continuous_scale='Blues'
)
fig.update_layout(
    title="Matriz de Confusão",
    xaxis_title="Previsão",
    yaxis_title="Real",
    height=400,
    width=500
)
st.plotly_chart(fig)

# Explicação detalhada da matriz
st.subheader("Interpretação da Matriz de Confusão")
st.write("""
- **Verdadeiro Negativo (TN)**: Número de passageiros corretamente previstos como 'Não Transportado'. Valor: {tn}.
- **Falso Positivo (FP)**: Número de passageiros incorretamente previstos como 'Transportado' quando na verdade não foram. Valor: {fp}.
- **Falso Negativo (FN)**: Número de passageiros incorretamente previstos como 'Não Transportado' quando na verdade foram. Valor: {fn}.
- **Verdadeiro Positivo (TP)**: Número de passageiros corretamente previstos como 'Transportado'. Valor: {tp}.
- **Implicações**: Uma alta taxa de Falsos Negativos pode indicar que o modelo está perdendo passageiros transportados, enquanto Falsos Positivos sugerem previsões excessivamente otimistas.
""".format(
    tn=conf_matrix[0, 0],
    fp=conf_matrix[0, 1],
    fn=conf_matrix[1, 0],
    tp=conf_matrix[1, 1]
))

st.write("**Relatório de Classificação**:")
st.write(pd.DataFrame(class_report).transpose())

# Validação cruzada
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
st.write(f"**Acurácia média na validação cruzada**: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

# Seção: Previsão Interativa
st.header("Prever um Passageiro")
st.write("Insira os dados do passageiro abaixo para prever se ele foi transportado.")

with st.form("passenger_form"):
    passenger_id = st.text_input("PassengerId", "9999_01")
    home_planet = st.selectbox("HomePlanet", ["Earth", "Europa", "Mars"])
    cryo_sleep = st.checkbox("CryoSleep")
    cabin = st.text_input("Cabin", "G/1500/S")
    destination = st.selectbox("Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])
    age = st.slider("Age", 0, 100, 30)
    vip = st.checkbox("VIP")
    room_service = st.number_input("RoomService", 0.0, 10000.0, 0.0)
    food_court = st.number_input("FoodCourt", 0.0, 10000.0, 0.0)
    shopping_mall = st.number_input("ShoppingMall", 0.0, 10000.0, 0.0)
    spa = st.number_input("Spa", 0.0, 10000.0, 0.0)
    vr_deck = st.number_input("VRDeck", 0.0, 10000.0, 0.0)
    name = st.text_input("Name", "John Doe")
    submitted = st.form_submit_button("Prever")

    if submitted:
        passenger_data = {
            'PassengerId': passenger_id,
            'HomePlanet': home_planet,
            'CryoSleep': cryo_sleep,
            'Cabin': cabin,
            'Destination': destination,
            'Age': age,
            'VIP': vip,
            'RoomService': room_service,
            'FoodCourt': food_court,
            'ShoppingMall': shopping_mall,
            'Spa': spa,
            'VRDeck': vr_deck,
            'Name': name
        }
        prediction, probability = predict_passenger(model, passenger_data, X_train, le)
        st.write(f"**Previsão**: {'Transportado' if prediction else 'Não Transportado'}")
        st.write(f"**Probabilidades [Não Transportado, Transportado]**: {probability}")

        st.header("Análise de Erros")
errors = X_val[y_pred != y_val].copy()
errors['True'] = y_val[y_pred != y_val]
errors['Predicted'] = y_pred[y_pred != y_val]
st.write("Exemplos de previsões erradas:")
st.write(errors.head())
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=errors, x='Age', hue='True', bins=30, ax=ax)
plt.title('Distribuição de Idade nos Erros')
st.pyplot(fig)

# Seção: Conclusão
st.header("Conclusão")
st.write("""
O modelo XGBoost alcançou uma acurácia sólida (~0.78-0.80) e generaliza bem, conforme validado por validação cruzada.
Features como CryoSleep, TotalSpend, GroupSize e IsAlone são altamente influentes.
Esta aplicação permite explorar os resultados e testar previsões interativamente.
""")
