from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
# Accendiamo il macchinario per il Machine Learning (Feature Engineering)
macchinario_encoder = None
if OneHotEncoder is not None:
    macchinario_encoder = OneHotEncoder(sparse_output=False)

# ==========================================
# CREAZIONE DATASET DI ESERCITAZIONE
# (Simuliamo un file del supermercato!)
# ==========================================

dati_supermercato = {
    'Prodotto': ['  birra messina ', 'pepsi zero', 'cracker', 'banane ', 'acqua'],
    'Reparto': ['Bevande', 'Bevande', 'Snack', 'Ortofrutta', 'Bevande'], # Nuova colonna aggiunta per l'esercizio ML!
    'Prezzo': [1.50, 1.20, 2.00, np.nan, 0.50],
    'Spese_Spedizione': [0.20, 0.10, 0.30, 0.50, 0.10],
    'Pezzi_Esposti': [10, 20, 5, np.nan, 50],
    'Incasso_Giornaliero': [45.0, 60.0, 10.0, 0.0, 800.0], # L'800 è il nostro outlier!
    'Ore_Straordinario': [2, np.nan, 1, 0, np.nan],
    'Data_Scadenza': ['2026-11-01', '2027-05-15', '2026-12-31', '2026-10-20', '2028-01-01']
}

df = pd.DataFrame(dati_supermercato)
print("--- DATASET ORIGINALE ---")
print(df)


# ==========================================
# 1. ESPLORAZIONE E GRAFICI (Il max e plt.axhline)
# ==========================================
print("\n--- 1. GRAFICO ---")
# Disegniamo un grafico a barre degli incassi e tracciamo una riga verde sul massimo
plt.bar(df['Prodotto'], df['Incasso_Giornaliero'])
plt.axhline(df['Incasso_Giornaliero'].max(), color="green", linestyle='-', label="Incasso Massimo")
plt.legend()
plt.show() # Chiudi la finestra del grafico per far proseguire lo script!


# ==========================================
# 2. OPERAZIONI VETTORIALI (+ e np.where)
# ==========================================
print("\n--- 2. OPERAZIONI VETTORIALI ---")
# Calcoliamo il costo totale
df["Costo_Totale"] = df["Prezzo"] + df["Spese_Spedizione"]

# Controlliamo se serve rifornire lo scaffale (meno di 15 pezzi)
df["Da_Rifornire"] = np.where(df["Pezzi_Esposti"] < 15, "SI", "NO")
print(df[['Prodotto', 'Pezzi_Esposti', 'Da_Rifornire']])


# ==========================================
# 3. VALORI MANCANTI (fillna)
# ==========================================
print("\n--- 3. VALORI MANCANTI ---")
# Chi non ha fatto straordinari ha uno 0
df["Ore_Straordinario"] = df["Ore_Straordinario"].fillna(0)
print(df[['Prodotto', 'Ore_Straordinario']])


# ==========================================
# 4. OUTLIER (Deviazione Standard e abs)
# ==========================================
print("\n--- 4. GESTIONE OUTLIER ---")
media_incasso = df["Incasso_Giornaliero"].mean()
dev_incasso = df["Incasso_Giornaliero"].std()

# Troviamo l'anomalia (l'incasso da 800€ sull'acqua che sballa tutto)
df["Anomalia"] = (abs(df["Incasso_Giornaliero"] - media_incasso) > 2 * dev_incasso)
print(df[['Prodotto', 'Incasso_Giornaliero', 'Anomalia']])


# ==========================================
# 5. STRINGHE E DATE E OTTIMIZZAZIONE (str, dt, category, downcast, set_index)
# ==========================================
print("\n--- 5. STRINGHE E DATE ---")
# Puliamo i nomi dei prodotti scritti male
df["Prodotto_Pulito"] = df["Prodotto"].str.strip().str.title()
print(df[['Prodotto', 'Prodotto_Pulito']])

# Convertiamo la colonna testuale in formato Data e calcoliamo i giorni
df["Data_Scadenza"] = pd.to_datetime(df["Data_Scadenza"])
df["Mese_Scadenza"] = df["Data_Scadenza"].dt.month
df["Giorni_alla_Scadenza"] = (df['Data_Scadenza'] - pd.Timestamp.today()).dt.days

# Ottimizzazioni di memoria
df['Prodotto_Pulito'] = df['Prodotto_Pulito'].astype("category")
df['Prezzo'] = pd.to_numeric(df['Prezzo'], downcast="float")

# Impostiamo il prodotto pulito come indice (lo mettiamo alla fine così non sballa le vecchie print)
df = df.set_index('Prodotto_Pulito')


# ==========================================
# 6. FEATURE ENGINEERING (One-Hot Encoding con Scikit-Learn)
# ==========================================
print("\n--- 6. MACHINE LEARNING: ONE-HOT ENCODING ---")

# Trasformiamo la colonna di testo "Reparto" in numeri 0 e 1 per l'Intelligenza Artificiale
reparti_numerici = macchinario_encoder.fit_transform(df[['Reparto']])

print("Ecco lo spezzatino in 0 e 1 per la colonna Reparto (Bevande, Ortofrutta, Snack):")
print(reparti_numerici)

# ==========================================
# 7. PREVISIONE REALE CON L'IA (Machine Learning)
# ==========================================
print("\n--- 7. PREVISIONE CON L'INTELLIGENZA ARTIFICIALE ---")

# L'IA odia i buchi (NaN). Diamo alle banane un numero finto di pezzi esposti (es. 20) per non farla crashare
df["Pezzi_Esposti"] = df["Pezzi_Esposti"].fillna(20)

# FASE 1: Prepariamo le variabili da studiare
# X = i dati su cui l'IA deve ragionare (le doppie quadre servono perché vuole una tabella)
X = df[["Pezzi_Esposti"]]
# y = l'obiettivo che l'IA deve imparare a indovinare
y = df["Da_Rifornire"]

# FASE 2: Accendiamo e addestriamo il cervello
cervello_ia = DecisionTreeClassifier()
cervello_ia.fit(X, y) # IL MOMENTO MAGICO: qui l'IA studia le tue vecchie decisioni e impara la regola!

# FASE 3: Facciamo una previsione sul futuro!
# Arriva un nuovo prodotto: le Gocciole. Abbiamo 8 pacchi sullo scaffale.
nuovo_prodotto = pd.DataFrame({"Pezzi_Esposti": [8]})

# Chiediamo all'IA di prendere una decisione sul nuovo prodotto
previsione = cervello_ia.predict(nuovo_prodotto)

print(f"Situazione: Abbiamo 8 pacchi di Gocciole sullo scaffale.")
print(f"Previsione dell'IA -> Devo rifornire? {previsione[0]}")