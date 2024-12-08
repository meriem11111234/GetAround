import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Config
st.set_page_config(
    page_title="Analyse des Retards Getaround 🚗",
    page_icon="⌛ ⏱️ 🚗 🚙",
    layout="centered"
)

# App
st.title('Analyse des Retards Getaround - Tableau de Bord ⏱️ 🚗')

st.markdown("""
    :wave: Bonjour et bienvenue sur ce tableau de bord !
    
    Lorsqu'ils utilisent Getaround, les conducteurs réservent des voitures pour une période spécifique : de quelques heures à plusieurs jours.
    Les utilisateurs doivent ramener la voiture à l'heure prévue.
    Il arrive de temps en temps que les conducteurs soient en retard pour le check-out.
    Les retours tardifs peuvent poser des problèmes pour le prochain conducteur, surtout si la voiture est réservée le même jour.
    Cela entraîne des retours négatifs de la part des clients qui doivent attendre le retour de la voiture. Certains annulent même leur location.
    
    :dart: L'objectif de ce tableau de bord est de fournir des indications sur l'impact de l'introduction d'un seuil de temps pour les locations.
    En fonction du seuil de temps, une voiture ne sera pas affichée dans les résultats de recherche si les heures de check-in ou check-out demandées sont trop proches.
    
    🧐 Dans ce tableau de bord, vous pouvez explorer l'historique des données d'utilisation de Getaround. L'objectif est de donner des indications sur le compromis entre le retard minimum et son impact sur l'utilisation et les revenus.
    En examinant les données historiques ci-dessous, cela peut lancer les discussions sur les questions suivantes :
    
    - À quelle fréquence les conducteurs sont-ils en retard pour le prochain check-in ? Comment cela impacte-t-il le prochain conducteur ?
    - Quelle part des revenus de nos propriétaires serait potentiellement affectée par cette fonctionnalité ?
    - Combien de locations seraient impactées par cette fonctionnalité en fonction du seuil et de la portée que nous choisissons ?
    - Combien de cas problématiques seront résolus en fonction du seuil et de la portée choisis ?
    
    🚀 Commençons !
""")
st.markdown("---")


@st.cache_data
def load_data():
    fname = 'get_around_delay_analysis_clean.csv'
    data = pd.read_csv(fname)
    return data


st.text('Chargement des données...')
dataset2 = load_data()

st.markdown("---")
st.subheader('Aperçu du jeu de données')
# Exécution si la case est cochée
if st.checkbox('Afficher les données traitées'):
    st.subheader('Aperçu de 10 lignes aléatoires')
    st.write(dataset2.sample(10))
st.markdown("""
    Dans ce jeu de données, certains outliers possibles ont été évités.
    Le jeu de données nettoyé contient 20 980 lignes. 330 lignes ont été suspectées d'être des outliers et ont été supprimées.
    De plus, des versions factorisées des caractéristiques non numériques ont été ajoutées pour diverses raisons.
""")
st.markdown("---")

# Graphique 1
fig1 = px.histogram(dataset2, x='delay')
st.plotly_chart(fig1, use_container_width=True)
st.subheader('Graphique 1 - Distribution des locations à l\'heure ou en retard')
st.markdown("""
    Nous pouvons voir que le retard au check-out est très courant parmi les conducteurs de Getaround.
    Cela peut aller de quelques minutes à plus d'une heure.
""")
st.markdown("---")
st.markdown("""
**Statut des locations :**

Le graphique montre deux états de location : "ended" (terminée) et "canceled" (annulée).  
La majorité des locations ont été terminées, tandis qu'une partie relativement plus petite a été annulée.

**Répartition des retards :**

Les barres empilées montrent les différentes catégories de retard :  
- **0 (No delay)** : Aucune minute de retard.  
- **1 (Delay < 10 mins)** : Retard de moins de 10 minutes.  
- **2 (10 ≤ Delay < 60 mins)** : Retard entre 10 et 60 minutes.  
- **3 (Delay ≥ 60)** : Retard de 60 minutes ou plus.  
- **4 (Not applicable)** : Cas où la catégorie de retard n'est pas applicable.

**Retards dans les locations terminées :**

Parmi les locations qui se sont terminées, une proportion significative n'a eu aucun retard (barre bleue).  
Cependant, il y a également un nombre notable de locations avec des retards de plus de 10 minutes (barres rouge et rose), indiquant que les retards sont relativement fréquents.

**Annulations :**

Pour les locations annulées, toutes les entrées sont marquées comme "Not applicable" (vert), ce qui est logique puisque si une location est annulée, il n'y a pas de check-out et donc pas de retard.

**Impact potentiel :**

Le graphique met en évidence que les retards ne sont pas rares et peuvent être suffisamment longs pour avoir un impact potentiel sur les locations suivantes.  
L'annulation de locations, bien que marquée comme non applicable pour les retards, pourrait être en partie due à des inquiétudes concernant des retards ou des perturbations prévues.
""")

# Graphique 2
fig2 = px.histogram(dataset2, x='state', color='delay')
st.plotly_chart(fig2, use_container_width=True)
st.subheader('Graphique 2 - Distribution des locations à l\'heure ou en retard selon leur statut')
st.markdown("""
    Environ 3 200 utilisateurs de Getaround ont annulé leur trajet, possiblement à cause du retard au check-out.
""")

st.markdown("")
st.markdown("---")

# Graphique 3
fig3 = px.histogram(dataset2, x='state', color='delay',
                    facet_col='checkin_type')
st.plotly_chart(fig3, use_container_width=True)
st.subheader('Graphique 3 - Distribution des locations à l\'heure ou en retard selon leur statut et type de check-in')
st.markdown("""
    Veuillez consulter la figure suivante pour les commentaires.
""")
st.markdown("---")

# Graphique 4
fig4 = px.histogram(dataset2, x='delay', color='state',
                    facet_col='checkin_type')
st.plotly_chart(fig4, use_container_width=True)
st.subheader('Graphique 4 - Interprétation améliorée de la figure précédente')
st.markdown("""
    Le type de check-in "connect" est beaucoup moins utilisé par les conducteurs que la méthode traditionnelle par mobile.
    Les conducteurs qui ont fait leur check-in avec la fonctionnalité "connect" ont eu proportionnellement beaucoup moins de retard que ceux qui ont utilisé la méthode sans "connect".

    Il semble que le type de check-in "connect"
    * réduit les retards au check-out et
    * est susceptible de réduire les frictions parmi les conducteurs de Getaround.
""")
st.markdown("---")

# Graphique 5
fig5 = px.box(
    dataset2,
    x='state',
    y='time_delta_with_previous_rental_in_minutes',
    facet_col='checkin_type')
fig5.update_layout(yaxis_title="Temps delta entre deux locations en minutes")
st.plotly_chart(fig5, use_container_width=True)
st.subheader('Graphique 5 - Distribution du temps delta entre deux locations en minutes')
st.markdown("""
    Le temps delta entre deux locations en minutes ne semble pas avoir d'impact évident.
""")
st.markdown("---")

# Graphique 6
st.markdown("Correlation between features")
corr_match = dataset2.corr().loc[:, ['factorized_delay']].abs(
).sort_values(by='factorized_delay', ascending=False)
key_cols = corr_match.index[:-3]  # sorted column names
df_corr = dataset2[key_cols].corr().abs()
fig6 = go.Figure()
fig6.add_trace(
    go.Heatmap(
        x=df_corr.columns,
        y=df_corr.index,
        z=np.array(df_corr)
    )
)
st.plotly_chart(fig6, use_container_width=True)
st.subheader('Graphique 6 - Corrélation entre les caractéristiques')
st.markdown("""
**Influence du type de check-in** :  
Le type de check-in influence modérément l'état de la location et la classification du retard, mais son impact sur le retard en minutes est limité.

**Autres facteurs** :  
Les corrélations plus faibles suggèrent que d'autres facteurs non représentés ici pourraient jouer un rôle plus important dans les retards et les annulations de locations.
""")

st.markdown("---")


# Conclusion

st.subheader('Conclusions')
# quelques calculs ci-dessous
tot_data = dataset2.shape[0]  # nombre total de cas
mask_0 = dataset2.state == "canceled"
tot_cancel = mask_0.sum()  # nombre total de cas annulés
tot_cancel_percent = round(100. * (tot_cancel / tot_data), 1)  # pourcentage
tot_ended = tot_data - tot_cancel  # nombre total de cas terminés
col_ = "delay_at_checkout_in_minutes"
mask_a = dataset2[col_] >= 0.0
tot_delay_today = mask_a.sum()  # nombre total de cas retardés
tot_delay_today_percent = round(
    100. * (tot_delay_today / tot_ended), 1)  # pourcentage

st.write("1. ", tot_delay_today, " (", tot_delay_today_percent,
         " pourcent) des conducteurs étaient en retard pour le prochain check-in.")
st.write("2. Cela a peut-être conduit ", tot_cancel, " (", tot_cancel_percent,
         " pourcent) d'utilisateurs à annuler leur demande de location.")

st.markdown("""
    Les retards augmentent les chances d'annulation d'une location, ce qui peut entraîner une perte de revenus pour l'entreprise. Cela augmente le risque financier.
    Il est fortement recommandé d'optimiser ce risque financier en introduisant un seuil sur les retards.
""")
input_threshold = st.slider(
    '3. Déplacez le curseur du seuil de retard pour voir son impact',
    0,
    150,
    step=5)

try:  # au cas où l'utilisateur a interagi avec le curseur
    mask_b = dataset2[col_] >= input_threshold
except BaseException:  # au cas où l'utilisateur n'a pas encore interagi avec le curseur
    mask_b = mask_a
tot_delay_tomorrow = mask_b.sum()  # nombre de retards après l'introduction du seuil
# nombre de problèmes de retard résolus
change_delay = tot_delay_today - tot_delay_tomorrow
change_delay_percent = round(100. * change_delay / tot_ended, 1)

st.write("\n\t :star: Si le seuil de retard ci-dessus avait été introduit, ",
         change_delay, " (", change_delay_percent,
         " pourcent) des cas problématiques auraient été résolus.\n")

type_checkin = st.selectbox(
    '4. Sélectionnez la fonctionnalité du type de check-in', [
        "Seulement connect", "Aucun changement"])
mask_c = dataset2["checkin_type"] == "connect"
total_delay_connect = (mask_a & mask_c).sum()
total_delay_connect_percent = round(
    100. * (total_delay_connect / tot_delay_today), 1)
tot_delay_tomorrow_percent = tot_delay_today_percent - total_delay_connect_percent
if type_checkin == "Seulement connect":
    st.write(
        "\t:star:Avec l'utilisation du type de check-in ci-dessus, ",
        tot_delay_tomorrow_percent,
        " pourcentage des cas problématiques auraient été résolus. \n")
else:
    st.write("\t:star:Désolé, l'utilisation de la fonctionnalité ci-dessus ne résout aucun problème.")
st.markdown("---")


# Graphique 7
threshold = np.arange(0, 60 * 24, 2.5)
rent_permin = 119. / 24. / 60.  # prix médian de la location $ par minute
rental_duration = 4 * 60.  # estimation de 4 heures pour toutes les locations annulées
col_ = 'delay_at_checkout_in_minutes'
risk_percentage = []
for val_delay in threshold:
    mask_0 = dataset2[col_] > val_delay
    tot_late_mins = dataset2.loc[mask_0, col_].sum()
    earn_late = rent_permin * tot_late_mins
    count_late = mask_0.sum()
    risk_late = count_late * rent_permin * rental_duration
    risk_percentage.append(risk_late / earn_late * 100.)  # en pourcentage

fig7 = px.line(x=threshold, y=risk_percentage)
fig7.add_hline(y=100., line_color="red")
fig7.update_layout(
    xaxis_title='Seuil en minutes',
    yaxis_title='Métrique utilisée pour quantifier le risque de perte d\'argent')
st.plotly_chart(fig7, use_container_width=True)
st.subheader('Graphique 7 - Risque financier en fonction du seuil de retard')
st.markdown("""
    Les retards au check-out causent des frictions parmi les conducteurs et posent un risque financier.
    À cette fin, une métrique est définie pour quantifier le risque de perte d'argent comme calculé ci-dessous
    1. Calculer le montant d'argent risqué en raison des retards pour chaque seuil de retard
    2. Calculer le montant d'argent gagné par les minutes supplémentaires dues aux retards.
    3. Calculer la métrique = montant d'argent risqué / argent gagné par les retards
    Sans aucun seuil, l'entreprise prend le risque de perdre beaucoup d'argent.
    Dans la situation actuelle, la métrique calculée est de 193%.
    Cela signifie que l'entreprise pourrait avoir perdu deux fois plus d'argent qu'elle n'en a gagné grâce aux retours tardifs des utilisateurs.
    C'est un risque important que l'entreprise a pris.
    J'assume le pire des scénarios où tous les retards de check-out entraînent l'annulation d'une nouvelle location de 4 heures.
    Il est nécessaire de trouver un niveau de seuil optimal pour réduire le risque financier.
    Dans le pire des scénarios, **à 60 minutes**, l'argent gagné par les retards est égal à l'argent risqué en raison des retards au check-out.
    *Il serait très utile de faire un test A/B pour optimiser ce seuil avant de l'appliquer à l'ensemble du réseau.*
    Pour les calculs, j'assume
    * un prix de location par jour de 119 dollars (obtenu à partir du jeu de données Getaround)
""")
st.markdown("---")

# Pied de page
st.write("")
