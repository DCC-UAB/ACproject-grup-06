import pandas as pd
from jinja2 import Template
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Carregar dataset
fitxer = 'netejat.csv'
df = pd.read_csv(fitxer, encoding='latin-1')

# Estadístiques
total_tweets = len(df)
count_0 = df[df['Ranking'] == 0].shape[0]
count_4 = df[df['Ranking'] == 4].shape[0]
mean_length = df['Tweet'].apply(len).mean()
min_length = df['Tweet'].apply(len).min()
max_length = df['Tweet'].apply(len).max()
std_length = df['Tweet'].apply(len).std()
shortest_tweet = df.loc[df['Tweet'].apply(len).idxmin()]['Tweet']
longest_tweet = df.loc[df['Tweet'].apply(len).idxmax()]['Tweet']

# Gràfiques
plt.figure(figsize=(10, 6))
df['Tweet_Length'] = df['Tweet'].apply(len)
plt.hist(df['Tweet_Length'], bins=50, color='blue', alpha=0.7)
plt.title('Distribució de la llargada dels Tweets')
plt.xlabel('Llargada del Tweet')
plt.ylabel('Nombre de Tweets')
plt.savefig('llargada_tweets.png')

plt.figure(figsize=(8, 6))
df['Ranking'].value_counts().plot(kind='bar', color=['blue', 'orange'])
plt.title('Distribució de Sentiments (0 i 4)')
plt.xlabel('Sentiment')
plt.ylabel('Nombre de Tweets')
plt.savefig('sentiments.png')

# Nuvol de paraules
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Tweet']))
wordcloud.to_file('nuvol_paraules.png')

# Plantilla HTML
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Informe del Dataset</title>
</head>
<body>
    <h1>Informe del Dataset de Tweets</h1>
    <h2>1. Descripció General</h2>
    <p><strong>Total de registres:</strong> {{ total_tweets }}</p>
    <p><strong>Distribució de Sentiments:</strong></p>
    <ul>
        <li>Sentiment 0: {{ count_0 }}</li>
        <li>Sentiment 4: {{ count_4 }}</li>
    </ul>

    <h2>2. Estadístiques dels Tweets</h2>
    <ul>
        <li><strong>Llargada mitjana:</strong> {{ mean_length }}</li>
        <li><strong>Llargada mínima:</strong> {{ min_length }}</li>
        <li><strong>Llargada màxima:</strong> {{ max_length }}</li>
        <li><strong>Desviació típica:</strong> {{ std_length }}</li>
        <li><strong>Tweet més curt:</strong> {{ shortest_tweet }}</li>
        <li><strong>Tweet més llarg:</strong> {{ longest_tweet }}</li>
    </ul>

    <h2>3. Gràfiques</h2>
    <img src="llargada_tweets.png" alt="Distribució de la llargada dels tweets" width="600">
    <img src="sentiments.png" alt="Distribució de sentiments" width="600">
    <h2>4. Nuvol de Paraules</h2>
    <img src="nuvol_paraules.png" alt="Nuvol de paraules" width="600">
</body>
</html>
"""

# Renderitzar HTML
template = Template(html_template)
html_content = template.render(
    total_tweets=total_tweets,
    count_0=count_0,
    count_4=count_4,
    mean_length=round(mean_length, 2),
    min_length=min_length,
    max_length=max_length,
    std_length=round(std_length, 2),
    shortest_tweet=shortest_tweet,
    longest_tweet=longest_tweet
)

# Guardar informe HTML
with open('informe_dataset.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Informe generat: informe_dataset.html")
