import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def clean_title(title):
    turkish_characters = {'ı': 'i', 'İ': 'I', 'ğ': 'g', 'Ğ': 'G', 'ü': 'u', 'Ü': 'U', 'ş': 's', 'Ş': 'S', 'ö': 'o',
                          'Ö': 'O', 'ç': 'c', 'Ç': 'C'}
    for tr, en in turkish_characters.items():
        title = title.replace(tr, en)
    return title.replace(' ', '_')

def extract_numbers(input_string):
    numbers = re.findall(r'\d+', input_string)
    return ''.join(numbers)

def get_real_estate_data(city, district=None):
    url = f'https://www.emlakjet.com/kiralik-konut/{city}-{district}' if district else f'https://www.emlakjet.com/kiralik-konut/{city}'

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error: Unable to fetch the page for {city}, {district}.")
        return None

    soup = BeautifulSoup(response.content, "html.parser")
    link_class = soup.find_all('a', class_='_3qUI9q')
    links = [div.get('href').strip() if div.get('href') else None for div in link_class]
    price_class = soup.find_all('p', class_='_2C5UCT')
    prices = [div.find('span').text.strip() if div.find('span') else None for div in price_class]

    real_estate_info_list = []
    for link in links:
        response = requests.get(f'https://www.emlakjet.com/{link}')
        if response.status_code != 200:
            print(f"Error: Unable to fetch the page.")
            return None

        soup = BeautifulSoup(response.content, "html.parser")
        info_class = soup.find_all('div', class_='_1bVOdb')

        real_estate_info = {}
        whitelist = ['Net_Metrekare', 'Bulundugu_Kat', 'Oda_Sayisi', 'Isitma_Tipi', 'Banyo_Sayisi',
                     'Esya_Durumu', 'Site_Icerisinde', 'Balkon_Durumu']

        for i in range(0, len(info_class), 2):
            title = clean_title(info_class[i].text.strip())
            if title in whitelist:
                value = info_class[i + 1].text.strip()
                if title == 'Net_Metrekare':
                    real_estate_info[title] = value.split(' ')[0]
                elif title == 'Bulundugu_Kat':
                    real_estate_info[title] = extract_numbers(value)
                elif title == 'Oda_Sayisi':
                    oda_sayisi_parts = re.findall(r'\d+', value)
                    oda_sayisi = sum(map(int, oda_sayisi_parts))
                    real_estate_info[title] = oda_sayisi
                elif title == 'Isitma_Tipi':
                    real_estate_info[title] = 0 if value == 'Klimalı' else 1
                elif title == 'Esya_Durumu':
                    real_estate_info[title] = 1 if value == 'Eşyalı' else 0
                elif title == 'Site_Icerisinde':
                    real_estate_info[title] = 1 if value == 'Evet' else 0
                elif title == 'Balkon_Durumu':
                    real_estate_info[title] = 1 if value == 'Var' else 0
                elif title == 'Banyo_Sayisi':
                    real_estate_info[title] = 0 if value == 'Yok' else int(value)
                else:
                    real_estate_info[title] = value

        real_estate_info_list.append(real_estate_info)

    for i in range(min(len(prices), len(real_estate_info_list))):
        real_estate_info_list[i]['Fiyat'] = extract_numbers(prices[i])

    return real_estate_info_list

def clean_user_input(user_input):
    oda_sayisi_parts = re.findall(r'\d+', user_input['Oda_Sayisi'])

    cleaned_input = {
        'Net_Metrekare': int(user_input['Net_Metrekare']),
        'Bulundugu_Kat': int(user_input['Bulundugu_Kat']),
        'Oda_Sayisi': int(sum(map(int, oda_sayisi_parts))),
        'Isitma_Tipi': 0 if user_input['Isitma_Tipi'] == 'Klimalı' else 1,
        'Banyo_Sayisi': 0 if user_input['Banyo_Sayisi'] == 'Yok' else 1,
        'Esya_Durumu': 1 if user_input['Esya_Durumu'] == 'Eşyalı' else 0,
        'Site_Icerisinde': 1 if user_input['Site_Icerisinde'] == 'Evet' else 0,
        'Balkon_Durumu': 1 if user_input['Balkon_Durumu'] == 'Var' else 0
    }
    return cleaned_input

user_input = {
    'Sehir': clean_title(input('Şehir (örn: antalya): ')),
    'Ilce': clean_title(input('Ilçe (örn: muratpasa): ')),
    'Net_Metrekare': input('Net Metrekare: '),
    'Bulundugu_Kat': input('Bulunduğu Kat: '),
    'Oda_Sayisi': input('Oda Sayısı: '),
    'Isitma_Tipi': input('Isıtma Tipi (Klimalı/Yok): '),
    'Banyo_Sayisi': input('Banyo Sayısı (örn. 1): '),
    'Esya_Durumu': input('Eşya Durumu (Eşyalı/Değil): '),
    'Site_Icerisinde': input('Site İçerisinde (Evet/Hayır): '),
    'Balkon_Durumu': input('Balkon Durumu (Var/Yok): ')
}

property_data = get_real_estate_data(user_input['Sehir'], user_input['Ilce'])
property_df = pd.DataFrame(property_data).dropna()

label_encoder = LabelEncoder()
property_df['Isitma_Tipi'] = label_encoder.fit_transform(property_df['Isitma_Tipi'])
property_df['Banyo_Sayisi'] = label_encoder.fit_transform(property_df['Banyo_Sayisi'])
property_df['Esya_Durumu'] = label_encoder.fit_transform(property_df['Esya_Durumu'])
property_df['Site_Icerisinde'] = label_encoder.fit_transform(property_df['Site_Icerisinde'])
property_df['Balkon_Durumu'] = label_encoder.fit_transform(property_df['Balkon_Durumu'])

X = property_df[['Bulundugu_Kat', 'Oda_Sayisi', 'Net_Metrekare', 'Isitma_Tipi', 'Banyo_Sayisi', 'Esya_Durumu', 'Site_Icerisinde', 'Balkon_Durumu']]
y = property_df['Fiyat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.apply(pd.to_numeric, errors='coerce').dropna()
y_train = y_train[X_train.index]

X_test = X_test.apply(pd.to_numeric, errors='coerce').dropna()
y_test = y_test[X_test.index]

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

feature_importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print('\n')
print(feature_importance_df)
print('\n')

cleaned_user_input = clean_user_input(user_input)
user_input_df = pd.DataFrame([cleaned_user_input])
user_input_df = user_input_df[X_train.columns]

user_input_df['Isitma_Tipi'] = label_encoder.transform([user_input_df['Isitma_Tipi'].values[0]])[0]
user_input_df['Banyo_Sayisi'] = label_encoder.transform([user_input_df['Banyo_Sayisi'].values[0]])[0]
user_input_df['Esya_Durumu'] = label_encoder.transform([user_input_df['Esya_Durumu'].values[0]])[0]
user_input_df['Site_Icerisinde'] = label_encoder.transform([user_input_df['Site_Icerisinde'].values[0]])[0]
user_input_df['Balkon_Durumu'] = label_encoder.transform([user_input_df['Balkon_Durumu'].values[0]])[0]

predicted_price = rf_model.predict(user_input_df)
print(f"Verilen Girdi için Tahmini Fiyat: {predicted_price[0]:,.2f} TL")
