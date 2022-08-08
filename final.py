
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import xgboost as xgb
import pickle   

st.title('House Price Prediction in King County, USA')
st.subheader('Information of House:')


# Input 1: Lot size
sqft_lot = st.number_input('1) Enter the size of house lot in sqft', 500, 1600000)

st.write('##')

# Input 2: Interior size
sqft_living = st.number_input('2) Enter the size of house interior spaces in sqft', 500,13000)

st.write('##')

# Input 3: Waterfront
waterfront = st.radio('3) Does the house have waterfront view?',['Yes','No'])
if waterfront == 'Yes':
    waterfront = 1
else:
    waterfront = 0

st.write('##')

# Input 4: House view 
view = st.slider('4) Select the rating of views from house', 1,5)
view = view-1

st.write('##')

# Input 5: Construction quality grade
grade = st.selectbox('5) Select the grade for construction quality',
                             range(1,14))

st.write('##')

# Input 6: Age of house when sold
age_sold = st.number_input('6) Enter the age of house', 0, 120)

st.write('##')

# Input 7: Cities
city_selected = st.selectbox('7) Select a city in King County', ['Algona','Auburn','Bellevue','Duvall',
                               'Eastgate','Issaquah','Kenmore','Kent','Lake Joy',
                               'Medina','Mercer Island','Midway','Morganville',
                               'North Bend','Queensgate','Redmond','Renton',
                               'Sammamish','Seattle','Shoreline','Snoqualmie',
                               'Spring Glen','Totem Lake','Tukwila','Vashon',
                               'Wabash','White Center','Wilderness Village',
                               'Woodinville','Yarrow Point'])

st.write('##')

# Values of longitudinal / latitudinal, temperature, precipitation & crime rates based on cities:
zipcode_dict= {'Algona': [47.309019889502764, -122.27064640883978, 76.0, 35.0, 51.0, 7.0],
                'Auburn': [47.30835357411588, -122.27841760722347, 77.0, 35.0, 48.0, 21.4],
                'Bellevue': [47.61264855875831, -122.13499334811529, 76.0, 37.0, 47.0, 9.6],
                'Duvall': [47.73709578947368, -121.95483157894738, 76.0, 35.0, 63.0, 7.2],
                'Eastgate': [47.6105219858156, -122.14207801418439, 76.0, 37.0, 47.0, 9.6],
                'Issaquah': [47.546261332250204, -122.0752193338749, 76.0, 35.0, 59.0, 7.8],
                'Kenmore': [47.755137809187275, -122.24602826855123, 76.0, 36.0, 45.0, 9.4],
                'Kent': [47.37615278293136, -122.1520064935065, 77.0, 36.0, 49.0, 18.1],
                'Lake Joy': [47.67148790322581, -121.84853225806452, 76.0, 35.0, 53.0, 8.1],
                'Medina': [47.62584, -122.23353999999999, 76.0, 37.0, 42.0, 6.1],
                'Mercer Island': [47.55984609929078, -122.22559219858155, 76.0, 37.0, 44.0, 6.2],
                'Midway': [47.373555999999994, -122.278848, 77.0, 36.0, 49.0, 18.1],
                'Morganville': [47.33321, -121.99947, 76.0, 35.0, 55.0, 8.6],
                'North Bend': [47.4733592760181, -121.75903619909502, 75.0, 34.0, 76.0, 9.5],
                'Queensgate': [47.75519230769231, -122.20117435897436, 76.0, 36.0, 45.0, 11.6],
                'Redmond': [47.68027425944842, -122.07937282941778, 76.0, 36.0, 45.0, 10.6],
                'Renton': [47.47802905447715, -122.16287852222918, 76.0, 36.0, 50.0, 19.9],
                'Sammamish': [47.607663, -122.03479499999999, 75.0, 35.0, 59.0, 7.9],
                'Seattle': [47.613313368220744, -122.33610688233428, 76.0, 37.0, 41.3, 32.3],
                'Shoreline': [47.74424133891213, -122.33422092050209, 74.0, 37.0, 42.0, 10.7],
                'Snoqualmie': [47.53173, -121.86192903225805, 76.0, 35.0, 60.0, 7.8],
                'Spring Glen': [47.558418518518515, -121.90525925925927, 76.0, 35.0, 60.0, 7.8],
                'Totem Lake': [47.70273592630501, -122.19865711361311, 76.0, 36.0, 45.0, 7.6],
                'Tukwila': [47.49893235638921, -122.28466354044548, 76.0, 37.0, 45.0, 41.4],
                'Vashon': [47.41742288135593, -122.4638220338983, 76.0, 37.0, 44.0, 11.6],
                'Wabash': [47.21145042735043, -121.99516666666666, 75.0, 35.0, 71.0, 8.6],
                'White Center': [47.520413643659715, -122.35750080256821, 76.0, 37.0, 44.0, 18.5],
                'Wilderness Village': [47.37060406779661, -122.03178983050846, 76.0, 35.0, 59.0, 10.7],
                'Woodinville': [47.74887473460722, -122.1022229299363, 74.0, 36.0, 45.0, 13.0],
                'Yarrow Point': [47.6161832807571, -122.20518927444795, 76.0, 37.0, 43.0, 5.5]}

# Convert dictionary into dataframe
df_zipcode = pd.DataFrame.from_dict(zipcode_dict)

# Lookup values based on city selected
for city in df_zipcode.columns:
    if (city == city_selected):
        city_details = df_zipcode[city].tolist()
        lat = city_details[0]
        long = city_details[1]
        summer_high = city_details[2]
        winter_low = city_details[3]
        precipitation = city_details[4]
        violent_crime = city_details[5]

# Show geographical coordinates, environmental features and crime rate
# Plot for temperatures, crime rate and precipitation
fig1, ax1 = plt.subplots(figsize=(12, 0.8))

df_temp = pd.DataFrame({'City':[city_selected,city_selected],
                   'Type':['High','Low'],'Temp':[summer_high,winter_low]})
ax1.barh(df_temp['City'][1], 80, color='whitesmoke')
ax1.barh(df_temp['City'][0], df_temp['Temp'][0], color='indianred', alpha=0.6)
ax1.barh(df_temp['City'][1], df_temp['Temp'][1], color='whitesmoke')

plt.title('Average Annual Temperature Range in ' + city_selected + ' (Fahrenheit)',
          y=1.2)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.get_yaxis().set_ticks([])
ax1.get_xaxis().set_ticks([])
plt.xlim(30, 80)

## Label (Summer temp)
bar1 = ax1.patches[1]
h1 = bar1.get_height()
w1 = bar1.get_width()
x1 = bar1.get_x()
y1 = bar1.get_y()
label_text1 = w1
label_x1 = x1 + w1 + 1
label_y1 = y1 + h1/2
ax1.text(label_x1, label_y1, label_text1, ha='left',    
                va='center')

## Label (Winter temp)
bar2 = ax1.patches[2]
h2 = bar2.get_height()
w2 = bar2.get_width()
x2 = bar2.get_x()
y2 = bar2.get_y()
label_text2 = w2
label_x2 = x2 + w2 -1
label_y2 = y2 + h2/2
ax1.text(label_x2, label_y2, label_text2, ha='right',    
                va='center')
st.pyplot(fig1)

st.write('###')

col1, col2 = st.columns([1,1],gap='large')

with col1:
    fig2, ax2 = plt.subplots(figsize=(5,4.5))
    ax2.pie([violent_crime,100-violent_crime], labels=[violent_crime,''],
            startangle=90, colors=('rosybrown','whitesmoke'))
    white_circle=plt.Circle( (0,0), 0.7, color='white')
    p=plt.gcf()
    p.gca().add_artist(white_circle)
    plt.title('Crime Rate in ' + city_selected + ' (%)', y=1, fontsize=10)
    st.pyplot(fig2)

with col2:
    fig3, ax3 = plt.subplots(figsize=(5,4))
    ax3.bar(city_selected, 80, color='whitesmoke')
    ax3.bar(city_selected, precipitation, color='darksalmon', alpha=0.8)
    plt.title('Average Annual Precipitation in ' + city_selected + ' (inch)',
              y=1.1, fontsize=11)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.get_yaxis().set_ticks([])
    ax3.get_xaxis().set_ticks([])
    plt.ylim(0, 80)

    
    ## Label (Precipitation)
    bar3 = ax3.patches[1]
    h3 = bar3.get_height()
    w3 = bar3.get_width()
    x3 = bar3.get_x()
    y3 = bar3.get_y()
    label_text3 = h3
    label_x3 = x3 + w3/2
    label_y3 = y3 + h3 + 5
    ax3.text(label_x3, label_y3, label_text3, ha='center',    
                va='top')
    
    st.pyplot(fig3)

# Log transformation
sqft_lot = np.log(sqft_lot)
sqft_living = np.log(sqft_living)
precipitation = np.log(precipitation)

# Min max normalization
sqft_living = (sqft_living-5.966147)/(9.396820-5.966147)
sqft_lot = (sqft_lot-6.253829)/(13.887104-6.253829)
view = (view-0)/(4-0)
grade = (grade-4)/(13-4)
lat = (lat-47.155900)/(47.777600-47.155900)
long = (long-(-122.519000))/(-121.315000-(-122.519000))
summer_high = (summer_high-74)/(77-74)
winter_low = (winter_low-34)/(37-34)
precipitation = (precipitation-3.720862)/(4.330733-3.720862)
violent_crime = (violent_crime-5.5)/(41.4-5.5)
age_sold = (age_sold-(-1))/(115-(-1))

# Combine inputs in a list
input_values = [sqft_living, sqft_lot, waterfront, view, grade, lat, long,
               summer_high, winter_low, precipitation, violent_crime, age_sold]

# Load clustering model and categorize data
model = pickle.load(open('km_model.json', 'rb'))
km_pred = model.predict([input_values])
cluster = km_pred[0]

# Convert input values as array
input_array = np.asarray(input_values)
input_array = input_array.reshape(1,-1)

st.write('#')

# Generate prediction
if st.button('Generate Prediction'):
    model = xgb.XGBRegressor()
    if cluster == 0:
        model.load_model('kmxgb_1.json')
    elif cluster == 1:
        model.load_model('kmxgb_2.json')
    elif cluster == 2:
        model.load_model('kmxgb_3.json')
    else:
        model.load_model('kmxgb_4.json')

    price_pred = model.predict(input_array)
    price_pred = price_pred[0]

    # Denormalize price
    price_denorm = price_pred*(15.856731-11.264464)+11.264464
    
    # Exponentiate price
    price_final = np.exp(price_denorm)
    price = '{:,}'.format(round(price_final,2))
    st.markdown('----')
    st.markdown("<h2 style='text-align: center; color: grey;'>Estimated House Price:</h2>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: grey;'>{price} USD</h1>", unsafe_allow_html=True)

st.write('#')


