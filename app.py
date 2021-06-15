import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import imageio
import webbrowser
import cv2

data = pd.read_csv('artists.csv')
# Sort artists by number of paintings
data = data.sort_values(by=['paintings'], ascending=False)

# Create a dataframe with artists having more than 200 paintings
data = data[data['paintings'] >= 200].reset_index()
data = data[['name', 'paintings']]
                        #total paintins 4299 / (20 -> top 20 * artist 'x' paitings)
data['class_weight'] = data.paintings.sum() / (data.shape[0] * data.paintings)

# Set class weights - assign higher weights to underrepresented classes
class_weights = data['class_weight'].to_dict()
labels = {0: 'Vincent_van_Gogh',
 1: 'Edgar_Degas',
 2: 'Pablo_Picasso',
 3: 'Pierre-Auguste_Renoir',
 4: 'Albrecht_Dürer',
 5: 'Paul_Gauguin',
 6: 'Francisco_Goya',
 7: 'Rembrandt',
 8: 'Alfred_Sisley',
 9: 'Titian',
 10: 'Marc_Chagall'}
# There is some problem recognizing 'Albrecht_Dürer' (don't know why, worth exploring)
# So I'll update this string as directory name to df's
updated_name = "Albrecht_Dürer".replace("_", " ")
data.iloc[4, 0] = updated_name

def load_model():
  model=tf.keras.models.load_model('my_model.h5')
  return model

model = load_model()

def predict(url,model):
    web_image = imageio.imread(url)
    web_image = cv2.resize(web_image, dsize=model.input_shape[1:3], )
    web_image = tf.keras.preprocessing.image.img_to_array(web_image)
    web_image = web_image/255.
    web_image = np.expand_dims(web_image, axis=0)
    prediction = model.predict_on_batch(web_image)
    data = df_percent(prediction,labels)
    plt.imshow(imageio.imread(url))
    plt.axis('off')
    #plt.show()
    st.pyplot()
    return data


def df_percent(prediction,labels):
    prediction = prediction.flatten()
    artists = pd.DataFrame(labels,index=[0]).T
    artists['proba'] = prediction
    artists.rename(columns={0:'artist name'}, inplace=True)
    artists.set_index('artist name',inplace=True)
    return artists

#url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/Vicente_L%C3%B3pez_Porta%C3%B1a_-_el_pintor_Francisco_de_Goya.jpg/800px-Vicente_L%C3%B3pez_Porta%C3%B1a_-_el_pintor_Francisco_de_Goya.jpg'


#make dataframe of urls paintings
paintings = {
    #Van Gogh
    'Sunflowers':'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Vincent_van_Gogh_-_Sunflowers_-_VGM_F458.jpg/240px-Vincent_van_Gogh_-_Sunflowers_-_VGM_F458.jpg',
    'Wheatfild with Crows':'https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/Vincent_van_Gogh_-_Wheatfield_with_crows_-_Google_Art_Project.jpg/260px-Vincent_van_Gogh_-_Wheatfield_with_crows_-_Google_Art_Project.jpg',
    'The Yellow House':'https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Vincent_van_Gogh_-_The_yellow_house_%28%27The_street%27%29.jpg/220px-Vincent_van_Gogh_-_The_yellow_house_%28%27The_street%27%29.jpg',
    'Starry Night':'https://i0.wp.com/arteref.com/wp-content/uploads/2021/02/Noite-estrelada-Van-Gogh-1024x811-1.jpg?resize=600%2C475&ssl=1',
    'Arles Room':'https://i0.wp.com/arteref.com/wp-content/uploads/2021/02/O-quarto-em-Arles.jpg?w=756&ssl=1',

    #Edgar_Degas
    'Orquestra da opera':'https://dasartes.com.br/dasartes.com.br/wp-content/uploads/2021/01/OYcJyOdg-768x1020.jpg',
    'Familia Bellelli':'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/Edgar_Germain_Hilaire_Degas_049.jpg/290px-Edgar_Germain_Hilaire_Degas_049.jpg',
    'Prima Ballerina':'https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Edgar_Germain_Hilaire_Degas_018.jpg/250px-Edgar_Germain_Hilaire_Degas_018.jpg',

    #Pablo_Picasso
    'Guernica':'https://i0.wp.com/arteref.com/wp-content/uploads/2020/10/Painel-Guernica-de-Pablo-Picasso-1937-Museu-Reina-Sofia-em-Madri-Espanha-1024x463-1.jpg?w=1024&ssl=1',
    'Les Demoiselles':'https://i1.wp.com/arteref.com/wp-content/uploads/2019/11/picasso-mate%CC%81ria.jpg?resize=600%2C623&ssl=1',
    'Three Musicians':'https://i2.wp.com/arteref.com/wp-content/uploads/2020/10/three-musicians-1.jpg?resize=600%2C538&ssl=1',
    'Family of Saltimbanques':'https://i1.wp.com/arteref.com/wp-content/uploads/2020/10/family-of-saltimbanques-1.jpg?resize=600%2C550&ssl=1',

    #Pierre-Auguste_Renoir
    'Duas irmãs':'https://i2.wp.com/arteref.com/wp-content/uploads/2019/12/Duas-irma%CC%83s-no-terrac%CC%A7o-1881.jpg?w=560&ssl=1',
    'As Grandes Banhistas':'https://i1.wp.com/arteref.com/wp-content/uploads/2019/12/As-Grandes-Banhistas-1887.jpg?w=700&ssl=1',


    #Albrecht_Dürer


    #Paul_Gauguin
    'Vallée Bretonne':'https://d32dm0rphc51dk.cloudfront.net/jsaOG8cOknnRajFUmbnn9g/large.jpg',
    'Arlésiennes':'https://i1.wp.com/www.historiadasartes.com/wp-content/uploads/2019/04/m_GauguinArlesiennes.jpg?resize=640%2C506',
    'nafea faa ipoipo':'http://www.pixelle.co/wp-content/uploads/2016/08/Paul-Gauguin-slider.jpg',

    #Francisco_Goya
    'The Spell':'https://i.pinimg.com/originals/ce/57/fa/ce57fadb4e6009e568e5480e17dbf406.jpg',
    'El Aquelarre':'https://i.pinimg.com/originals/4a/25/bc/4a25bcc01f4991ec8e23e18f2c47cc48.jpg',
    'Mi Síndrome de Diógenes':'https://i.pinimg.com/originals/1b/0c/ee/1b0cee005e4e15962f452a4debf333c5.jpg',


    #Rembrandt
    'The Night Watch':'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/The_Night_Watch_-_HD.jpg/1200px-The_Night_Watch_-_HD.jpg',
    'Parable of the Rich Man':'http://upload.wikimedia.org/wikipedia/commons/c/ca/Rembrandt_-_Parable_of_the_Rich_Man_-_WGA19247.jpg',

    #Alfred_Sisley
    'The Bridge at Villeneuve la Garenne':'https://1.bp.blogspot.com/-WjTjaX6TMWA/WffJYZ8YXfI/AAAAAAAAdj8/4Exh0Uk4_uQmhEW7eSxHSSvvpMeJqw6gACLcBGAs/s640/Alfred%2BSisley%2B-%2BThe%2BBridge%2Bat%2BVilleneuve-la-Garenne%252C%2B1872%2B%25E2%2580%2593%2B%25C3%25B3leo%2Bsobre%2Btela%2B%25E2%2580%2593%2B49%252C5%2Bx%2B65%252C4%2Bcm%2B%25E2%2580%2593%2BMetropolitan%2BMuseum%2Bof%2BArt%252C%2BNew%2BYork%252C%2BUSA.jpg',
    'The Moret Bridge':'https://artsdot.com/ADC/Art-ImgScreen-3.nsf/O/A-8DNVDZ/$FILE/Alfred_sisley-the_moret_bridge_in_the_sunlight.Jpg',
    'Moret sur Loing':'https://upload.wikimedia.org/wikipedia/commons/e/e8/Alfred_Sisley_-_Moret-sur-Loing_%281891%29.jpg',

    #Titian
    'Rapto de Europa':'https://s.ebiografia.com/img/or/ap/o_rapto_de_europa_c.jpg',
    'Assunção da Virgem':'https://hav320142.files.wordpress.com/2014/10/ticiano_08.jpg',
    'Amor sacro e Amor Profano':'https://s.ebiografia.com/img/am/or/amor_sacro_e_amor_profano_c.jpg',
    'Baccanale degli Andrii':'https://upload.wikimedia.org/wikipedia/commons/b/b0/Titian_Bacchanal_1523_1524.jpg',


    #Marc_Chagall
    'Eu e a aldeia':'https://s.ebiografia.com/img/eu/ea/eu_e_a_aldeia_1911.jpg',
    ' les mariés de la tour eiffel':'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTvO0KHO-7yHgvZEbRUsYGghTaWM2ZxYqLiTOHdEwqJzon-gTN7F0iRkp2kUkn5pd5eWNI&usqp=CAU',
    'Paris through the window':'https://wahooart.com/Art.nsf/O/8XYGKP/$File/Marc-Chagall-Paris-through-the-Window.JPG'

}

paint = pd.DataFrame(paintings.items(),columns=['obras','url'])


"""
# Paintings Classification

"""
st.set_option('deprecation.showPyplotGlobalUse', False)

add_selectbox = st.sidebar.selectbox(
    "Select your painting and copy url",
    paint['obras']
)

st.table(paint[paint['obras']==add_selectbox])

url = st.text_input('The painting URL link')
if url:
    result = predict(url,model)
    st.write("Predicted artist =", result['proba'].idxmax())
    st.write("Prediction probability =", result['proba'].max()*100, "%")
    st.dataframe(result.sort_values('proba',ascending=False))
    #st.write("The bibliograpy of " + result['proba'].idxmax())

    if st.button("The bibliograpy of " + result['proba'].idxmax()):
        webbrowser.open_new_tab('https://pt.wikipedia.org/wiki/'+result['proba'].idxmax())
