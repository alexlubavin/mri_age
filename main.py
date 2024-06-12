import streamlit as st
from PIL import Image
import numpy as np

from tensorflow import keras # v==2.12.1

# Стартовая страница

st.title('MRI-AGE')

st.markdown("ОПРЕДЕЛЕНИЕ БИОЛОГИЧЕСКОГО ВОЗРАСТА ПО МРТ С ПОМОЩЬЮ НЕЙРОННОЙ СЕТИ")

# Виджет загрузки изображения МРТ
upload_image = st.file_uploader('Загрузите сагитальный срез МРТ', 
                         type='jpg', 
                         accept_multiple_files=False, 
                         key=None, 
                         help=None, 
                         on_change=None, 
                         args=None, 
                         kwargs=None, 
                         disabled=False, 
                         label_visibility="visible")
# Виджет предпросмотра
try:
    st.image(image = upload_image, 
                     caption=None, 
                     width=None, 
                     use_column_width=True, 
                     clamp=False, 
                     channels="RGB", 
                     output_format="jpg")
    
    # Перевод фото в оттенки серого, 50х50px без обрезки, перевод в одномерный массив
    
    image = Image.open(upload_image)
    
    new_image = image.convert('L')
    
    new_image = new_image.resize((50, 50))
    
    image_to_arr = np.asarray(new_image, dtype='uint8')
    
    image_to_arr = image_to_arr.ravel()
    
    image_to_arr = image_to_arr / 255
    
    inputs = np.reshape(image_to_arr, (1, 50, 50))
    
    model = keras.models.load_model('model')

    res = np.argmax(model.predict(inputs))
    
    if res == 0:
        output = 'Предполагаемый возраст мозга от 35 до 39 лет'
    elif res == 1:
        output = 'Предполагаемый возраст мозга от 40 до 44 лет'
    elif res == 2:
        output = 'Предполагаемый возраст мозга от 45 до 49 лет'
    else:
        output = 'Предполагаемый возраст мозга от 50 до 55 лет'

    st.write(output)
      
except:
    pass




