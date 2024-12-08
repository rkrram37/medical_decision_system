import numpy as np 
import pandas as pd 
import streamlit as st

df = pd.read_csv('dataset.csv')
df['SEX'] = df['SEX'].map({'M': 0 ,'F': 1})

x = df.iloc[:, :10]
y = df.iloc[:, -1]		

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=44)
model.fit(x, y)


def main():
    background_image = """
    <style>
    p, div
    {
    color:#000;
    }
    </style>
    """
    
    
    from PIL import Image
    img = Image.open("1.webp")

    # display image using streamlit
    # width is used to set the width of an image
    st.image(img, width=800)
    
    
    st.markdown(background_image, unsafe_allow_html=True)
    st.header("Intelligent Medical Decision Support System")
    
    HAEMATOCRIT = st.text_input('HAEMATOCRIT')
    HAEMOGLOBINS = st.text_input('HAEMOGLOBINS')
    ERYTHROCYTE = st.text_input('ERYTHROCYTE')
    LEUCOCYTE = st.text_input('LEUCOCYTE')
    THROMBOCYTE = st.text_input('THROMBOCYTE')
    MCH = st.text_input('MCH')
    MCHC = st.text_input('MCHC')
    MCV = st.text_input('MCV')
    AGE = st.text_input('AGE')
    SEX = st.selectbox("SEX: ",
                     ['0', '1'])
    
    if(st.button('Submit')):
   
        input_data = pd.DataFrame(
            {
                "HAEMATOCRIT": [HAEMATOCRIT],
                "HAEMOGLOBINS": [HAEMOGLOBINS],
                "ERYTHROCYTE": [ERYTHROCYTE],
                "LEUCOCYTE": [LEUCOCYTE],
                "THROMBOCYTE": [THROMBOCYTE],
                "MCH": [MCH],
                "MCHC": [MCHC],
                "MCV": [MCV],
                "AGE": [AGE],
                "SEX": [SEX],
            }
        )

        prediction = model.predict(input_data)
        result = prediction[0]
        
        if(result == 1.0):
            st.balloons() 
            st.error("You need Medical Decision Support System")
        
        elif(result == 0.0):
            st.snow()
            st.success("You are Healthy, You do not need Medical Decision Support System")            
    



if __name__ == "__main__":
    main()
