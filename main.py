import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# Sayfa Ayarları
st.set_page_config(
#sayfa adı yukardaki
   page_title="California Housing Price Prediction",

#projeye eklediğim ikonu kullandım
    page_icon="images/house.png",

#Get help alanı ve about alanı (sağdaki üç nokta)
    menu_items={
        "Get help": "mailto:datascience@gmail.com",
        "About": "For More Information\n" + "https://github.com/bnurdemirhan/DSNov22"
    }
)

# Başlık Ekleme (sitedeki ana başlık)
st.title("California Housing Price Prediction")

# Markdown Oluşturma
st.markdown("A machine learning model is developed to predict the housing price based on various features of houses._")


# Resim Ekleme
st.image("images/housecalifornia.jpg")

st.markdown("This application predicts housing prices using data from California Housing Dataset.")
st.markdown("*Let's help the research team!*")

st.image("https://resources.pollfish.com/wp-content/uploads/2020/11/MARKET_RESEARCH_FOR_REAL_ESTATE_IN_CONTENT_1.png")

# Header Ekleme
st.header("Data Dictionary")

st.markdown("- **median_income**: Median income of the block")

# Pandasla veri setini okuyalım
df = pd.read_pickle("train_df.pkl")

#verisetinden 8 örnek getir,her sayfa değiştiğinde farklı önrnek
st.dataframe(df.sample(8))

#sidebbar
