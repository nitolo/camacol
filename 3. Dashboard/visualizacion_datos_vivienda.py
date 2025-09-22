# ==============================================================================
# PRUEBA TÉCNICA CAMACOL - VISUALIZACIÓN DE DATOS
# ==============================================================================

# Autor: Nicolás Torres
# - Elaboro un dashboard con Streamlit para que sea más facilmente reproducible

# ==============================================================================
# LIBRERÍAS
# ==============================================================================

# Importamos las librerías necesarias

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import streamlit as st
import plotly.express as px

# ==============================================================================
# TITULO DEL DASHBOARD
# ==============================================================================
st.title("Dashboard de Precios de Propiedades")
st.markdown("---")

# ==============================================================================
# CARGA DE DATOS
# ==============================================================================
#ruta_base = "C:/Users/ntorreslo/Downloads/CAMACOL/3. Dashboard/"

@st.cache_data
def load_data():
    #df = pd.read_excel(os.path.join(ruta_base, "ws_fr_vn_ver2_cs_limpia_mineria.xlsx"))
    df = pd.read_excel("ws_fr_vn_ver2_cs_limpia_mineria.xlsx")
    return df

df_base = load_data()

# ==============================================================================
# FILTROS
# ==============================================================================
st.header("Filtros de Datos")

# Lista de opciones para los filtros
ciudades = sorted(df_base['ciudad'].dropna().unique())
proyectos = sorted(df_base['proyecto'].dropna().unique())

# Crear los selectores para los filtros
ciudad_seleccionada = st.selectbox("Selecciona una ciudad:", ['Todas'] + ciudades)
proyecto_seleccionado = st.selectbox("Selecciona un proyecto:", ['Todos'] + proyectos)

# Aplicar los filtros
df_filtrado = df_base.copy()

if ciudad_seleccionada != 'Todas':
    df_filtrado = df_filtrado[df_filtrado['ciudad'] == ciudad_seleccionada]

if proyecto_seleccionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['proyecto'] == proyecto_seleccionado]


# ==============================================================================
# VISUALIZACIONES
# ==============================================================================

st.header("Análisis de Datos")

if not df_filtrado.empty:
    st.write(f"Mostrando {len(df_filtrado)} registros.")
    
    # Gráfico de dispersión para ver la relación entre precio y área
    st.subheader("Relación entre Precio y Área (m²)")
    fig_scatter = px.scatter(
        df_filtrado,
        x="area_m2",
        y="precio",
        hover_data=['descripcion', 'estrato', 'alcobas', 'banos', 'precio'],
        title="Precio vs. Área"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Vista tabular de los datos filtrados
    st.subheader("Datos Filtrados")
    st.dataframe(df_filtrado[['precio', 'area_m2', 'estrato', 'alcobas', 'banos', 'descripcion']])

else:
    st.warning("No se encontraron datos que coincidan con los filtros seleccionados. Por favor, ajusta los criterios.")


