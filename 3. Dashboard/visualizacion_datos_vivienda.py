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
estratos = sorted(df_base['estrato'].dropna().unique())

# Crear los selectores para los filtros
ciudad_seleccionada = st.sidebar.multiselect("Selecciona una ciudad:", ciudades, default = ciudades)
proyecto_seleccionado = st.sidebar.multiselect("Selecciona un proyecto:", proyectos, default = proyectos)
estratos_seleccionados = st.sidebar.multiselect("Selecciona uno o más estratos:", estratos, default=estratos)

# Aplicar los filtros
df_filtrado = df_base.copy()

if ciudad_seleccionada:
    df_filtrado = df_filtrado[df_filtrado['ciudad'].isin(ciudad_seleccionada)]

if proyecto_seleccionado:
    df_filtrado = df_filtrado[df_filtrado['proyecto'].isin(proyecto_seleccionado)]

if estratos_seleccionados:
    df_filtrado = df_filtrado[df_filtrado['estrato'].isin(estratos_seleccionados)]

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
        hover_data=['estrato', 'alcobas', 'banos', 'precio'],
        title="Precio vs. Área"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Vista tabular de los datos filtrados
    st.subheader("Datos Filtrados")
    st.dataframe(df_filtrado[['precio', 'area_m2', 'estrato', 'alcobas', 'banos', 'proyecto']])

else:
    st.warning("No se encontraron datos que coincidan con los filtros seleccionados. Por favor, ajusta los criterios.")


st.subheader("Distribución de Precios")
fig_hist_precio = px.histogram(df_filtrado, x="precio", nbins=50, title="Distribución de Precios")
st.plotly_chart(fig_hist_precio, use_container_width=True)

st.subheader("Distribución de Área (m2)")
fig_hist_area = px.histogram(df_filtrado, x="area_m2", nbins=50, title="Distribución de Área (m²)")
st.plotly_chart(fig_hist_area, use_container_width=True)

#

if 'destacado' in df_filtrado.columns:
    st.subheader("Destacados vs. No Destacados")
    
    # Calcular precio promedio
    df_destacado = df_filtrado.groupby('destacado')['precio'].mean().reset_index()
    df_destacado['destacado'] = df_destacado['destacado'].replace({1: 'Destacado', 0: 'No Destacado'})
    
    fig_barra_destacado = px.bar(
        df_destacado, 
        x='destacado', 
        y='precio', 
        title='Precio Promedio de Propiedades',
        labels={'destacado': 'Tipo de Proyecto', 'precio': 'Precio Promedio'},
        color='destacado'
    )
    st.plotly_chart(fig_barra_destacado, use_container_width=True)

st.subheader("Proyectos con del m2 más barato")
# Calcular el precio por m²
df_filtrado['precio_por_m2'] = df_filtrado['precio'] / df_filtrado['area_m2']

# Obtener los 5 proyectos con el precio por m² más bajo
proyectos_baratos = df_filtrado.groupby('proyecto')['precio_por_m2'].mean().nsmallest(5).reset_index()
proyectos_baratos.columns = ['Proyecto', 'Precio Promedio por m²']

# Formatear la columna de precios con el símbolo $
proyectos_baratos['Precio Promedio por m²'] = proyectos_baratos['Precio Promedio por m²'].map('${:,.2f}'.format)

st.dataframe(proyectos_baratos)
