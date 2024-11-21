# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Otros objetivos
# -----------------------------------------------------------------------
import math

# Gráficos
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

def exploracion_dataframe(dataframe, columna_control, estadisticos = False):
    """
    Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
    valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas
    para columnas categóricas y numéricas, agrupadas por la columna de control.

    Params:
    - dataframe (DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.

    Returns: 
    No devuelve nada directamente, pero imprime en la consola la información exploratoria.
    """
    print(f"El número de datos es {dataframe.shape[0]} y el de columnas es {dataframe.shape[1]}")
    print("\n ..................... \n")

    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ..................... \n")
    
    
    # generamos un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0])
    
    # Tipos de columnas
    print("\n ..................... \n")
    print(f"Los tipos de las columnas son:")
    display(pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"]))
    
    # Enseñar solo las columnas categoricas (o tipo objeto)
    print("\n ..................... \n")
    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = dataframe.select_dtypes(include = "O")
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col.upper()} tiene las siguientes valore únicos:")
        display(pd.DataFrame(dataframe[col].value_counts()).head())    
    
    # como estamos en un problema de A/B testing y lo que realmente nos importa es comparar entre el grupo de control y el de test, los principales estadísticos los vamos a sacar de cada una de las categorías
    if estadisticos == True:
        for categoria in dataframe[columna_control].unique():
            dataframe_filtrado = dataframe[dataframe[columna_control] == categoria]
            #Describe de objetos
            print("\n ..................... \n")

            print(f"Los principales estadísticos de las columnas categóricas para el {categoria.upper()} son: ")
            display(dataframe_filtrado.describe(include = "O").T)

            #Hacer un describe
            print("\n ..................... \n")
            print(f"Los principales estadísticos de las columnas numéricas para el {categoria.upper()} son: ")
            display(dataframe_filtrado.describe().T)
    else: 
        pass

def separarar_df(dataframe):
    return dataframe.select_dtypes(include=np.number), dataframe.select_dtypes(include="O")

def plot_numericas(dataframe,grafica_size = (15,10)):
    cols_numericas = dataframe.columns
    filas = math.ceil(len(cols_numericas)/2)
    fig, axes = plt.subplots(nrows= filas,ncols=2,figsize = grafica_size)
    axes = axes.flat

    for i, col in enumerate(cols_numericas):
        sns.histplot(x= col,data=dataframe,ax= axes[i])
        axes[i].set_title(f"{col}")
        axes[i].set_xlabel("")

    if len(cols_numericas) % 2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass
    plt.tight_layout()

def plot_categoricas(dataframe, paleta="mako",grafica_size = (15,10)):
    cols_categoricas = dataframe.columns
    filas = math.ceil(len(cols_categoricas)/2)
    fig, axes = plt.subplots(nrows= filas,ncols=2,figsize = grafica_size)
    axes = axes.flat

    for i, col in enumerate(cols_categoricas):
        sns.countplot(  x= col,data=dataframe,
                        ax= axes[i],
                        hue = col,
                        palette=paleta,
                        order=dataframe[col].value_counts().index,
                        legend=False)
        axes[i].set_title(f"{col}")
        axes[i].set_xlabel("")
        axes[i].tick_params(rotation=90)
    
    plt.tight_layout()
    if len(cols_categoricas) % 2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass
    

def relacion_vr_categoricas(dataframe,variable_respuesta,paleta="mako",grafica_size = (15,10)):
    df_cat = separarar_df(dataframe)[1]
    cols_categoricas = df_cat.columns
    filas = math.ceil(len(cols_categoricas)/2)
    fig, axes = plt.subplots(nrows= filas,ncols=2,figsize = grafica_size)
    axes = axes.flat

    for indice,columna in enumerate(cols_categoricas):
        datos_agrupados = dataframe.groupby(columna)[variable_respuesta].mean().reset_index().sort_values(variable_respuesta,ascending=False)
        sns.barplot(x= columna,
                    y= variable_respuesta,
                    data = datos_agrupados,
                    ax= axes[indice],
                    hue= columna,
                    legend=False,
                    palette=paleta)
        axes[indice].tick_params(rotation=90)
        axes[indice].set_title(f"Relación entre: {columna} y {variable_respuesta}")
    
    if len(cols_categoricas) % 2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass
    
    plt.tight_layout()

def relacion_vr_numericas(dataframe,variable_respuesta,paleta="mako",grafica_size = (15,10)):
    numericas = separarar_df(dataframe)[0]
    cols_numericas = numericas.columns
    filas = math.ceil(len(cols_numericas)/2)
    fig, axes = plt.subplots(nrows= filas,ncols=2,figsize = grafica_size)
    axes = axes.flat

    for indice,columna in enumerate(cols_numericas):
        if columna == variable_respuesta:
            fig.delaxes(axes[indice])
        else:
            sns.scatterplot(x = columna,
                            y = variable_respuesta,
                            data = numericas,
                            ax = axes[indice],
                            hue = columna,
                            legend = False,
                            palette=paleta)
            axes[indice].set_title(columna)
    
    plt.tight_layout()

    if len(cols_numericas) % 2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass
    
def matriz_correlacion(dataframe):
    plt.figure(figsize=(10,7))
    matriz_corr = dataframe.corr(numeric_only=True)
    mascara = np.triu(np.ones_like(matriz_corr,dtype = np.bool_))
    sns.heatmap(matriz_corr,
                annot=True,
                vmin= -1,
                vmax=1,
                mask = mascara)
    
def detectar_outliers(dataframe,colorear="orange",grafica_size = (15,10)):
    df_num = separarar_df(dataframe)[0]
    num_filas =  math.ceil(len(df_num.columns)/2)

    fig, axes = plt.subplots(ncols=2,nrows=num_filas, figsize= grafica_size)
    axes = axes.flat

    for indice, columna in enumerate(df_num.columns):

        sns.boxplot(x=columna,
                    data = df_num,
                    ax= axes[indice],
                    color= colorear,
                    flierprops = {"markersize":5,"markerfacecolor":"red"})
        axes[indice].set_title(f"Outliers de {columna}")
        axes[indice].set_xlabel("")
    if len(df_num.columns) % 2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass

    plt.tight_layout()


# Esta me la tengo que ver para ver que funcione
def diferencia_tras_rellenar_nulos(df_before, df_after):
    # Obtener el describe de ambos DataFrames y solo media mediana y desviacion estándar

    describe_before = df_before.describe().T
    describe_after = df_after.describe().T
        
    # Calcular el porcentaje de cambio
    difference_percent = ((describe_after - describe_before) / describe_before) * 100
        
    # Mostrar las tres tablas: describe antes, describe después y porcentaje de cambio
    print("\n ..................... \n")
    print("Estadísticas antes de la operación:")
    display(describe_before)
    print("\n ..................... \n")
    print("Estadísticas después de la operación:")
    display(describe_after)
    print("\n ..................... \n")
    print("Diferencia porcentual:")
    display(difference_percent.fillna(0))  # Llenar NaN con 0 en caso de que no haya cambios
