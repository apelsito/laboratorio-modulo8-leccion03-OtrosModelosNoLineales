import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns 

def visualizar_categoricas(dataframe,lista_cols_categoricas,variable_respuesta,tipo_grafica = "boxplot", grafica_size=(15,10),paleta="mako",barplot_calc="mean"):
    
    num_filas = math.ceil(len(lista_cols_categoricas)/2)

    fig , axes = plt.subplots(ncols=2 , nrows=num_filas,figsize= grafica_size)
    axes = axes.flat

    for indice, columna in enumerate(lista_cols_categoricas):
        if tipo_grafica.lower() == "boxplot":
            sns.boxplot(x= columna,
                        y= variable_respuesta,
                        data=dataframe,
                        whis = 1.5,
                        hue=columna,
                        legend=False,
                        ax = axes[indice])
        elif tipo_grafica.lower() == "barplot":
            sns.barplot(x=columna,
                        y= variable_respuesta,
                        ax = axes[indice],
                        estimator=barplot_calc,
                        palette=paleta,
                        data=dataframe)
        else:
            print("Debes elegir entre boxplot y barplot")
    
        axes[indice].set_title(f"Relación {columna} con {variable_respuesta}")
        axes[indice].set_xlabel("")
        axes[indice].tick_params(rotation=90)

    if num_filas % 2 != 0:
        fig.delaxes(axes[-1])   
    plt.tight_layout()
