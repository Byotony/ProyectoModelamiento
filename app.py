from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from pandas import ExcelWriter
import io
from io import BytesIO
import base64    

arranque = Flask(__name__)


@arranque.route("/")
def inicio():
    return render_template("index.html")

@arranque.route("/manual")
def manual():
    return render_template("manual.html")

@arranque.route("/pronostico")
def pronostico():
    return render_template("pronostico.html")

@arranque.route("/nunaleatorio")
def nunaleatorio():
    return render_template("nunaleatorio.html")

@arranque.route("/montecarlo")
def montecarlo():
    return render_template("montecarlo.html")

@arranque.route("/montecarloejemplo")
def montecarloejemplo():
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import io
    import base64
    datos = pd.DataFrame()
    demanda = [100, 200, 300, 400, 500, 600]
    probabilidad = [0.15, 0.25, 0.18, 0.22, 0.13, 0.07]
    datos["DEMANDA"] = demanda
    datos["PROBABILIDAD"] = probabilidad
    data = datos.to_html(classes="table table-hover table-striped",
                   justify="justify-all", border=0)
    buf = io.BytesIO() ##
    plt.plot(datos)
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)##
    canvas.print_png(buf)##
    fig.clear()##
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')##


    a1= np.cumsum(datos["PROBABILIDAD"]) #Cálculo la suma acumulativa de las probabilidades
    x2=datos
    x2['FPA'] =a1
    data2 = x2.to_html(classes="table table-hover table-striped",
                   justify="justify-all", border=0)

   
    x2['Min'] = x2['FPA']
    x2['Max'] = x2['FPA']
    x2
    data3 = x2.to_html(classes="table table-hover table-striped",
                   justify="justify-all", border=0)


    
    lis = x2["Min"].values
    lis2 = x2['Max'].values
    lis[0]= 0
    for i in range(1,6):
        lis[i] = lis2[i-1]
        print(i,i-1)
    x2['Min'] = lis
    data4 = x2.to_html(classes="table table-hover table-striped",
                   justify="justify-all", border=0)


    datos2 = pd.DataFrame()
    ri = [0.11, 0.44, 0.90, 0.52, 0.00, 0.54, 0.56, 0.66, 0.52, 0.46, 0.24, 0.31, 0.48, 0.03, 0.50, 0.65, 0.80, 0.74, 0.32, 0.66]
    datos2 ["ri"] = ri
    data5 = datos2.transpose().to_html(classes="table table-hover table-striped",
                   justify="justify-all", border=0)



    max = x2 ['Max'].values
    min = x2 ['Min'].values
    datosv1= []
    simu=pd.DataFrame()
    for a in range(len(datos2)):
        for b in range(len(x2)):
            if(datos2["ri"][a]>=x2["Min"][b] and datos2["ri"][a]<x2["Max"][b]):
                datosv1.append(x2["DEMANDA"][b])
    simu["ri"]=datos2["ri"]
    simu["DEMANDA"]=datosv1
    data6 = simu.transpose().to_html(classes="table table-hover table-striped",
                   justify="justify-all", border=0)


    x=simu["DEMANDA"].sum()
    promedio=x/28
    
    data7 = promedio


    return render_template('montecarloejemplo.html',data=data,data2=data2,data3=data3,data4=data4,data5=data5,data6=data6,data7=data7,image=plot_url)

@arranque.route("/mcuadradosmedios",methods=["GET"]) #refrescar la pagina
def rendercuadradosmedios():
    return render_template("mcuadradosmedios.html")

@arranque.route("/mcuadradosmedios",methods=["POST"])
def mcuadradosmedios():
    
    import matplotlib.pyplot as plt

    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    
    n = request.form.get('numeroIteraciones', type=int)
    r = request.form.get('semilla', type=int)

    l = len(str(r))
    lista = []
    lista2 = []
    i = 1
    while i <= n:
        x = str(r*r)
        if l % 2 == 0:
            x = x.zfill(l*2)
        else:
            x = x.zfill(l)
        y = (len(x)-l)/2
        y = int(y)
        r = int(x[y:y+l])
        lista.append(r)
        lista2.append(x)
        i = i+1
    df = pd.DataFrame({'X2': lista2, 'Xi': lista})
    dfrac = df["Xi"]/10**l
    df['ri'] = dfrac

    buf = io.BytesIO()
    x1 = df['ri']
    plt.plot(x1)
    plt.title('Generador de Números Aleatorios Cuadrados Medios')
    plt.xlabel('Serie')
    plt.ylabel('Aleatorios')
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    data = df.to_html(classes="table table-hover table-striped",
                      justify="justify-all", border=0)

    return render_template('mcuadradosmedios.html',data=data,image=plot_url)


@arranque.route("/mcongruencialineal", methods=["GET"]) #refrescar la pagina
def mcongruencialineal():
    return render_template("mcongruencialineal.html")


@arranque.route("/mcongruencialineal", methods=["POST"]) #refrescar la pagina
def rendercongruencialineal():
    
    n = request.form.get("numeroIteraciones", type=int)
    x0 = request.form.get("semilla", type=int)
    a = request.form.get("multiplicador", type=int)
    c = request.form.get("incremento", type=int)
    m = request.form.get("modulo", type=int)

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import ExcelWriter
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO
    import base64

    #n, m, a, x0, c = 20,1000,101,4,457
    x = [1]*n
    r = [0.1]*n
    for i in range(0, n):
        x[i] = ((a*x0)+c) % m
        x0 = x[i]
        r[i] = x0/m
    df = pd.DataFrame({'Xn': x, 'ri': r})

    # Graficamos los numeros generados
    buf = io.BytesIO()
    plt.plot(r, marker='o')
    plt.title('Generador de Números Aleatorios Congruencial Lineal')
    plt.xlabel('Serie')
    plt.ylabel('Aleatorios')
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    data = df.to_html(classes="table table-hover table-striped",
                      justify="justify-all", border=0)

    return render_template('mcongruencialineal.html', data=data, image=plot_url)


@arranque.route("/mcongruenciamulti", methods=["GET"]) #refrescar la pagina
def mcongruenciamulti():
    return render_template("mcongruenciamulti.html")

@arranque.route("/mcongruenciamulti", methods=["POST"]) #refrescar la pagina
def rendercongruenciamulti():

    n = request.form.get("numeroIteraciones", type=int)
    x0 = request.form.get("semilla", type=int)
    a = request.form.get("multiplicador", type=int)
    m = request.form.get("modulo", type=int)

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import ExcelWriter
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO
    import base64

    # n, m, a, x0 = 20, 1000, 747, 123
    x = [1] * n
    r = [0.1] * n
    for i in range(0, n):
        x[i] = (a*x0) % m
        x0 = x[i]
        r[i] = x0 / m
    d = {'Xn': x, 'ri': r}
    df = pd.DataFrame(data=d)

    buf = io.BytesIO()
    plt.plot(r, 'g-', marker='o',)
    plt.title('Generador de Números Aleatorios Congruencial Multiplicativo')
    plt.xlabel('Serie')
    plt.ylabel('Aleatorios')
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    data = df.to_html(classes="table table-hover table-striped",
                      justify="justify-all", border=0)
    
    return render_template('mcongruenciamulti.html', data=data, image=plot_url)


@arranque.route("/pmediamovil")
def pmediamovil():

    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    import base64
    from matplotlib.backends.backend_agg import FigureCanvasAgg 
    
    datos = {'JUGADOR':['zombs', 'ShahZaM', 'dapr', 'SicK', 
                  'cNed', 'starxo', 'Kiles', 'nAts', 'Chronicle', 'd3ffo'],
                'DIAMANTE':[13, 13, 14, 12, 12, 12, 6, 14, 13, 12],
                'ORO':[5, 5, 3, 3, 4, 4, 4, 6, 6, 4],
                'PLATA':[1, 1, 1, 1, 2, 1, 2, 0, 0, 2],
                'TIERLIST':[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                'EARNINGS':[113.55, 113.45, 112.87, 112.65, 107.735, 104.394, 104.093, 103.829, 103.574, 102.773]}
    data = pd.DataFrame(datos).to_html(classes="table table-hover table-striped", justify="justify-all", border=2)
    

    datos1 = {'JUGADOR':['zombs', 'ShahZaM', 'dapr', 'SicK', 
                  'cNed', 'starxo', 'Kiles', 'nAts', 'Chronicle', 'd3ffo'],
                'EARNINGS':[113.55, 113.45, 112.87, 112.65, 107.735, 104.394, 104.093, 103.829, 103.574, 102.773]}
    #print(datos1)
    movil = pd.DataFrame(datos1)
    moviles = movil.transpose().to_html(classes="table table-hover table-striped", justify="justify-all", border=2)

    # mostramos los 5 primeros registros
    print(moviles)

    # calculamos para la primera media móvil MMO_3
    for i in range(0,movil.shape[0]-2):
        movil.loc[movil.index[i+2],'MMO_3'] = np.round(((movil.iloc[i,1]+movil.iloc[i+1,1]+movil.iloc[i+2,1])/3),1)
        
    # calculamos para la segunda media móvil MMO_4
    for i in range(0,movil.shape[0]-3):
        movil.loc[movil.index[i+3],'MMO_4'] = np.round(((movil.iloc[i,1]+movil.iloc[i+1,1]+movil.iloc[i+2,1]+movil.iloc[i+
    3,1])/4),1)
        
    # calculamos la proyeción final
    proyeccion = movil.iloc[7:,[1,2,3]]
    p1,p2,p3 =proyeccion.mean()

    # incorporamos al DataFrame
    a = movil.append({'JUGADOR':'Random','EARNINGS':p1, 'MMO_3':p2, 'MMO_4':p3},ignore_index=True)
    # mostramos los resultados
    a['e_MM3'] = a['EARNINGS']-a['MMO_3']
    a['e_MM4'] = a['EARNINGS']-a['MMO_4']
    a
    movillle=a.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)

    buf = io.BytesIO()
    plt.figure(figsize=[4,4])
    
    fig = plt.gcf()
    
    plt.grid(True)
    plt.plot(a['EARNINGS'],label='EARNINGS',marker='o')
    plt.plot(a['MMO_3'],label='Media Móvil Random')
    plt.plot(a['MMO_4'],label='Media Móvil Random')
    plt.legend(loc=2)
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')


    movil = pd.DataFrame(datos1)
    
    # mostramos los 5 primeros registros
    
    alfa = 0.1
    unoalfa = 1. - alfa
    for i in range(0,movil.shape[0]-1):
        movil.loc[movil.index[i+1],'SN'] = np.round(movil.iloc[i,1],1)
    for i in range(2,movil.shape[0]):
        movil.loc[movil.index[i],'SN'] = np.round(movil.iloc[i-1,1],1)*alfa + np.round(movil.iloc[i-1,2],1)*unoalfa
    i=i+1
    p1=0
    p2=np.round(movil.iloc[i-1,1],1)*alfa + np.round(movil.iloc[i-1,2],1)*unoalfa
    a = movil.append({'JUGADOR':'Random','EARNINGS':p1, 'SN':p2},ignore_index=True)
    print(a)
    tabla2 = a.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)
    
    
    a = pd.DataFrame(datos1)
    x = a.index.values
    y= a["EARNINGS"]
    # ajuste de la recta (polinomio de grado 1 f(x) = ax + b)
    p = np.polyfit(x,y,1) # 1 para lineal, 2 para polinomio ...
    p0,p1 = p
    print ("El valor de p0 = ", p0, "Valor de p1 = ", p1)


    buf = io.BytesIO()
    plt.figure(figsize=[4,4])
    
    fig = plt.gcf()
    y_ajuste = p[0]*x + p[1]
    print (y_ajuste)
    # dibujamos los datos experimentales de la recta
    p_datos =plt.plot(x,y,'b.')
    # Dibujamos la recta de ajuste
    p_ajuste = plt.plot(x,y_ajuste, 'r-')
    plt.title('Ajuste lineal por mínimos cuadrados')
    plt.xlabel('Eje x')
    plt.ylabel('Eje y')
    plt.legend(('Datos experimentales','Ajuste lineal',), loc="upper left")
    
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
   
    plot_url1 = base64.b64encode(buf.getvalue()).decode('UTF-8')

    buf = io.BytesIO()
    plt.figure(figsize=[4,4])
    p = np.polyfit(x,y,2)
    p1,p2,p3 = p
    print ("El valor de p0 = ", p0, "Valor de p1 = ", p1, " el valor de p2 = ",p2)
    
    y_ajuste = p[0]*x*x + p[1]*x + p[2]
    # dibujamos los datos experimentales de la recta
    p_datos =plt.plot(x,y,'b.')
    # Dibujamos la curva de ajuste
    p_ajuste = plt.plot(x,y_ajuste, 'r-')
    plt.title('Ajuste Polinomial por mínimos cuadrados')
    plt.xlabel('Eje x')
    plt.ylabel('Eje y')
    plt.legend(('Datos experimentales','Ajuste Polinomial',), loc="upper left")
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
   
    plot_url2 = base64.b64encode(buf.getvalue()).decode('UTF-8')
    
    n=x.size
    x1 = []
    x2 = []
    for i in [12,13]:
        y1_ajuste = p[0]*i*i + p[1]*i + p[2]
        print (f" z = {i} w = {y1_ajuste}")
        x1.append(i)
        x2.append(y1_ajuste)
        
    a["y_ajuste"]=y_ajuste

    dp = pd.DataFrame({'JUGADOR':['Random','Random'], 'EARNINGS':[0,0],'y_ajuste':x2})
    dp
    a = a.append(dp,ignore_index=True)
    print(a)
    tabla3 = a.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)
    
    buf = io.BytesIO()
    plt.figure(figsize=[4,4])
    x = a.index.values
    y_ajuste = a["y_ajuste"]
    y= a["EARNINGS"]
    p_datos =plt.plot(y,'b.')
    p_ajuste = plt.plot(x,y_ajuste, 'r-')
    plt.title('Ajuste Polinomial por mínimos cuadrados')
    plt.xlabel('Eje x')
    plt.ylabel('tasa Pasiva Referencial')
    plt.axvspan(0,9,alpha=0.3,color='y') # encajonamos los datos iniciales
    plt.legend(('Datos experimentales','Ajuste Polinomial',), loc="upper left")
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
   
    plot_url3 = base64.b64encode(buf.getvalue()).decode('UTF-8')



    return render_template("pmediamovil.html", data=data, movillle=movillle, movil=moviles, image=plot_url, tabla2=tabla2, image2=plot_url1, image3=plot_url2, tabla3=tabla3,
    image4=plot_url3) 


@arranque.route("/psuavizacion") #refrescar la pagina
def suavizacion():

    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import numpy as np
    # leemos los datos de la tabla del directorio Data de trabajo
    datos = {'JUGADOR':['zombs', 'ShahZaM', 'dapr', 'SicK', 
                  'cNed', 'starxo', 'Kiles', 'nAts', 'Chronicle', 'd3ffo'],
                'EARNINGS':[113.55, 113.45, 112.87, 112.65, 107.735, 104.394, 104.093, 103.829, 103.574, 102.773]}
    movil = pd.DataFrame(datos)

    # mostramos los 5 primeros registros
    movil.head()
    alfa = 0.1
    unoalfa = 1. - alfa
    for i in range(0,movil.shape[0]-1):
        movil.loc[movil.index[i+1],'SN'] = np.round(movil.iloc[i,1],1)
    for i in range(2,movil.shape[0]):
        movil.loc[movil.index[i],'SN'] = np.round(movil.iloc[i-1,1],1)*alfa + np.round(movil.iloc[i-1,2],1)*unoalfa
    i=i+1
    p1=0
    p2=np.round(movil.iloc[i-1,1],1)*alfa + np.round(movil.iloc[i-1,2],1)*unoalfa
    a = movil.append({'JUGADOR':"Random",'EARNINGS':p1, 'SN':p2},ignore_index=True)
    print(a)
    tabla = a.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)
    # movil

    buf = io.BytesIO()
    plt.figure(figsize=[8,8])
    plt.grid(True)
    plt.title('JUGADORES')
    plt.plot(a['EARNINGS'],label='Tasa ingreso Referencial',marker='o')
    plt.plot(a['SN'],label='Alisamiento exponencial')
    plt.legend(loc=2)
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')
    
    return render_template('psuavizacion.html', tabla=tabla, image=plot_url)
    

@arranque.route("/pregresionlineal") #refrescar la pagina
def pregresionlineal():
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    import base64
    from matplotlib.backends.backend_agg import FigureCanvasAgg 
    
    datos = {'JUGADOR':['zombs', 'ShahZaM', 'dapr', 'SicK', 
                  'cNed', 'starxo', 'Kiles', 'nAts', 'Chronicle', 'd3ffo'],
                'DIAMANTE':[13, 13, 14, 12, 12, 12, 6, 14, 13, 12],
                'ORO':[5, 5, 3, 3, 4, 4, 4, 6, 6, 4],
                'PLATA':[1, 1, 1, 1, 2, 1, 2, 0, 0, 2],
                'TIERLIST':[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                'EARNINGS':[113.55, 113.45, 112.87, 112.65, 107.735, 104.394, 104.093, 103.829, 103.574, 102.773]}
    data = pd.DataFrame(datos).to_html(classes="table table-hover table-striped", justify="justify-all", border=2)
    

    datos1 = {'JUGADOR':['zombs', 'ShahZaM', 'dapr', 'SicK', 
                  'cNed', 'starxo', 'Kiles', 'nAts', 'Chronicle', 'd3ffo'],
                'EARNINGS':[113.55, 113.45, 112.87, 112.65, 107.735, 104.394, 104.093, 103.829, 103.574, 102.773]}
    #print(datos1)
    movil = pd.DataFrame(datos1)
    moviles = movil.transpose().to_html(classes="table table-hover table-striped", justify="justify-all", border=2)

    # mostramos los 5 primeros registros
    print(moviles)

    # calculamos para la primera media móvil MMO_3
    for i in range(0,movil.shape[0]-2):
        movil.loc[movil.index[i+2],'MMO_3'] = np.round(((movil.iloc[i,1]+movil.iloc[i+1,1]+movil.iloc[i+2,1])/3),1)
        
    # calculamos para la segunda media móvil MMO_4
    for i in range(0,movil.shape[0]-3):
        movil.loc[movil.index[i+3],'MMO_4'] = np.round(((movil.iloc[i,1]+movil.iloc[i+1,1]+movil.iloc[i+2,1]+movil.iloc[i+
    3,1])/4),1)
        
    # calculamos la proyeción final
    proyeccion = movil.iloc[7:,[1,2,3]]
    p1,p2,p3 =proyeccion.mean()

    # incorporamos al DataFrame
    a = movil.append({'JUGADOR':'Random','EARNINGS':p1, 'MMO_3':p2, 'MMO_4':p3},ignore_index=True)
    # mostramos los resultados
    a['e_MM3'] = a['EARNINGS']-a['MMO_3']
    a['e_MM4'] = a['EARNINGS']-a['MMO_4']
    a
    movillle=a.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)

    buf = io.BytesIO()
    plt.figure(figsize=[4,4])
    
    fig = plt.gcf()
    
    plt.grid(True)
    plt.plot(a['EARNINGS'],label='EARNINGS',marker='o')
    plt.plot(a['MMO_3'],label='Media Móvil Random')
    plt.plot(a['MMO_4'],label='Media Móvil Random')
    plt.legend(loc=2)
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')


    movil = pd.DataFrame(datos1)
    
    # mostramos los 5 primeros registros
    
    alfa = 0.1
    unoalfa = 1. - alfa
    for i in range(0,movil.shape[0]-1):
        movil.loc[movil.index[i+1],'SN'] = np.round(movil.iloc[i,1],1)
    for i in range(2,movil.shape[0]):
        movil.loc[movil.index[i],'SN'] = np.round(movil.iloc[i-1,1],1)*alfa + np.round(movil.iloc[i-1,2],1)*unoalfa
    i=i+1
    p1=0
    p2=np.round(movil.iloc[i-1,1],1)*alfa + np.round(movil.iloc[i-1,2],1)*unoalfa
    a = movil.append({'JUGADOR':'Random','EARNINGS':p1, 'SN':p2},ignore_index=True)
    print(a)
    tabla2 = a.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)
    
    
    a = pd.DataFrame(datos1)
    x = a.index.values
    y= a["EARNINGS"]
    # ajuste de la recta (polinomio de grado 1 f(x) = ax + b)
    p = np.polyfit(x,y,1) # 1 para lineal, 2 para polinomio ...
    p0,p1 = p
    print ("El valor de p0 = ", p0, "Valor de p1 = ", p1)


    buf = io.BytesIO()
    plt.figure(figsize=[4,4])
    
    fig = plt.gcf()
    y_ajuste = p[0]*x + p[1]
    print (y_ajuste)
    # dibujamos los datos experimentales de la recta
    p_datos =plt.plot(x,y,'b.')
    # Dibujamos la recta de ajuste
    p_ajuste = plt.plot(x,y_ajuste, 'r-')
    plt.title('Ajuste lineal por mínimos cuadrados')
    plt.xlabel('Eje x')
    plt.ylabel('Eje y')
    plt.legend(('Datos experimentales','Ajuste lineal',), loc="upper left")
    
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
   
    plot_url1 = base64.b64encode(buf.getvalue()).decode('UTF-8')

    buf = io.BytesIO()
    plt.figure(figsize=[4,4])
    p = np.polyfit(x,y,2)
    p1,p2,p3 = p
    print ("El valor de p0 = ", p0, "Valor de p1 = ", p1, " el valor de p2 = ",p2)
    
    y_ajuste = p[0]*x*x + p[1]*x + p[2]
    # dibujamos los datos experimentales de la recta
    p_datos =plt.plot(x,y,'b.')
    # Dibujamos la curva de ajuste
    p_ajuste = plt.plot(x,y_ajuste, 'r-')
    plt.title('Ajuste Polinomial por mínimos cuadrados')
    plt.xlabel('Eje x')
    plt.ylabel('Eje y')
    plt.legend(('Datos experimentales','Ajuste Polinomial',), loc="upper left")
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
   
    plot_url2 = base64.b64encode(buf.getvalue()).decode('UTF-8')
    
    n=x.size
    x1 = []
    x2 = []
    for i in [12,13]:
        y1_ajuste = p[0]*i*i + p[1]*i + p[2]
        print (f" z = {i} w = {y1_ajuste}")
        x1.append(i)
        x2.append(y1_ajuste)
        
    a["y_ajuste"]=y_ajuste

    dp = pd.DataFrame({'JUGADOR':['Random','Random'], 'EARNINGS':[0,0],'y_ajuste':x2})
    dp
    a = a.append(dp,ignore_index=True)
    print(a)
    tabla3 = a.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)
    
    buf = io.BytesIO()
    plt.figure(figsize=[4,4])
    x = a.index.values
    y_ajuste = a["y_ajuste"]
    y= a["EARNINGS"]
    p_datos =plt.plot(y,'b.')
    p_ajuste = plt.plot(x,y_ajuste, 'r-')
    plt.title('Ajuste Polinomial por mínimos cuadrados')
    plt.xlabel('Eje x')
    plt.ylabel('tasa Pasiva Referencial')
    plt.axvspan(0,9,alpha=0.3,color='y') # encajonamos los datos iniciales
    plt.legend(('Datos experimentales','Ajuste Polinomial',), loc="upper left")
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
   
    plot_url3 = base64.b64encode(buf.getvalue()).decode('UTF-8')



    return render_template("pregresionlineal.html", data=data, movillle=movillle, movil=moviles, image=plot_url, tabla2=tabla2, image2=plot_url1, image3=plot_url2, tabla3=tabla3,
    image4=plot_url3) 


@arranque.route("/pregresionexpo") #refrescar la pagina
def pregresionexpo():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    import base64
    from matplotlib.backends.backend_agg import FigureCanvasAgg 
    
    datos = {'JUGADOR':['zombs', 'ShahZaM', 'dapr', 'SicK', 
                  'cNed', 'starxo', 'Kiles', 'nAts', 'Chronicle', 'd3ffo'],
                'DIAMANTE':[13, 13, 14, 12, 12, 12, 6, 14, 13, 12],
                'ORO':[5, 5, 3, 3, 4, 4, 4, 6, 6, 4],
                'PLATA':[1, 1, 1, 1, 2, 1, 2, 0, 0, 2],
                'TIERLIST':[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                'EARNINGS':[113.55, 113.45, 112.87, 112.65, 107.735, 104.394, 104.093, 103.829, 103.574, 102.773]}
    data = pd.DataFrame(datos).to_html(classes="table table-hover table-striped", justify="justify-all", border=2)
    

    datos1 = {'JUGADOR':['zombs', 'ShahZaM', 'dapr', 'SicK', 
                  'cNed', 'starxo', 'Kiles', 'nAts', 'Chronicle', 'd3ffo'],
                'EARNINGS':[113.55, 113.45, 112.87, 112.65, 107.735, 104.394, 104.093, 103.829, 103.574, 102.773]}
    #print(datos1)
    movil = pd.DataFrame(datos1)
    moviles = movil.transpose().to_html(classes="table table-hover table-striped", justify="justify-all", border=2)

    # mostramos los 5 primeros registros
    print(moviles)

    # calculamos para la primera media móvil MMO_3
    for i in range(0,movil.shape[0]-2):
        movil.loc[movil.index[i+2],'MMO_3'] = np.round(((movil.iloc[i,1]+movil.iloc[i+1,1]+movil.iloc[i+2,1])/3),1)
        
    # calculamos para la segunda media móvil MMO_4
    for i in range(0,movil.shape[0]-3):
        movil.loc[movil.index[i+3],'MMO_4'] = np.round(((movil.iloc[i,1]+movil.iloc[i+1,1]+movil.iloc[i+2,1]+movil.iloc[i+
    3,1])/4),1)
        
    # calculamos la proyeción final
    proyeccion = movil.iloc[7:,[1,2,3]]
    p1,p2,p3 =proyeccion.mean()

    # incorporamos al DataFrame
    a = movil.append({'JUGADOR':'Random','EARNINGS':p1, 'MMO_3':p2, 'MMO_4':p3},ignore_index=True)
    # mostramos los resultados
    a['e_MM3'] = a['EARNINGS']-a['MMO_3']
    a['e_MM4'] = a['EARNINGS']-a['MMO_4']
    a
    movillle=a.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)

    buf = io.BytesIO()
    plt.figure(figsize=[4,4])
    
    fig = plt.gcf()
    
    plt.grid(True)
    plt.plot(a['EARNINGS'],label='EARNINGS',marker='o')
    plt.plot(a['MMO_3'],label='Media Móvil Random')
    plt.plot(a['MMO_4'],label='Media Móvil Random')
    plt.legend(loc=2)
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')


    movil = pd.DataFrame(datos1)
    
    # mostramos los 5 primeros registros
    
    alfa = 0.1
    unoalfa = 1. - alfa
    for i in range(0,movil.shape[0]-1):
        movil.loc[movil.index[i+1],'SN'] = np.round(movil.iloc[i,1],1)
    for i in range(2,movil.shape[0]):
        movil.loc[movil.index[i],'SN'] = np.round(movil.iloc[i-1,1],1)*alfa + np.round(movil.iloc[i-1,2],1)*unoalfa
    i=i+1
    p1=0
    p2=np.round(movil.iloc[i-1,1],1)*alfa + np.round(movil.iloc[i-1,2],1)*unoalfa
    a = movil.append({'JUGADOR':'Random','EARNINGS':p1, 'SN':p2},ignore_index=True)
    print(a)
    tabla2 = a.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)
    
    
    a = pd.DataFrame(datos1)
    x = a.index.values
    y= a["EARNINGS"]
    # ajuste de la recta (polinomio de grado 1 f(x) = ax + b)
    p = np.polyfit(x,y,1) # 1 para lineal, 2 para polinomio ...
    p0,p1 = p
    print ("El valor de p0 = ", p0, "Valor de p1 = ", p1)


    buf = io.BytesIO()
    plt.figure(figsize=[4,4])
    
    fig = plt.gcf()
    y_ajuste = p[0]*x + p[1]
    print (y_ajuste)
    # dibujamos los datos experimentales de la recta
    p_datos =plt.plot(x,y,'b.')
    # Dibujamos la recta de ajuste
    p_ajuste = plt.plot(x,y_ajuste, 'r-')
    plt.title('Ajuste lineal por mínimos cuadrados')
    plt.xlabel('Eje x')
    plt.ylabel('Eje y')
    plt.legend(('Datos experimentales','Ajuste lineal',), loc="upper left")
    
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
   
    plot_url1 = base64.b64encode(buf.getvalue()).decode('UTF-8')

    buf = io.BytesIO()
    plt.figure(figsize=[4,4])
    p = np.polyfit(x,y,2)
    p1,p2,p3 = p
    print ("El valor de p0 = ", p0, "Valor de p1 = ", p1, " el valor de p2 = ",p2)
    
    y_ajuste = p[0]*x*x + p[1]*x + p[2]
    # dibujamos los datos experimentales de la recta
    p_datos =plt.plot(x,y,'b.')
    # Dibujamos la curva de ajuste
    p_ajuste = plt.plot(x,y_ajuste, 'r-')
    plt.title('Ajuste Polinomial por mínimos cuadrados')
    plt.xlabel('Eje x')
    plt.ylabel('Eje y')
    plt.legend(('Datos experimentales','Ajuste Polinomial',), loc="upper left")
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
   
    plot_url2 = base64.b64encode(buf.getvalue()).decode('UTF-8')
    
    n=x.size
    x1 = []
    x2 = []
    for i in [12,13]:
        y1_ajuste = p[0]*i*i + p[1]*i + p[2]
        print (f" z = {i} w = {y1_ajuste}")
        x1.append(i)
        x2.append(y1_ajuste)
        
    a["y_ajuste"]=y_ajuste

    dp = pd.DataFrame({'JUGADOR':['Random','Random'], 'EARNINGS':[0,0],'y_ajuste':x2})
    dp
    a = a.append(dp,ignore_index=True)
    print(a)
    tabla3 = a.to_html(classes="table table-hover table-striped", justify="justify-all", border=2)
    
    buf = io.BytesIO()
    plt.figure(figsize=[4,4])
    x = a.index.values
    y_ajuste = a["y_ajuste"]
    y= a["EARNINGS"]
    p_datos =plt.plot(y,'b.')
    p_ajuste = plt.plot(x,y_ajuste, 'r-')
    plt.title('Ajuste Polinomial por mínimos cuadrados')
    plt.xlabel('Eje x')
    plt.ylabel('tasa Pasiva Referencial')
    plt.axvspan(0,9,alpha=0.3,color='y') # encajonamos los datos iniciales
    plt.legend(('Datos experimentales','Ajuste Polinomial',), loc="upper left")
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
   
    plot_url3 = base64.b64encode(buf.getvalue()).decode('UTF-8')



    return render_template("pregresionexpo.html", data=data, movillle=movillle, movil=moviles, image=plot_url, tabla2=tabla2, image2=plot_url1, image3=plot_url2, tabla3=tabla3,
    image4=plot_url3)


@arranque.route("/modamm", methods=["GET"])
def modamm():
    return render_template("modamm.html")


@arranque.route("/modamm", methods=["POST"])
def rendermodamm():
    file = request.files['file'].read()
    tipoArch= request.form.get("tipoarchivo")
    columna = request.form.get("nombreColumna")

    # importamos la libreria Pandas, matplotlib y numpy que van a ser de mucha utilidad para poder hacer gráficos
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO
    import base64
    from pandas import DataFrame

    # leemos los datos de la tabla del directorio Data de trabajo
    if tipoArch=='1':
        
        datos = pd.read_excel(file)
        
        
    elif tipoArch=='2':
        datos = pd.read_csv(io.StringIO(file.decode('utf-8')))
        
    elif tipoArch=='3':
        datos = pd.read_json(file)

    # Presentamos los datos en un DataFrame de Pandas
    datos

    # Preparando para el grafico para la columna TOTAL PACIENTES
    buf = io.BytesIO()
    x = datos[columna]
    plt.figure(figsize=(10, 5))
    plt.hist(x, bins=8, color='blue')
    plt.axvline(x.mean(), color='red', label='Media')
    plt.axvline(x.median(), color='yellow', label='Mediana')
    plt.axvline(x.mode()[0], color='green', label='Moda')
    plt.xlabel('Total de datos')
    plt.ylabel('Frecuencia')
    plt.legend()

    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    media = datos[columna].mean()
    moda = datos[columna].mode()
    mediana = datos[columna].median()

    df = pd.DataFrame(columns=('Media', 'Moda', 'Mediana'))
    df.loc[len(df)] = [media, moda, mediana]
    df
    data = df.to_html(classes="table table-striped",
                      justify="justify-all", border=0)

    # Tomamos los datos de las columnas
    df2 = datos[[columna]].describe()
    # describe(), nos presenta directamente la media, desviación standar, el valor mínimo, valor máximo, el 1er cuartil, 2do Cuartil, 3er Cuartil
    data2 = df2.transpose().to_html(classes="table table-hover table-striped",
                        justify="justify-all", border=0)

    return render_template('modamm.html', data=data, data2=data2, image=plot_url)


@arranque.route("/modysimu") #refrescar la pagina
def modysimu():
    return render_template("modysimu.html")



@arranque.route("/lineaespera",  methods=["GET"]) #refrescar la pagina
def lineaespera():
    return render_template("lineaespera.html")


@arranque.route("/lineaespera", methods=["POST"]) #refrescar la pagina
def renderlineaespera():
    landa = request.form.get("landa", type=float)
    nu = request.form.get("miu", type=float)
    num = request.form.get("numeroIteraciones", type=int)
    x0 = request.form.get("semilla", type=int)
    a = request.form.get("multiplicador", type=int)
    c = request.form.get("incremento", type=int)
    m = request.form.get("modulo", type=int)

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import ExcelWriter
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO
    import base64
    import math
    import random
    from pandas import DataFrame

    #La probabilidad de hallar el sistema ocupado o utilización del sistema:
    p = []
    p = landa/nu
    #La probabilidad de que no haya unidades en el sistema este vacía u ocioso :
    Po = []
    Po = 1.0 - (landa/nu)
    #Longitud esperada en cola, promedio de unidades en la línea de espera:
    Lq = []
    Lq = landa*landa / (nu * (nu - landa))
    #/ (nu * (nu - landa))
    # Número esperado de clientes en el sistema(cola y servicio) :
    L = []
    L = landa / (nu - landa)
    #El tiempo promedio que una unidad pasa en el sistema:
    W = []
    W = 1 / (nu - landa)
    #Tiempo de espera en cola:
    Wq = []
    Wq = W - (1.0 / nu)
    print(Wq)
    #La probabilidad de que haya n unidades en el sistema:
    n = 1
    Pn = []
    Pn = (landa/nu)*n*Po

    df = pd.DataFrame(columns=('lambda', 'nu', 'p',
                      'Po', 'Lq', 'L', 'W', 'Wq', 'Pn'))
    df.loc[len(df)] = [landa, nu, p, Po, Lq, L, W, Wq, Pn]
    df

    data = df.to_html(classes="table table-hover table-striped",
                      justify="justify-all", border=0)

    i = 0
    # Landa y nu ya definidos
    # Atributos del DataFrame
    """
    ALL # ALEATORIO DE LLEGADA DE CLIENTES
    ASE # ALEATORIO DE SERVICIO
    TILL TIEMPO ENTRE LLEGADA
    TISE TIEMPO DE SERVICIO
    TIRLL TIEMPO REAL DE LLEGADA
    TIISE TIEMPO DE INICIO DE SERVICIO
    TIFSE TIEMPO FINAL DE SERVICIO
    TIESP TIEMPO DE ESPERA
    TIESA TIEMPO DE SALIDA
    numClientes NUMERO DE CLIENTES
    dfLE DATAFRAME DE LA LINEA DE ESPERA
    """
    numClientes = num
    i = 0
    indice = ['ALL', 'ASE', 'TILL', 'TISE',
              'TIRLL', 'TIISE', 'TIFSE', 'TIESP', 'TIESA']
    Clientes = np.arange(numClientes)
    dfLE = pd.DataFrame(index=Clientes, columns=indice).fillna(0.000)

    #np.random.seed(num)

    # n, m, a, x0 = 20, 1000, 747, 123
    x = [1] * num
    r = [0.1] * num
    for j in range(0, num):
        x[j] = ((a*x0)+c) % m
        x0 = x[j]
        #r[j] = x0 / m
        dfLE['ALL'][j] = x0 / m
        #dfLE['ASE'][j] = x0 / m

    # n, m, a, x0 = 20, 1000, 747, 123
    x = [1] * num
    r = [0.1] * num
    for j in range(0, num):
        x[j] = (a*x0) % m
        x0 = x[j]
        #r[j] = x0 / m
        #dfLE['ALL'][j] = x0 / m
        dfLE['ASE'][j] = x0 / m

    for i in Clientes:
        if i == 0:
            #dfLE['ASE'][i] = random.random()
            dfLE['TILL'][i] = -landa*np.log(dfLE['ALL'][i])
            dfLE['TISE'][i] = -nu*np.log(dfLE['ASE'][i])
            dfLE['TIRLL'][i] = dfLE['TILL'][i]
            dfLE['TIISE'][i] = dfLE['TIRLL'][i]
            dfLE['TIFSE'][i] = dfLE['TIISE'][i] + dfLE['TISE'][i]
            dfLE['TIESA'][i] = dfLE['TIESP'][i] + dfLE['TISE'][i]
        else:
            #dfLE['ASE'][i] = random.random()
            dfLE['TILL'][i] = -landa*np.log(dfLE['ALL'][i])
            dfLE['TISE'][i] = -nu*np.log(dfLE['ASE'][i])
            dfLE['TIRLL'][i] = dfLE['TILL'][i] + dfLE['TIRLL'][i-1]
            dfLE['TIISE'][i] = max(dfLE['TIRLL'][i], dfLE['TIFSE'][i-1])
            dfLE['TIFSE'][i] = dfLE['TIISE'][i] + dfLE['TISE'][i]
            dfLE['TIESP'][i] = dfLE['TIISE'][i] - dfLE['TIRLL'][i]
            dfLE['TIESA'][i] = dfLE['TIESP'][i] + dfLE['TISE'][i]
    nuevas_columnas = pd.core.indexes.base.Index(["A_LLEGADA", "A_SERVICIO", "TIE_LLEGADA", "TIE_SERVICIO",
                                                  "TIE_EXACTO_LLEGADA", "TIE_INI_SERVICIO", "TIE_FIN_SERVICIO",
                                                  "TIE_ESPERA", "TIE_EN_SISTEMA"])

    dfLE.columns = nuevas_columnas
    dfLE

    # Graficamos los numeros generados
    buf = io.BytesIO()
    plt.plot(dfLE['A_LLEGADA'], label='A_LLEGADA')
    plt.plot(dfLE['A_SERVICIO'], label='A_SERVICIO')
    plt.plot(dfLE['TIE_LLEGADA'], label='TIE_LLEGADA')
    plt.plot(dfLE['TIE_SERVICIO'], label='TIE_SERVICIO')
    plt.plot(dfLE['TIE_EXACTO_LLEGADA'], label='TIE_EXACTO_LLEGADA')
    plt.plot(dfLE['TIE_INI_SERVICIO'], label='TIE_INI_SERVICIO')
    plt.plot(dfLE['TIE_FIN_SERVICIO'], label='TIE_FIN_SERVICIO')
    plt.plot(dfLE['TIE_ESPERA'], label='TIE_ESPERA')
    plt.plot(dfLE['TIE_EN_SISTEMA'], label='TIE_EN_SISTEMA')
    plt.legend(loc=2)
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    kl = dfLE["TIE_ESPERA"]
    jl = dfLE["TIE_EN_SISTEMA"]
    ll = dfLE["A_LLEGADA"]
    pl = dfLE["A_SERVICIO"]
    ml = dfLE["TIE_INI_SERVICIO"]
    nl = dfLE["TIE_FIN_SERVICIO"]

    klsuma = sum(kl)
    klpro = (klsuma/num)
    jlsuma = sum(jl)
    jlpro = jlsuma/num
    dfLE.loc[num] = ['-', '-', '-', '-', '-', '-', 'SUMA', klsuma, jlsuma]
    dfLE.loc[(num+1)] = ['-', '-', '-', '-',
                         '-', '-', 'PROMEDIO', klpro, jlpro]

    dfLE

    data2 = dfLE.to_html(
        classes="table table-hover table-striped", justify="justify-all", border=0)

    """ writer = ExcelWriter("static/file/data.xlsx")
    dfLE.to_excel(writer, index=False)
    writer.save()

    dfLE.to_csv("static/file/data.csv", index=False) """

    dfLE2 = pd.DataFrame(dfLE.describe())
    data3 = dfLE2.to_html(
        classes="table table-hover table-striped", justify="justify-all", border=0)

    return render_template('lineaespera.html', data=data, data2=data2, data3=data3, image=plot_url)



@arranque.route("/sisinventario", methods=["GET"]) #refrescar la pagina
def sisinventario():
    return render_template("sisinventario.html")

@arranque.route("/sisinventario", methods=["POST"]) 
def rendersisinventario():

    D = request.form.get("demanda", type=float)
    Co = request.form.get("costoOrdenar", type=float)
    Ch = request.form.get("costoMantenimiento", type=float)
    P = request.form.get("costoProducto", type=float)
    Tespera = request.form.get("tiempoEspera", type=float)
    DiasAno = request.form.get("diasAno", type=int)
    num = request.form.get("numeroIteraciones", type=int)

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import ExcelWriter
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO
    import base64
    import math
    from math import sqrt
    from pandas import DataFrame

    Q = round(sqrt(((2*Co*D)/Ch)), 2)
    N = round(D / Q, 2)
    R = round((D / DiasAno) * Tespera, 2)
    T = round(DiasAno / N, 2)
    CoT = N * Co
    ChT = round(Q / 2 * Ch, 2)
    MOQ = round(CoT + ChT, 2)
    CTT = round(P * D + MOQ, 2)

    df = pd.DataFrame(columns=('Q', 'N', 'R', 'T', 'CoT', 'ChT', 'MOQ', 'CTT'))
    df.loc[len(df)] = [Q, N, R, T, CoT, ChT, MOQ, CTT]
    df

    data = df.to_html(classes="table table-striped",
                      justify="justify-all", border=0)

    # Programa para generar el gráfico de costo mínimo
    indice = ['Q', 'Costo_ordenar', 'Costo_Mantenimiento',
        'Costo_total', 'Diferencia_Costo_Total']
    # Generamos una lista ordenada de valores de Q

    periodo = np.arange(0, num)

    def genera_lista(Q):
        n = num
        Q_Lista = []
        i = 1
        Qi = Q
        Q_Lista.append(Qi)
        for i in range(1, 9):
            Qi = Qi - 60
            Q_Lista.append(Qi)

        Qi = Q
        for i in range(9, n):
            Qi = Qi + 60
            Q_Lista.append(Qi)
        return Q_Lista

    Lista = genera_lista(Q)
    Lista.sort()

    dfQ = DataFrame(index=periodo, columns=indice).fillna(0)

    dfQ['Q'] = Lista
    #dfQ

    for period in periodo:
        dfQ['Costo_ordenar'][period] = D * Co / dfQ['Q'][period]
        dfQ['Costo_Mantenimiento'][period] = dfQ['Q'][period] * Ch / 2
        dfQ['Costo_total'][period] = dfQ['Costo_ordenar'][period] + \
            dfQ['Costo_Mantenimiento'][period]
        dfQ['Diferencia_Costo_Total'][period] = dfQ['Costo_total'][period] - MOQ
    dfQ

    # Graficamos los numeros generados
    buf = io.BytesIO()
    plt.plot(dfQ['Costo_ordenar'], label='Costo_ordenar')
    plt.plot(dfQ['Costo_Mantenimiento'], label='Costo_Mantenimiento')
    plt.plot(dfQ['Costo_total'], label='Costo_total')
    plt.legend()

    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    data2 = dfQ.to_html(classes="table table-hover table-striped",
                        justify="justify-all", border=0)

    def make_data(product, policy, periods):
        periods += 1
        # Create zero-filled Dataframe
        period_lst = np.arange(periods)  # index
        header = ['INV_INICIAL', 'INV_NETO_INICIAL', 'DEMANDA', 'INV_FINAL', 'INV_FINAL_NETO',
            'VENTAS_PERDIDAS', 'INV_PROMEDIO', 'CANT_ORDENAR', 'TIEMPO_LLEGADA']
        df = DataFrame(index=period_lst, columns=header).fillna(0)
        # Create a list that will store each period order
        order_l = [Order(quantity=0, lead_time=0)
                   for x in range(periods)]
                       # Fill DataFrame
        for period in period_lst:
            if period == 0:
                df['INV_INICIAL'][period] = product.initial_inventory
                df['INV_NETO_INICIAL'][period] = product.initial_inventory
                df['INV_FINAL'][period] = product.initial_inventory
                df['INV_FINAL_NETO'][period] = product.initial_inventory
            if period >= 1:
                df['INV_INICIAL'][period] = df['INV_FINAL'][period - 1] + \
                    order_l[period - 1].quantity
                df['INV_NETO_INICIAL'][period] = df['INV_FINAL_NETO'][period -
                    1] + pending_order(order_l, period)
                #demand = int(product.demand())
                demand = D
                # We can't have negative demand
                if demand > 0:
                    df['DEMANDA'][period] = demand
                else:
                    df['DEMANDA'][period] = 0
                # We can't have negative INV_INICIAL
                if df['INV_INICIAL'][period] - df['DEMANDA'][period] < 0:
                    df['INV_FINAL'][period] = 0
                else:
                    df['INV_FINAL'][period] = df['INV_INICIAL'][period] - \
                        df['DEMANDA'][period]
                order_l[period].quantity, order_l[period].lead_time = placeorder(
                    product, df['INV_FINAL'][period], policy, period)
                df['INV_FINAL_NETO'][period] = df['INV_NETO_INICIAL'][period] - \
                    df['DEMANDA'][period]
                if df['INV_FINAL_NETO'][period] < 0:
                    df['VENTAS_PERDIDAS'][period] = abs(
                        df['INV_FINAL_NETO'][period])
                    df['INV_FINAL_NETO'][period] = 0
                else:
                    df['VENTAS_PERDIDAS'][period] = 0
                df['INV_PROMEDIO'][period] = (
                    df['INV_NETO_INICIAL'][period] + df['INV_FINAL_NETO'][period]) / 2.0
                df['CANT_ORDENAR'][period] = order_l[period].quantity
                df['TIEMPO_LLEGADA'][period] = order_l[period].lead_time
        return df

    def pending_order(order_l, period):
        """Return the order that arrives in actual period"""
        indices = [i for i, order in enumerate(order_l) if order.quantity]
        sum = 0
        for i in indices:
            if period-(i + order_l[i].lead_time+1) == 0:
                sum += order_l[i].quantity
        return sum

    def demanda(self):
            if self.demand_dist == "Constant":
                return self.demand_p1
            elif self.demand_dist == "Normal":
                return make_distribution(
                    np.random.normal,
                    self.demand_p1,
                    self.demand_p2)()
            elif self.demand_dist == "Triangular":
                return make_distribution(
                    np.random_triangular,
                    self.demand_p1,
                    self.demand_p2,
                    self.demand_p3)()
    def lead_time(self):
            if self.leadtime_dist == "Constant":
                return self.leadtime_p1
            elif self.leadtime_dist == "Normal":
                return make_distribution(
                    np.random.normal,
                    self.leadtime_p1,
                    self.leadtime_p2)()
            elif self.leadtime_dist == "Triangular":
                return make_distribution(
                    np.random_triangular,
                    self.leadtime_p1,
                    self.leadtime_p2,
                    self.leadtime_p3)()

    def __repr__(self):
           return '<Product %r>' % self.name

    def placeorder(product, final_inv_pos, policy, period):
        #lead_time = int(product.lead_time())
        lead_time = Tespera
        # Qs = if we hit the reorder point s, order Q units
        if policy['method'] == 'Qs' and \
                final_inv_pos <= policy['param2']:
            return policy['param1'], lead_time
        # RS = if we hit the review period R and the reorder point S, order: (S -
        # final inventory pos)
        elif policy['method'] == 'RS' and \
            period % policy['param1'] == 0 and \
                final_inv_pos <= policy['param2']:
            return policy['param2'] - final_inv_pos, lead_time
        else:
            return 0, 0

    politica = {'method': "Qs", 'param1': 50,'param2': 20}

    class Order(object):
        """Object that stores basic data of an order"""

        def __init__(self, quantity, lead_time):
            self.quantity = quantity
            self.lead_time = lead_time

    class product(object):
        def __init__ (self, name,price,order_cost,initial_inventory,demand_dist,demand_p1,demand_p2,demand_p3,leadtime_dist,leadtime_p1,leadtime_p2,leadtime_p3):
            self.name = name
            self.price = price
            self.order_cost = order_cost
            self.initial_inventory = initial_inventory
            self.demand_dist = demand_dist
            self.demand_p1 = demand_p1
            self.demand_p2 = demand_p2
            self.demand_p3 = demand_p3
            self.leadtime_dist = leadtime_dist
            self.leadtime_p1 = leadtime_p1
            self.leadtime_p2 = leadtime_p2
            self.leadtime_p3 = leadtime_p3
    producto = product("Mesa", 18.0, 20.0,100,"Constant",80.0,0.0,0.0,"Constant",1.0,0.0,0.0)

    num = num - 1
    df = make_data(producto, politica, num)
    df

    data3 = df.to_html(classes="table table-hover table-striped",
                       justify="justify-all", border=0)

    """ writer = ExcelWriter("static/file/data.xlsx")
    df.to_excel(writer, index=False)
    writer.save()

    df.to_csv("static/file/data.csv", index=False) """

    return render_template('sisinventario.html', data=data, data2=data2, data3=data3, image=plot_url)



if __name__ == '__main__':
    arranque.run(port=5000,debug=True)