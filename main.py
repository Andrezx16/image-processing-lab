"""
SEMAFORO INTELIGENTE POR VISION ARTIFICIAL
Camara -> gris -> histograma -> analisis de iluminacion -> filtro lineal
       -> umbralizacion -> vecindad 3x3 -> CONTEO DE PIXELES -> decision -> mensaje

Teclas:  q / Esc / X = salir | n = simular ruido (nada > sal > pimienta > ambos)
         b = calibrar: 1a vez guarda el FONDO vacio, 2a vez guarda el OBJETO
             (calcula el % que ocupa), 3a vez reinicia | r = borrar todo
         v = verificar filtros manuales contra cv2.filter2D | s = guardar captura
"""
import random
import cv2
import numpy as np

# ---------------- UMBRALES (ajustables) ----------------
OSC, CLA = 70, 180            # limites de zona oscura / clara del histograma
MEDIA_MIN, MEDIA_MAX = 80, 190
PCT_OSC_MAX, PCT_CLA_MAX = 50, 50
CONTRASTE_MIN = 25            # desviacion estandar minima
RUIDO_DIF, RUIDO_MAX = 40, 3  # diferencia (niveles) y % maximo de pixeles ruidosos
DIF_FONDO = 30                # diferencia contra el fondo para contar un pixel como objeto
AREA_MOUSE = 30               # % del mouse por defecto (si no se calibra con b)
TOLERANCIA = 0.20             # margen relativo: se acepta el % de referencia +-20 %
PRESENCIA = 2                 # % minimo para decir que "hay algo" en la escena
OBJETO = "MOUSE"              # cambiar a "TARJETA" si se usa la tarjeta

VENTANA = "SISTEMA DE INSPECCION"
VERDE, AMARILLO, ROJO = (0, 160, 0), (0, 200, 230), (0, 0, 200)

# Filtro lineal de suavizado (el mismo de tu codigo), normalizado
H5 = np.array([[0, 1, 2, 1, 0],
               [1, 3, 5, 3, 1],
               [2, 5, 9, 5, 2],
               [1, 3, 5, 3, 1],
               [0, 1, 2, 1, 0]], dtype=np.float32)
H5 /= H5.sum()
H3 = np.ones((3, 3), dtype=np.float32) / 9


# ================= OPERACION DE PIXEL =================
def negativo(img):
    """g(x,y) = 255 - f(x,y)"""
    return (255 - img.astype(np.int16)).astype(np.uint8)


# ================= HISTOGRAMA =================
def histograma_lento(img):
    """Version con ciclos (la de tu codigo)."""
    h = np.zeros(256)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            h[img[y, x]] += 1
    return h


def histograma(img):
    """Mismo resultado, rapido (para tiempo real)."""
    return np.bincount(img.ravel(), minlength=256).astype(np.float64)


# ================= VECINDAD / FILTRO LINEAL (manual) =================
def same_lento(img, H, i, j):
    """Tu version con 4 ciclos: correcta pero demasiado lenta para video."""
    f, c = img.shape
    fh, ch = H.shape
    I = np.zeros((f + i + fh - 1 - i, c + j + ch - 1 - j))
    I[i:i + f, j:j + c] = img
    res = np.zeros((f, c))
    for px in range(f):
        for py in range(c):
            s = 0
            for x in range(fh):
                for y in range(ch):
                    s += I[px + x, py + y] * H[x, y]
            res[px, py] = s
    return res


def same(img, H, i, j):
    """Misma operacion, pero los dos ciclos de la imagen se reemplazan por
    sumas de la imagen desplazada (solo se recorre el kernel)."""
    f, c = img.shape
    fh, ch = H.shape
    I = np.zeros((f + fh - 1, c + ch - 1), np.float32)
    I[i:i + f, j:j + c] = img
    res = np.zeros((f, c), np.float32)
    for x in range(fh):
        for y in range(ch):
            res += H[x, y] * I[x:x + f, y:y + c]
    return res


def vecindad_3x3(img):
    """g(x,y) = 1/9 * suma_{i,j=-1..1} f(x+i, y+j)"""
    return same(img.astype(np.float32), H3, 1, 1)


# ================= RUIDO SAL Y PIMIENTA (tu codigo) =================
def ruido(img, p, sop):
    I = img.copy().astype("uint8")
    f, c = I.shape
    valor = 255 if sop == "sal" else 0
    num_bloques = int(f * c * p / 100) // 6
    for _ in range(num_bloques):
        alto, ancho = (3, 2) if random.random() < 0.5 else (2, 3)
        y = random.randint(0, f - alto)
        x = random.randint(0, c - ancho)
        I[y:y + alto, x:x + ancho] = valor
    return I


# ================= UMBRALIZACION Y CONTEO DE PIXELES =================
def otsu(h):
    """Umbral que mejor separa dos grupos del histograma (varianza entre clases)."""
    p = h / h.sum()
    k = np.arange(256)
    w0 = np.cumsum(p)
    mu = np.cumsum(p * k)
    den = w0 * (1 - w0)
    sb = np.zeros(256)
    ok = den > 1e-12                      # evita dividir entre cero (imagen de un solo tono)
    sb[ok] = (mu[-1] * w0[ok] - mu[ok]) ** 2 / den[ok]
    return int(np.argmax(sb))


def detectar_objeto(filtrada, fondo=None, pct_ref=None):
    """Devuelve la mascara limpia y los datos del objeto, usando conteo de pixeles."""
    f = filtrada.astype(np.float32)
    if fondo is not None:                       # modo referencia: pixel - fondo
        mask = np.abs(f - fondo.astype(np.float32)) > DIF_FONDO
        modo = "calibrado" if pct_ref is not None else "fondo (b: objeto)"
    else:                                       # modo Otsu: el grupo minoritario es el objeto
        t = otsu(histograma(filtrada))
        mask = f <= t
        if mask.mean() > 0.5:
            mask = ~mask
        modo = f"Otsu (t={t})"
    # limpieza por vecindad 3x3: se queda el pixel si >50% de su vecindad es objeto
    mask = vecindad_3x3(mask.astype(np.float32)) > 0.5
    mask[:3, :] = mask[-3:, :] = False          # bordes afectados por el relleno de ceros
    mask[:, :3] = mask[:, -3:] = False

    area = int(mask.sum())                      # CONTEO DE PIXELES
    pct = area / mask.size * 100
    ref = pct_ref if pct_ref is not None else AREA_MOUSE   # % esperado del objeto
    lo, hi = ref * (1 - TOLERANCIA), ref * (1 + TOLERANCIA)
    d = {"mask": mask, "area": area, "pct": pct, "modo": modo, "lo": lo, "hi": hi,
         "hay_objeto": pct >= PRESENCIA, "detectado": lo <= pct <= hi,
         "bbox": None, "centro": None}
    if d["hay_objeto"]:
        cols, filas = mask.sum(axis=0), mask.sum(axis=1)   # proyecciones (pixeles por columna / fila)
        xs, ys = np.where(cols >= 2)[0], np.where(filas >= 2)[0]
        if len(xs) and len(ys):
            d["bbox"] = (int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1]))
        py, px = np.nonzero(mask)
        d["centro"] = (int(px.mean()), int(py.mean()))
    return d


# ================= ANALISIS =================
def analizar(gris, fondo=None, pct_ref=None):
    r = {}
    # 1) Histograma e iluminacion
    h = histograma(gris)
    total = h.sum()
    niveles = np.arange(256)
    r["h"] = h
    r["media"] = (h * niveles).sum() / total
    r["contraste"] = np.sqrt((h * (niveles - r["media"]) ** 2).sum() / total)
    r["osc"] = h[:OSC].sum() / total * 100
    r["med"] = h[OSC:CLA].sum() / total * 100
    r["cla"] = h[CLA:].sum() / total * 100

    # 2) Filtro lineal (suavizado 5x5) y estimacion de ruido por vecindad:
    #    un pixel que difiere mucho de su entorno suavizado es ruido impulsivo
    suave = same(gris.astype(np.float32), H5, 2, 2)
    dif = np.abs(gris.astype(np.float32) - suave)[2:-2, 2:-2]
    r["ruido"] = (dif > RUIDO_DIF).mean() * 100
    filtrada = np.clip(suave, 0, 255).astype(np.uint8)

    # 3) Umbralizacion + vecindad 3x3 + conteo de pixeles del objeto
    r.update(detectar_objeto(filtrada, fondo, pct_ref))

    # 4) Operacion de pixel: negativo (para visualizacion)
    r["filtrada"], r["negativo"] = filtrada, negativo(filtrada)

    # 5) Decision (condicionales): cada condicion se evalua por separado
    if r["media"] > MEDIA_MAX or r["cla"] > PCT_CLA_MAX:
        r["ilum"] = ("SOBREEXPUESTA", ROJO)
    elif r["osc"] > 10 and r["cla"] > 10 and r["med"] < 20:
        r["ilum"] = ("CONTRALUZ", AMARILLO)          # regla mas estricta: va antes
    elif r["media"] < MEDIA_MIN or r["osc"] > PCT_OSC_MAX:
        r["ilum"] = ("INSUFICIENTE", AMARILLO)
    else:
        r["ilum"] = ("ADECUADA", VERDE)
    r["cont"] = ("BAJO", AMARILLO) if r["contraste"] < CONTRASTE_MIN else ("SUFICIENTE", VERDE)
    r["rui"] = ("ALTO", ROJO) if r["ruido"] > RUIDO_MAX else ("BAJO", VERDE)
    if r["detectado"]:
        r["obj"] = ("DETECTADO", VERDE)
    elif r["hay_objeto"]:
        r["obj"] = (f"NO ES {OBJETO}", ROJO)
    else:
        r["obj"] = ("NO DETECTADO", ROJO)

    # Mensaje de las CONDICIONES DE LA IMAGEN (primera que falle, si no, aprobada)
    if r["ilum"][0] == "SOBREEXPUESTA":
        cond = (ROJO, "ALERTA", "Imagen sobreexpuesta. Inspeccion no confiable.")
    elif r["ilum"][0] == "CONTRALUZ":
        cond = (AMARILLO, "ADVERTENCIA", "Contraluz. Cambie la iluminacion.")
    elif r["ilum"][0] == "INSUFICIENTE":
        cond = (AMARILLO, "ADVERTENCIA", "Iluminacion insuficiente. Mejore la captura.")
    elif r["cont"][0] == "BAJO":
        cond = (AMARILLO, "ADVERTENCIA", "Bajo contraste en la imagen.")
    elif r["rui"][0] == "ALTO":
        cond = (ROJO, "ALERTA", "Ruido significativo. Se recomienda filtrar.")
    else:
        cond = (VERDE, "INSPECCION APROBADA", "Iluminacion adecuada. Contraste suficiente.")
    r["cond"] = cond

    # Mensaje del OBJETO
    if r["detectado"]:
        r["objmsg"] = (VERDE, f"{OBJETO} DETECTADO", "Inspeccion correcta.")
    elif r["hay_objeto"]:
        r["objmsg"] = (ROJO, f"NO ES {OBJETO}",
                       f"Objeto de {r['pct']:.0f} % (esperado {r['lo']:.0f}-{r['hi']:.0f} %).")
    else:
        r["objmsg"] = (ROJO, f"{OBJETO} NO DETECTADO", "Revise la imagen.")

    # Resultado final: el peor de los dos semaforos
    colores = (cond[0], r["objmsg"][0])
    if ROJO in colores:
        r["color"], r["titulo"] = ROJO, "ALERTA"
    elif AMARILLO in colores:
        r["color"], r["titulo"] = AMARILLO, "ADVERTENCIA"
    else:
        r["color"], r["titulo"] = VERDE, "INSPECCION APROBADA"
    r["msg"] = f"{cond[2]} | {r['objmsg'][1]}"
    return r


# ================= VISUALIZACION =================
def texto(img, s, pos, esc=0.55, col=(255, 255, 255), g=1):
    cv2.putText(img, s, pos, cv2.FONT_HERSHEY_SIMPLEX, esc, col, g, cv2.LINE_AA)


def bloque(canvas, x0, x1, etiqueta, color, titulo, msg):
    """Cuadro de semaforo con etiqueta, titulo y mensaje."""
    canvas[0:75, x0:x1] = color
    tc = (0, 0, 0) if color == AMARILLO else (255, 255, 255)
    texto(canvas, etiqueta, (x0 + 12, 18), 0.45, tc)
    texto(canvas, titulo, (x0 + 12, 46), 0.8, tc, 2)
    texto(canvas, msg, (x0 + 12, 67), 0.5, tc)


def dibujar(vista, r, modo_ruido):
    TOP = 110                                   # alto de la zona de mensajes
    vista = vista.copy()
    if r["bbox"] is not None:                   # recuadro y centro del objeto
        x0, y0, x1, y1 = r["bbox"]
        cv2.rectangle(vista, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv2.circle(vista, r["centro"], 3, (0, 0, 255), -1)
    vista = cv2.resize(vista, (640, 480), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((TOP + 540, 980, 3), np.uint8)
    canvas[TOP:TOP + 480, :640] = vista

    # --- zona superior: dos semaforos + resultado final ---
    bloque(canvas, 0, 490, "CONDICIONES DE LA IMAGEN", *r["cond"])
    bloque(canvas, 490, 980, f"OBJETO: {OBJETO}", *r["objmsg"])
    cv2.line(canvas, (490, 0), (490, 75), (0, 0, 0), 2)
    canvas[75:TOP] = r["color"]
    tc = (0, 0, 0) if r["color"] == AMARILLO else (255, 255, 255)
    texto(canvas, f"RESULTADO FINAL:  {r['titulo']}", (12, 99), 0.7, tc, 2)

    # --- histograma ---
    px, py, pw, ph = 650, TOP + 18, 320, 140
    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (60, 60, 60), -1)
    hn = r["h"] / max(r["h"].max(), 1) * (ph - 5)
    pts = np.array([[px + int(k * pw / 255), py + ph - int(hn[k])] for k in range(256)])
    cv2.polylines(canvas, [pts], False, (255, 255, 255), 1)
    for lim in (OSC, CLA):
        xl = px + int(lim * pw / 255)
        cv2.line(canvas, (xl, py), (xl, py + ph), (0, 200, 255), 1)
    texto(canvas, "Histograma", (px, py - 5), 0.45)

    # --- miniaturas: filtrada, negativo y mascara ---
    mascara = r["mask"].astype(np.uint8) * 255
    for k, (nombre, im) in enumerate((("Filtrada", r["filtrada"]), ("Negativo", r["negativo"]),
                                      ("Mascara", mascara))):
        x0 = px + k * 109
        mini = cv2.cvtColor(cv2.resize(im, (102, 77)), cv2.COLOR_GRAY2BGR)
        canvas[TOP + 178:TOP + 255, x0:x0 + 102] = mini
        texto(canvas, nombre, (x0, TOP + 172), 0.45)

    # --- estado de cada condicion ---
    texto(canvas, "ESTADO", (px, TOP + 277), 0.45, (200, 200, 200))
    filas = (("Iluminacion", r["ilum"]), ("Contraste", r["cont"]),
             ("Ruido", r["rui"]), (OBJETO.capitalize(), r["obj"]))
    for k, (nombre, (estado, col)) in enumerate(filas):
        y = TOP + 300 + k * 22
        cv2.circle(canvas, (px + 8, y - 5), 7, col, -1)
        texto(canvas, f"{nombre}: {estado}", (px + 24, y), 0.55)

    # --- valores numericos ---
    lineas = [f"Promedio / Contraste: {r['media']:.1f} / {r['contraste']:.1f}",
              f"Osc/Med/Cla: {r['osc']:.0f} / {r['med']:.0f} / {r['cla']:.0f} %",
              f"Ruido:   {r['ruido']:.1f} % (max {RUIDO_MAX})",
              f"Objeto:  {r['pct']:.1f} % ({r['area']} px)",
              f"Fondo:   {100 - r['pct']:.1f} % ({r['mask'].size - r['area']} px)",
              f"Esperado: {r['lo']:.1f}-{r['hi']:.1f} %",
              f"Umbral:  {r['modo']}",
              f"Ruido simulado: {modo_ruido or 'no'}"]
    for k, linea in enumerate(lineas):
        texto(canvas, linea, (px, TOP + 400 + k * 17), 0.46)

    # --- ayuda de teclas bajo el video ---
    texto(canvas, "q/Esc salir | n ruido | b calibrar (fondo, objeto) | r borrar",
          (10, TOP + 500), 0.45, (200, 200, 200))
    texto(canvas, "v verificar filtros | s guardar captura", (10, TOP + 516), 0.45, (200, 200, 200))
    return canvas


def verificar(gris):
    """Compara tus filtros manuales con cv2.filter2D en un recorte pequeno."""
    c = gris[:40, :60].astype(np.float32)
    for nombre, H, o in (("3x3 promedio", H3, 1), ("5x5 gaussiano", H5, 2)):
        lento = same_lento(c, H, o, o)
        rapido = same(c, H, o, o)
        cvf = cv2.filter2D(c, -1, H, borderType=cv2.BORDER_CONSTANT)
        print(f"{nombre}: |lento-rapido|={np.abs(lento - rapido).max():.4f}  "
              f"|manual-OpenCV|={np.abs(rapido - cvf).max():.4f}")
    print("Histograma lento == rapido:", np.array_equal(histograma_lento(gris), histograma(gris)))


# ================= PROGRAMA PRINCIPAL =================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: no se pudo abrir la camara")
        return
    modos = [None, "sal", "pimienta", "ambos"]
    im = 0
    fondo, pct_ref = None, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        peq = cv2.resize(frame, (320, 240))
        gris = cv2.cvtColor(peq, cv2.COLOR_BGR2GRAY)
        modo = modos[im]
        if modo in ("sal", "ambos"):
            gris = ruido(gris, 5, "sal")
        if modo in ("pimienta", "ambos"):
            gris = ruido(gris, 5, "pimienta")
        vista = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR) if modo else peq

        r = analizar(gris, fondo, pct_ref)
        cv2.imshow(VENTANA, dibujar(vista, r, modo))

        k = cv2.waitKey(1) & 0xFF
        # salir con q, Esc o con la X de la ventana
        if k in (ord("q"), 27) or cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif k == ord("n"):
            im = (im + 1) % len(modos)
        elif k == ord("b"):
            if fondo is None:                       # 1a vez: fondo vacio
                fondo = r["filtrada"].copy()
                print("Fondo guardado. Ponga el objeto y pulse b otra vez.")
            elif pct_ref is None:                   # 2a vez: objeto sobre el fondo
                if r["pct"] < PRESENCIA:
                    print("No se ve el objeto sobre el fondo. Intente de nuevo.")
                else:
                    pct_ref = r["pct"]
                    print(f"Objeto guardado: ocupa {pct_ref:.1f} % de la imagen.")
            else:                                   # 3a vez: reiniciar
                fondo, pct_ref = None, None
                print("Calibracion reiniciada. Muestre el fondo vacio y pulse b.")
        elif k == ord("r"):
            fondo, pct_ref = None, None
            print("Referencia borrada: modo Otsu.")
        elif k == ord("v"):
            verificar(gris)
        elif k == ord("s"):
            cv2.imwrite("captura.png", dibujar(vista, r, modo))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()