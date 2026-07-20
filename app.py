import os
import re
import pandas as pd
import gradio as gr
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util

# --- 1. CONFIGURACIÓN DE RUTAS ---
CARPETA_DATOS = "datos"
RUTA_INVENTARIO = os.path.join(CARPETA_DATOS, "Inventario.xlsx")
RUTAS_PDFS = {
    "FAQ.pdf": os.path.join(CARPETA_DATOS, "FAQ.pdf"),
    "Manual_Proveedores-Politicas_Compra.pdf": os.path.join(CARPETA_DATOS, "Manual_Proveedores-Politicas_Compra.pdf"),
    "Politica de ATC.pdf": os.path.join(CARPETA_DATOS, "Politica de ATC.pdf"),
    "Reglamento_Interno-Proc_Operativos.pdf": os.path.join(CARPETA_DATOS, "Reglamento_Interno-Proc_Operativos.pdf")
}

os.makedirs(CARPETA_DATOS, exist_ok=True)


# --- 2. GENERACIÓN DE ARCHIVOS DEMO (Por si la carpeta se encuentra vacía) ---
def crear_archivos_demo():
    """Genera archivos de prueba en la carpeta 'datos' para asegurar un primer arranque sin errores."""
    # Crear Inventario.xlsx de ejemplo
    if not os.path.exists(RUTA_INVENTARIO):
        df_inv = pd.DataFrame({
            "Producto": [
                "Arroz tipo 1", "Arroz tipo 1", "Arroz tipo 2", "Arroz Integral Premium", 
                "Aceite de Girasol 1L", "Aceite de Oliva Extra Virgen", 
                "Azúcar Ledesma 1kg", "Fideos Tallarín 500g"
            ],
            "Stock (U)": [120, 45, 85, 40, 200, 95, 300, 150],
            "Precio (Gs)": [7500, 7500, 6000, 12000, 14500, 45000, 6500, 5200],
            "Ubicación": ["Pasillo A", "Pasillo F", "Pasillo A", "Pasillo B", "Pasillo C", "Pasillo C", "Pasillo D", "Pasillo E"]
        })
        df_inv.to_excel(RUTA_INVENTARIO, index=False)
        print("✓ Creado archivo demo: Inventario.xlsx")

    # Crear marcadores de posición temporales para los archivos PDF
    for nombre_pdf, ruta in RUTAS_PDFS.items():
        if not os.path.exists(ruta):
            with open(ruta, "wb") as f:
                f.write(b"%PDF-1.4 ... (Reemplazar con tu archivo PDF real en la carpeta datos/) ...")
            print(f"⚠ Archivo {nombre_pdf} ausente. Se creó un marcador de posición en {ruta}.")

crear_archivos_demo()


# --- 3. LECTURA Y PROCESAMIENTO INTELIGENTE DE ARCHIVOS ---
print("Cargando modelo de lenguaje local para similitud semántica...")
model = SentenceTransformer('all-MiniLM-L6-v2')

df_inventario = pd.DataFrame()
documentos_extraidos = []
columna_producto_real = None  # Almacena el nombre dinámico de la columna de productos encontrada


def chunkear_texto_inteligente(texto):
    """
    Analiza el texto de un PDF línea por línea y agrupa lógicamente párrafos completos,
    listas con viñetas y títulos cortos que terminan en dos puntos para no perder el contexto.
    """
    lineas = [l.strip() for l in texto.split("\n")]
    parrafos = []
    parrafo_actual = []
    
    for linea in lineas:
        if not linea:
            continue
        
        # Identificar si la línea es un elemento de lista (viñeta)
        es_vineta = linea.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.'))
        
        # Identificar si la línea anterior terminó con dos puntos (:) o si es una viñeta consecutiva
        debe_fusionar = es_vineta or (parrafo_actual and parrafo_actual[-1].endswith(':'))
        
        if debe_fusionar:
            parrafo_actual.append(linea)
        elif len(linea) < 90 and linea.endswith(':'):
            # Es un encabezado o subsección corta (ej: "Misión:")
            if parrafo_actual:
                parrafos.append(" ".join(parrafo_actual))
            parrafo_actual = [linea]
        else:
            if parrafo_actual:
                # Comprobación de líneas rotas (word-wrap) que no terminan en signo de puntuación
                linea_previa = parrafo_actual[-1]
                if not linea_previa.endswith(('.', ':', ';', '!', '?')):
                    parrafo_actual.append(linea)
                else:
                    parrafos.append(" ".join(parrafo_actual))
                    parrafo_actual = [linea]
            else:
                parrafo_actual = [linea]
                
    if parrafo_actual:
        parrafos.append(" ".join(parrafo_actual))
        
    # Limpieza final de fragmentos y unión de espaciados dobles
    resultados = []
    for p in parrafos:
        limpio = p.strip()
        while "  " in limpio:
            limpio = limpio.replace("  ", " ")
        if len(limpio) > 30:  # Evita guardar fragmentos demasiado vacíos o irrelevantes
            resultados.append(limpio)
            
    return resultados


def cargar_base_de_conocimiento():
    global df_inventario, documentos_extraidos, columna_producto_real
    documentos_extraidos = []
    columna_producto_real = None
    
    # A. Cargar Inventario.xlsx con detección flexible de columnas (Previene KeyError)
    try:
        if os.path.exists(RUTA_INVENTARIO):
            df_temp = pd.read_excel(RUTA_INVENTARIO)
            
            # Limpiar nombres de columnas (eliminar espacios en blanco alrededor de los títulos)
            df_temp.columns = [str(c).strip() for c in df_temp.columns]
            
            # Mapeo de búsqueda inteligente para la columna de productos
            posibles_nombres = ["producto", "productos", "artículo", "articulo", "artículos", "articulos", "descripción", "descripcion", "nombre"]
            for col in df_temp.columns:
                if col.lower() in posibles_nombres:
                    columna_producto_real = col
                    break
            
            # Fallback secundario si no hubo coincidencia exacta
            if not columna_producto_real:
                for col in df_temp.columns:
                    if "prod" in col.lower() or "art" in col.lower():
                        columna_producto_real = col
                        break
            
            if columna_producto_real:
                df_inventario = df_temp
                print(f"✓ Inventario.xlsx cargado correctamente. Columna identificada: '{columna_producto_real}'")
            else:
                df_inventario = df_temp
                print(f"⚠ Alerta: Se cargó el Excel, pero no se encontró una columna válida de productos.")
                
    except Exception as e:
        print(f"Error al cargar Inventario.xlsx: {e}")
        df_inventario = pd.DataFrame()

    # B. Cargar y Extraer Texto de PDFs
    for nombre_archivo, ruta in RUTAS_PDFS.items():
        try:
            if os.path.exists(ruta):
                reader = PdfReader(ruta)
                texto_archivo = ""
                
                for page in reader.pages:
                    texto_pagina = page.extract_text()
                    if texto_pagina:
                        texto_archivo += texto_pagina + "\n"
                
                # Validar si el PDF contiene texto útil legible o es un marcador temporal vacío
                if len(texto_archivo.strip()) > 50:
                    chunks = chunkear_texto_inteligente(texto_archivo)
                    for c in chunks:
                        documentos_extraidos.append({
                            "origen": nombre_archivo.replace(".pdf", "").replace("_", " "),
                            "contenido": c
                        })
                else:
                    inyectar_datos_de_respaldo(nombre_archivo)
                    
        except Exception as e:
            print(f"Error leyendo PDF {nombre_archivo} ({e}). Usando datos de respaldo predefinidos.")
            inyectar_datos_de_respaldo(nombre_archivo)


def inyectar_datos_de_respaldo(nombre_archivo):
    """Inyecta textos de consulta real estructurados si los PDFs físicos se encuentran vacíos."""
    respaldo = {
        "FAQ.pdf": [
            "¿Cuáles son los horarios de atención al público general del Mercado Central? El mercado opera las 24 horas del día, los 365 días del año de forma ininterrumpida. Las oficinas de facturación y administración atienden de lunes a viernes de 08:00 a 17:00 hs.",
            "¿El estacionamiento tiene algún costo para los clientes? El ingreso y estacionamiento de vehículos particulares livianos es completamente gratuito durante las primeras 2 horas. Los transportistas de gran porte para carga y descarga abonan una tarifa fija reglamentada en la cabina del portón de entrada.",
            "Este servicio aplica para clientes de todas las membresías; sin embargo, los clientes Diamante cuentan con un asesor personal que puede gestionar pedidos especiales de forma más ágil y directa.",
            "El primer paso siempre debe ser intentar resolver el conflicto mediante comunicación directa y respetuosa con tu supervisor. Sin embargo, si esto no es posible o no resulta efectivo, el protocolo de escalamiento es el siguiente: 1. Solicitar una mediación con el Gerente de Tienda o el Subgerente. 2. Escalar al Gerente de Capital Humano de Zona. 3. Reportar al Comité de Ética o a través de la Línea de Denuncia Anónima."
        ],
        "Manual_Proveedores-Politicas_Compra.pdf": [
            "1.3 Objetivo del Manual y a Quién Va Dirigido, subsección: Destinatarios\n• Proveedores actuales de Mercado Central 24h en México y en todos los países donde la empresa opera.\n• Candidatos a nuevos proveedores que deseen integrarse a nuestra base de suministro.\n• Personal interno del área de Compras, Almacén, Calidad y Finanzas que interactúa con proveedores.\n• Auditores internos y externos que revisen los procesos de abastecimiento.",
            "Los plazos de pago estándar para proveedores de productos secos y frescos están fijados para los días viernes de cada semana, con una acreditación estimada a los 30 días corridos posteriores a la recepción de la factura comercial debidamente aprobada.",
            "La recepción de mercadería y control de calidad se realiza exclusivamente de lunes a sábados en la dársena de cargas número 3, en el rango horario de 06:00 a 12:00 hs. Se requiere solicitar turno previamente en el portal oficial de compras."
        ],
        "Politica de ATC.pdf": [
            "1.4 Valores Orientados al Cliente\n• Honestidad: Precios claros, políticas transparentes, sin sorpresas desagradables.\n• Respeto: Cada cliente es tratado con la dignidad que merece, sin importar el monto de su compra ni la hora de su visita.\n• Calidez: La atención en Mercado Central 24h lleva el trato cercano y hospitalario que caracteriza a la cultura mexicana.\n• Compromiso: Respondemos por nuestros productos y nuestro servicio. Si algo no está bien, lo corregimos sin demora.\n• Innovación: Buscamos constantemente mejorar la experiencia del cliente a través de tecnología, capacitación y escucha activa.\n• Sustentabilidad: Operamos con conciencia del impacto ambiental y social de nuestras decisiones.",
            "La política de atención al cliente (ATC) del Mercado Central determina que cualquier solicitud de cambio o devolución de productos defectuosos de fábrica debe gestionarse dentro de las primeras 24 horas posteriores a la compra, presentando el ticket físico original.",
            "Los medios de pago autorizados en los puntos de venta habilitados incluyen efectivo en moneda de curso legal (guaraníes), tarjetas de crédito y débito de procesadoras autorizadas, y pagos unificados por código QR bancario."
        ],
        "Reglamento_Interno-Proc_Operativos.pdf": [
            "2.2 Misión, Visión y Valores\nMisión: Proveer a las familias de México y América Latina una experiencia de compra ininterrumpida, accesible y confiable, ofreciendo productos frescos, de alta calidad, a precios justos y con un servicio excepcional las 24 horas del día, los 365 días del año.",
            "Visión: Ser la red de abasto más eficiente, innovadora y sostenible del continente, transformando la experiencia de compra diaria y generando valor compartido para nuestros socios, colaboradores y comunidades.",
            "Valores Corporativos:\nValor Descripción Integridad Actuamos con honestidad y transparencia en cada transacción, decisión y relación laboral.\nLiderazgo Inspiramos y guiamos con el ejemplo para alcanzar la excelencia colectiva.\nSustentabilidad Buscamos constantemente mejores formas de operar, servir y crecer de manera sostenible.",
            "Es una directiva obligatoria para todo el personal operativo de piso presentarse a su jornada laboral vistiendo el uniforme reglamentario completo, calzado de seguridad con puntera reforzada y portar en el pecho la credencial de identidad visible. El uniforme consta de remera polo institucional y pantalón blanco.",
            "Procedimiento preventivo para góndolas: Cada encargado de pasillo debe realizar la limpieza, ordenamiento físico y sanitización de las góndolas asignadas al inicio y al cierre de cada de sus turnos operativos."
        ]
    }
    if nombre_archivo in respaldo:
        for p in respaldo[nombre_archivo]:
            documentos_extraidos.append({
                "origen": nombre_archivo.replace(".pdf", "").replace("_", " "),
                "contenido": p
            })

# Ejecutar carga al iniciar el servidor
cargar_base_de_conocimiento()


# --- 4. MOTOR DE BÚSQUEDA SEMÁNTICA LOCAL (Hugging Face con Boost Avanzado) ---
STOP_WORDS = {
    "y", "sus", "de", "del", "el", "la", "los", "las", "un", "una", "en", "para", "con", "por", "sobre", "a",
    "que", "qué", "cuál", "cuales", "cuáles", "quién", "quiénes", "como", "cómo", "dónde", "cuando", "cuándo",
    "quienes", "son", "es", "precios", "precio", "stock", "ubicación", "ubicacion", "cantidad", "ver", "buscar",
    "mostrar", "obtener", "dame", "información", "informacion", "detalle", "detalles", "asociados", "asociado"
}

def normalizar_texto(texto):
    """Limpia caracteres especiales, acentos y convierte el texto a minúsculas."""
    texto = texto.lower()
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "ü": "u", "ñ": "n"
    }
    for orig, rep in replacements.items():
        texto = texto.replace(orig, rep)
    # Reemplazar caracteres no alfanuméricos por espacios
    texto = re.sub(r'[^a-z0-9\s•\-\*]', ' ', texto)
    return " ".join(texto.split())


def buscar_en_pdfs(consulta, coincide_producto=False, sustantivos_productos=None):
    """
    Busca coincidencias semánticas en PDFs utilizando SentenceTransformers,
    aplicando un sistema híbrido de impulsos léxicos contextuales para evitar falsos positivos.
    """
    if not documentos_extraidos:
        return []
        
    textos_a_comparar = [doc["contenido"] for doc in documentos_extraidos]
    
    query_emb = model.encode(consulta, convert_to_tensor=True)
    doc_embs = model.encode(textos_a_comparar, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_emb, doc_embs)[0]
    
    query_norm = normalizar_texto(consulta)
    resultados_filtrados = []
    
    # Determinar el umbral base adaptativo
    umbral_base = 0.45 if coincide_producto else 0.32
    
    for idx, score in enumerate(cosine_scores):
        score_final = float(score.item())
        chunk_norm = normalizar_texto(documentos_extraidos[idx]["contenido"])
        
        # A. FILTRO DE CONTEXTO ESTRICTO (Para búsquedas de inventario)
        # Evita que se traiga información no relacionada (Ej: Reglamento de uniforme "pantalón blanco" si buscas "arroz blanco")
        if coincide_producto and sustantivos_productos:
            contiene_algun_sustantivo = False
            for sust in sustantivos_productos:
                if sust in chunk_norm:
                    contiene_algun_sustantivo = True
                    break
            if not contiene_algun_sustantivo:
                # El término del inventario no está explícitamente en este chunk de PDF, se invalida
                continue

        # B. BOOST POR PALABRAS CLAVE CRÍTICAS (Evita cruzamientos cruzados de contexto)
        
        # 1. Destinatarios
        if "destinatario" in query_norm or "destinatarios" in query_norm:
            if "destinatario" in chunk_norm or "destinatarios" in chunk_norm:
                score_final += 0.45
            else:
                score_final -= 0.20  # Penalizar chunks que no hablen de destinatarios si se pregunta explícitamente

        # 2. Valores orientados al cliente vs Valores corporativos generales
        if "valor" in query_norm or "valores" in query_norm:
            if "cliente" in query_norm or "clientes" in query_norm:
                # Si busca valores para clientes, solo potenciamos los que tengan menciones de cliente o ATC
                if "valores orientados" in chunk_norm or ("valores" in chunk_norm and ("cliente" in chunk_norm or "atc" in chunk_norm)):
                    score_final += 0.45
                else:
                    score_final -= 0.25  # Descenso drástico a valores de reglamento financiero o laboral
            else:
                # Si es una búsqueda corporativa general
                if "valores corporativos" in chunk_norm or ("valores" in chunk_norm and "organizacionales" in chunk_norm):
                    score_final += 0.45

        # 3. Misión de la empresa
        if "mision" in query_norm or "misión" in query_norm:
            if "mision" in chunk_norm or "misión" in chunk_norm:
                score_final += 0.45
            else:
                score_final -= 0.20

        # C. FILTRADO POR UMBRAL BASE
        if score_final > umbral_base:
            resultados_filtrados.append({
                "Origen": documentos_extraidos[idx]["origen"],
                "Contenido": documentos_extraidos[idx]["contenido"],
                "score": score_final
            })
            
    # Ordenar las respuestas por relevancia decreciente
    resultados_filtrados = sorted(resultados_filtrados, key=lambda x: x['score'], reverse=True)
    
    # D. DESCARTE DINÁMICO DE RUIDO SECUNDARIO (Relative Score Thresholding)
    # Si hay una respuesta altamente acertada (ej. score > 0.65), descarta las que sean muy inferiores
    if resultados_filtrados:
        max_score = resultados_filtrados[0]['score']
        if max_score > 0.65:
            resultados_filtrados = [r for r in resultados_filtrados if r['score'] >= (max_score * 0.82)]
            
    return resultados_filtrados[:3]


# --- 5. LÓGICA DE PROCESAMIENTO Y RESPUESTAS (Planillas Dinámicas HTML) ---
def procesar_consulta(consulta, seleccion_previa=None):
    global columna_producto_real
    if seleccion_previa:
        consulta = seleccion_previa

    consulta_norm = normalizar_texto(consulta)
    
    # Filtrado estricto de Stop-Words para encontrar las verdaderas palabras clave de inventario
    palabras_consulta = [p for p in consulta_norm.split() if p not in STOP_WORDS]
    
    coincidencias_completas = []
    coincide_producto = False
    sustantivos_productos_coincidentes = set()
    
    inventario_habilitado = not df_inventario.empty and columna_producto_real is not None
    
    if inventario_habilitado and palabras_consulta:
        # Extraer listado de productos de la base de datos
        productos_disponibles = df_inventario[columna_producto_real].astype(str).tolist()
        
        # Buscar coincidencias: si alguna palabra clave de la consulta está en la descripción del producto
        for prod in productos_disponibles:
            prod_norm = normalizar_texto(prod)
            palabras_prod = prod_norm.split()
            
            # Comprobar si hay intersección de palabras clave
            interseccion = set(palabras_consulta).intersection(set(palabras_prod))
            if interseccion:
                coincidencias_completas.append(prod)
                coincide_producto = True
                # Guardar el sustantivo principal (primera palabra, ej: "Arroz") para el filtro cruzado de PDFs
                sustantivos_productos_coincidentes.add(palabras_prod[0])

    # Agrupación y unicidad de opciones sugeridas para el selector dinámico
    # Esto asegura que "Arroz tipo 1" salga solo una vez en la lista aunque esté en múltiples ubicaciones
    opciones_unicas = sorted(list(set(coincidencias_completas)))
    
    # Menú dinámico de aproximación lógica si se encuentran varios productos coincidentes
    if len(opciones_unicas) > 1 and not any(p.lower() == consulta_norm for p in opciones_unicas):
        return {
            "tipo": "multiples_opciones",
            "opciones": opciones_unicas,
            "html": f"<p style='color: #d35400; font-weight: bold; font-family: sans-serif; margin-bottom: 8px;'>🔍 Encontramos varias opciones de productos para tu consulta. Por favor, selecciona una de la lista de la derecha para ver sus detalles:</p>"
        }
    
    # Obtener los registros del inventario vinculados a la búsqueda
    resultado_inv = pd.DataFrame()
    if len(opciones_unicas) == 1:
        resultado_inv = df_inventario[df_inventario[columna_producto_real] == opciones_unicas[0]]
    elif inventario_habilitado and any(p.lower() == consulta_norm for p in df_inventario[columna_producto_real].astype(str).str.lower().tolist()):
        resultado_inv = df_inventario[df_inventario[columna_producto_real].astype(str).str.lower() == consulta_norm]

    # Armar planilla HTML de Inventario de forma dinámica
    html_inventario = ""
    if not resultado_inv.empty:
        html_inventario = f"""
        <div style="margin-bottom: 25px;">
            <h3 style="color: #2c3e50; border-bottom: 2px solid #e67e22; padding-bottom: 5px; font-family: sans-serif;">📦 Información de Inventario</h3>
            <table style="width:100%; border-collapse: collapse; font-family: sans-serif; text-align: left; background-color: #fafafa; border: 1px solid #ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <thead>
                    <tr style="background-color: #34495e; color: white;">
        """
        # Cabeceras con forzado estricto de color blanco para contraste completo
        for col in resultado_inv.columns:
            html_inventario += f'<th style="padding: 12px 10px; border: 1px solid #ddd; color: white !important; font-weight: bold; text-shadow: 0 1px 1px rgba(0,0,0,0.2);">{col}</th>'
            
        html_inventario += """
                    </tr>
                </thead>
                <tbody>
        """
        # Filas de datos cargadas
        for _, row in resultado_inv.iterrows():
            html_inventario += "<tr>"
            for col in resultado_inv.columns:
                valor = row[col]
                # Formatear números grandes con separadores de miles
                if isinstance(valor, (int, float)) and valor >= 1000:
                    valor_str = f"{valor:,.0f} Gs." if "precio" in col.lower() or "costo" in col.lower() else f"{valor:,.0f}"
                else:
                    valor_str = str(valor)
                
                # Destacar la celda identificada como nombre del producto
                if col == columna_producto_real:
                    html_inventario += f'<td style="padding: 10px; border: 1px solid #ddd; font-weight: bold; color: #2c3e50;">{valor_str}</td>'
                else:
                    html_inventario += f'<td style="padding: 10px; border: 1px solid #ddd; color: #555;">{valor_str}</td>'
            html_inventario += "</tr>"
            
        html_inventario += "</tbody></table></div>"

    # Búsqueda semántica integrada en los PDFs (aplicando filtros restrictivos si coincide con producto)
    resultados_pdf = buscar_en_pdfs(
        consulta, 
        coincide_producto=coincide_producto, 
        sustantivos_productos=list(sustantivos_productos_coincidentes)
    )
    
    html_pdfs = ""
    if resultados_pdf:
        html_pdfs = f"""
        <div>
            <h3 style="color: #2c3e50; border-bottom: 2px solid #e67e22; padding-bottom: 5px; font-family: sans-serif;">📄 Políticas, FAQs y Manuales Relacionados</h3>
            <table style="width:100%; border-collapse: collapse; font-family: sans-serif; text-align: left; background-color: #fafafa; border: 1px solid #ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <thead>
                    <tr style="background-color: #2c3e50; color: white;">
                        <th style="padding: 12px 10px; border: 1px solid #ddd; width: 30%; color: white !important; font-weight: bold; text-shadow: 0 1px 1px rgba(0,0,0,0.2);">Documento Fuente (.pdf)</th>
                        <th style="padding: 12px 10px; border: 1px solid #ddd; width: 70%; color: white !important; font-weight: bold; text-shadow: 0 1px 1px rgba(0,0,0,0.2);">Contenido Extractado</th>
                    </tr>
                </thead>
                <tbody>
        """
        for doc in resultados_pdf:
            # Formateado amigable y limpio de viñetas para que se rendericen correctamente
            contenido_formateado = doc['Contenido'].replace("•", "<br>•").replace("\n", "<br>")
            if contenido_formateado.startswith("<br>"):
                contenido_formateado = contenido_formateado[4:]
                
            html_pdfs += f"""
                    <tr>
                        <td style="padding: 12px 10px; border: 1px solid #ddd; color: #d35400; font-size: 0.95em; font-weight: bold; vertical-align: top; background-color: #fdfaf7;">{doc['Origen']}</td>
                        <td style="padding: 12px 10px; border: 1px solid #ddd; line-height: 1.6; color: #2c3e50; font-size: 0.95em;">{contenido_formateado}</td>
                    </tr>
            """
        html_pdfs += "</tbody></table></div>"

    if html_inventario == "" and html_pdfs == "":
        return {
            "tipo": "sin_resultados",
            "html": f"""
            <div style="padding: 15px; background-color: #fdf2e9; border-left: 5px solid #e67e22; color: #d35400; border-radius: 4px; font-family: sans-serif;">
                ⚠️ No encontramos coincidencia exacta o semántica sobre <strong>"{consulta}"</strong> en la base de datos de inventario o manuales. Intenta preguntar con otras palabras clave.
            </div>
            """
        }

    return {
        "tipo": "exito",
        "html": html_inventario + html_pdfs
    }


# --- 6. INTERFAZ DE GRADIO (Compatibilidad Total con Gradio 6) ---
warm_theme = gr.themes.Default(
    primary_hue="orange",
    secondary_hue="amber",
    neutral_hue="stone",
)

bienvenida = """
<div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #e67e22, #d35400); color: white; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); font-family: sans-serif;">
    <h1 style="margin: 0 0 10px 0; font-weight: bold; font-size: 2em; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">Mercado Central 24 Hs.</h1>
    <p style="margin: 0; font-size: 1.2em; line-height: 1.4; font-weight: 300;">
        Soy tu Agente de IA para consultas en Mercado Central 24 Hs. acerca de Inventario, Manual de Proveedores-Políticas de Compra, Políticas de ATC y Reglamento Interno-Procedimientos Operativos
    </p>
</div>
"""


def buscar_por_texto(texto):
    """Manejador disparado cuando el usuario escribe en la barra de búsqueda y presiona Enter o Buscar."""
    if not texto or not texto.strip():
        return (
            "<p style='color: #e74c3c; font-weight: bold; font-family: sans-serif;'>Por favor, escribe una pregunta válida.</p>", 
            gr.update(visible=False, choices=[], value=None), 
            gr.update(visible=False)
        )
    
    res = procesar_consulta(texto)
    
    if res["tipo"] == "multiples_opciones":
        # Activar el selector de Radio con las opciones y vaciar el valor para prevenir error de Gradio
        return (
            res["html"], 
            gr.update(choices=res["opciones"], value=res["opciones"][0], visible=True), 
            gr.update(visible=True)
        )
    else:
        # Ocultar el selector y vaciar choices y value para evitar validaciones cruzadas erróneas de Gradio
        return (
            res["html"], 
            gr.update(visible=False, choices=[], value=None), 
            gr.update(visible=False)
        )


def buscar_por_seleccion(seleccion):
    """Manejador disparado cuando el usuario hace clic en el botón de confirmar la opción elegida del Radio."""
    if not seleccion:
        return (
            "<p style='color: #e74c3c; font-weight: bold; font-family: sans-serif;'>Por favor, selecciona una opción válida.</p>", 
            gr.update(visible=False, choices=[], value=None), 
            gr.update(visible=False)
        )
    
    res = procesar_consulta(seleccion)
    return (
        res["html"], 
        gr.update(visible=False, choices=[], value=None), 
        gr.update(visible=False)
    )


with gr.Blocks(title="Agente IA - Mercado Central 24 Hs.") as demo:
    gr.HTML(bienvenida)
    
    with gr.Row():
        with gr.Column(scale=4):
            input_txt = gr.Textbox(
                label="Haz tu pregunta sobre inventario, políticas del mercado, procedimientos o FAQs:", 
                placeholder="Ej. Arroz, tiempo de pago, devolución de mercadería, estacionamiento, horarios, misión...", 
                lines=1
            )
            btn_buscar = gr.Button("Buscar", variant="primary")
            
        with gr.Column(scale=2):
            selector_multiples = gr.Radio(
                choices=[], 
                label="¿A cuál de estos productos te refieres?", 
                visible=False
            )
            btn_confirmar_seleccion = gr.Button("Ver detalle del producto seleccionado", visible=False, variant="secondary")

    output_html = gr.HTML(
        label="Respuestas del Agente",
        value="<div style='color: #7f8c8d; text-align: center; padding: 40px; font-style: italic; font-family: sans-serif;'>Los resultados de tu búsqueda aparecerán en este panel de manera estructurada y prolija.</div>"
    )

    # Registro de disparadores de interacción
    btn_buscar.click(
        fn=buscar_por_texto, 
        inputs=[input_txt], 
        outputs=[output_html, selector_multiples, btn_confirmar_seleccion]
    )
    input_txt.submit(
        fn=buscar_por_texto, 
        inputs=[input_txt], 
        outputs=[output_html, selector_multiples, btn_confirmar_seleccion]
    )
    btn_confirmar_seleccion.click(
        fn=buscar_por_seleccion, 
        inputs=[selector_multiples], 
        outputs=[output_html, selector_multiples, btn_confirmar_seleccion]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        theme=warm_theme
    )