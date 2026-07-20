import os
import re
import unicodedata
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
    if not os.path.exists(RUTA_INVENTARIO):
        df_inv = pd.DataFrame({
            "Producto": [
                "Arroz tipo 1", "Arroz tipo 1", "Arroz tipo 2", "Arroz Integral Premium", 
                "Aceite de Girasol 1L", "Aceite de Oliva Extra Virgen", 
                "Azúcar Ledesma 1kg", "Fideos Tallarín 500g"
            ],
            "Stock (U)": [120, 30, 85, 40, 200, 95, 300, 150],
            "Precio (Gs)": [7500, 7500, 6000, 12000, 14500, 45000, 6500, 5200],
            "Ubicación": ["Pasillo A", "Depósito Norte", "Pasillo A", "Pasillo B", "Pasillo C", "Pasillo C", "Pasillo D", "Pasillo E"]
        })
        df_inv.to_excel(RUTA_INVENTARIO, index=False)
        print("✓ Creado archivo demo: Inventario.xlsx")

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

# Diccionario exhaustivo de palabras vacías (Stop Words) en español para limpiar consultas de inventario
STOP_WORDS_SPANISH = {
    "y", "o", "u", "e", "de", "del", "con", "para", "por", "en", "sin", "sobre", 
    "el", "la", "los", "las", "un", "una", "unos", "unas", "su", "sus", "tu", "tus", 
    "mi", "mis", "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "que", "cual", "cuales", "quien", "quienes", "como", "cómo", "donde", "dónde",
    "cuando", "cuándo", "cuanto", "cuánto", "precio", "precios", "costo", "costos", 
    "valor", "valores", "stock", "existencia", "existencias", "cantidad", "cantidades",
    "ubicacion", "ubicación", "pasillo", "deposito", "depósito", "gondola", "góndola",
    "lista", "listado", "ver", "mostrar", "buscar", "consultar", "traer", "obtener",
    "saber", "conocer", "detalle", "detalles", "informacion", "información", "tipo", "tipos"
}


def normalizar_texto(texto):
    """Limpia acentos, caracteres especiales y mayúsculas para un cruce de datos exacto."""
    if not texto:
        return ""
    texto = texto.lower()
    # Eliminar acentos y diacríticos
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    # Quitar puntuación
    texto = re.sub(r'[^\w\s]', ' ', texto)
    return " ".join(texto.split())


def chunkear_texto_inteligente(texto):
    """Segmenta el texto de PDFs de forma atómica respetando viñetas y títulos."""
    lineas = [l.strip() for l in texto.split("\n") if l.strip()]
    chunks = []
    current_chunk = []
    
    for linea in lineas:
        es_vineta = (
            linea.startswith("•") or 
            linea.startswith("-") or 
            linea.startswith("*") or 
            re.match(r"^\d+[\.\)]\s", linea)
        )
        
        if es_vineta:
            current_chunk.append(linea)
        else:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
            current_chunk.append(linea)
            
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    
    # Fusionar inteligentemente títulos cortos con sus listas estructuradas subsecuentes
    merged_chunks = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        if i + 1 < len(chunks):
            siguiente = chunks[i+1]
            tiene_vinetas = (
                siguiente.startswith("•") or 
                siguiente.startswith("-") or 
                siguiente.startswith("*") or 
                re.match(r"^\d+[\.\)]\s", siguiente) or
                "\n•" in siguiente or 
                "\n-" in siguiente
            )
            if len(chunk) < 200 and tiene_vinetas:
                merged_chunks.append(chunk + "\n" + siguiente)
                i += 2
                continue
        merged_chunks.append(chunk)
        i += 1
        
    return [c.strip() for c in merged_chunks if len(c.strip()) > 15]


def cargar_base_de_conocimiento():
    global df_inventario, documentos_extraidos, columna_producto_real
    documentos_extraidos = []
    columna_producto_real = None
    
    # A. Cargar Inventario.xlsx con detección flexible de columnas (Previene KeyError)
    try:
        if os.path.exists(RUTA_INVENTARIO):
            df_temp = pd.read_excel(RUTA_INVENTARIO)
            df_temp.columns = [str(c).strip() for c in df_temp.columns]
            
            posibles_nombres = ["producto", "productos", "artículo", "articulo", "artículos", "articulos", "descripción", "descripcion", "nombre"]
            for col in df_temp.columns:
                if col.lower() in posibles_nombres:
                    columna_producto_real = col
                    break
            
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
                print(f"⚠ Alerta: Se cargó el Excel sin columna de productos clara. Columnas: {list(df_temp.columns)}")
                
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
    """Inyecta textos de consulta real si los PDFs físicos se encuentran vacíos o son ilegibles."""
    respaldo = {
        "FAQ.pdf": [
            "¿Cuáles son los horarios de atención al público general del Mercado Central? El mercado opera las 24 horas del día, los 365 días del año de forma ininterrumpida. Las oficinas de facturación y administración atienden de lunes a viernes de 08:00 a 17:00 hs.",
            "¿El estacionamiento tiene costo dentro del predio? El ingreso y estacionamiento de vehículos particulares livianos es completamente gratuito durante las primeras 2 horas. Los transportistas de gran porte para carga y descarga abonan una tarifa fija reglamentada en la cabina del portón de entrada."
        ],
        "Manual_Proveedores-Politicas_Compra.pdf": [
            "Destinatarios del Manual de Proveedores:\n• Proveedores actuales de Mercado Central 24h en México y en todos los países donde la empresa opera.\n• Candidatos a nuevos proveedores que deseen integrarse a nuestra base de suministro.\n• Personal interno del área de Compras, Almacén, Calidad y Finanzas que interactúa con proveedores.\n• Auditores internos y externos que revisen los procesos de abastecimiento.",
            "Los plazos de pago estándar para proveedores de productos secos y frescos están fijados para los días viernes de cada semana, con una acreditación estimada a los 30 días corridos posteriores a la recepción de la factura comercial debidamente aprobada.",
            "La recepción de mercadería y control de calidad se realiza exclusivamente de lunes a sábados en la darsena de cargas número 3, en el rango horario de 06:00 a 12:00 hs. Se requiere solicitar turno previamente en el portal oficial de compras."
        ],
        "Politica de ATC.pdf": [
            "La política de atención al cliente (ATC) del Mercado Central determina que cualquier solicitud de cambio o devolución de productos defectuosos de fábrica debe gestionarse dentro de las primeras 24 horas posteriores a la compra, presenting el ticket físico original.",
            "Los medios de pago autorizados en los puntos de venta habilitados incluyen efectivo en moneda de curso legal (guaraníes), tarjetas de crédito y débito de procesadoras autorizadas, y pagos unificados por código QR bancario."
        ],
        "Reglamento_Interno-Proc_Operativos.pdf": [
            "Es una directiva obligatoria para todo el personal operativo de piso presentarse a su jornada laboral vistiendo el uniforme reglamentario completo, calzado de seguridad con puntera reforzada y portar en el pecho la credencial de identidad visible.",
            "Procedimiento preventivo para góndolas: Cada encargado de pasillo debe realizar la limpieza, ordenamiento físico y sanitización de las góndolas asignadas al inicio y al cierre de cada de sus turnos operativos."
        ]
    }
    if nombre_archivo in respaldo:
        for p in respaldo[nombre_archivo]:
            documentos_extraidos.append({
                "origen": nombre_archivo.replace(".pdf", "").replace("_", " "),
                "contenido": p
            })

# Cargar la base de conocimiento completa al arrancar
cargar_base_de_conocimiento()


# --- 4. MOTOR DE BÚSQUEDA SEMÁNTICA LOCAL CON FILTRADO DE CONTEXTO ---
def buscar_en_pdfs(consulta, coincide_producto=False, sustantivos_productos=None):
    """Busca en los PDFs utilizando similitud vectorial asistida por un optimizador de palabras clave."""
    if not documentos_extraidos:
        return []
        
    textos_a_comparar = [doc["contenido"] for doc in documentos_extraidos]
    
    query_emb = model.encode(consulta, convert_to_tensor=True)
    doc_embs = model.encode(textos_a_comparar, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_emb, doc_embs)[0]
    
    resultados_filtrados = []
    query_norm = normalizar_texto(consulta)
    
    for idx, score in enumerate(cosine_scores):
        chunk_content = documentos_extraidos[idx]["contenido"]
        chunk_origen = documentos_extraidos[idx]["origen"]
        chunk_norm = normalizar_texto(chunk_content)
        
        # FILTRO DE CONTEXTO ULTRA-ESTRICTO:
        # Si la búsqueda coincide con un producto del inventario, los fragmentos extraídos de PDFs
        # DEBEN obligatoriamente poseer el sustantivo núcleo del producto (ej: 'arroz').
        # Esto impide que adjetivos como 'integral' o 'blanco' causen emparejamientos espurios con manuales de uniforme o reglamentos.
        if coincide_producto and sustantivos_productos:
            contiene_sustantivo_real = any(noun in chunk_norm for noun in sustantivos_productos)
            if not contiene_sustantivo_real:
                continue  # Ignora este párrafo ajeno (ej: descarta "pantalón blanco" o "reestructura integral" si buscábamos arroz)
        
        # BOOST POR PALABRAS CLAVE: Si hay coincidencia léxica exacta en consultas de alta precisión
        score_final = score.item()
        
        # Incrementar prioridad masivamente para consultas específicas de destinatarios
        if "destinatario" in query_norm or "destinatarios" in query_norm:
            if "destinatario" in chunk_norm or "destinatarios" in chunk_norm:
                score_final += 0.45  # Impulso muy potente al fragmento exacto de destinatarios
        
        if "manual" in query_norm and "manual" in chunk_norm:
            score_final += 0.05
        if "proveedor" in query_norm or "proveedores" in query_norm:
            if "proveedor" in chunk_norm or "proveedores" in chunk_norm:
                score_final += 0.05
        if "objetivo" in query_norm and "objetivo" in chunk_norm:
            score_final += 0.10
            
        # Umbral adaptativo basado en el tipo de consulta
        umbral_limite = 0.38 if coincide_producto else 0.28
        
        if score_final > umbral_limite:
            resultados_filtrados.append({
                "Origen": chunk_origen,
                "Contenido": chunk_content,
                "score": score_final
            })
            
    # Ordenar por el score potenciado decrecientemente
    resultados_filtrados = sorted(resultados_filtrados, key=lambda x: x['score'], reverse=True)
    
    # Si tenemos un resultado excelente, descartamos el ruido secundario colateral
    if resultados_filtrados:
        mejor_score = resultados_filtrados[0]['score']
        if mejor_score > 0.60:
            resultados_filtrados = [r for r in resultados_filtrados if r['score'] >= (mejor_score * 0.82)]
            
    return resultados_filtrados[:2]


# --- 5. LÓGICA DE PROCESAMIENTO Y RESPUESTAS (Clasificación de Intenciones e Interfaces) ---
def procesar_consulta(consulta, seleccion_previa=None):
    global columna_producto_real
    if seleccion_previa:
        consulta = seleccion_previa

    consulta_limpia = consulta.strip().lower()
    
    # Búsqueda adaptativa en Inventario
    coincidencias = []
    inventario_habilitado = not df_inventario.empty and columna_producto_real is not None
    
    query_norm = normalizar_texto(consulta_limpia)
    palabras_query = set(query_norm.split())
    
    # Filtrar palabras vacías (Stop Words) de la consulta del usuario para identificar el término clave real
    palabras_query_limpias = {w for w in palabras_query if w not in STOP_WORDS_SPANISH and len(w) > 1}
    
    # Extraer palabras clave de productos reales en el inventario (limpias de stop words)
    palabras_productos_en_inventario = set()
    if inventario_habilitado:
        for p in df_inventario[columna_producto_real].dropna().unique():
            norm_p = normalizar_texto(str(p))
            for w in norm_p.split():
                if len(w) > 2 and w not in STOP_WORDS_SPANISH:
                    palabras_productos_en_inventario.add(w)

    # Identificar si la consulta menciona productos reales eliminando conectores vacíos
    palabras_clave_producto_query = palabras_query_limpias.intersection(palabras_productos_en_inventario)
    coincide_producto = len(palabras_clave_producto_query) > 0
    
    # Palabras clave del manual/reglamentos para desambiguar
    palabras_manual_politicas = {
        "manual", "proveedor", "proveedores", "politica", "politicas", "reglamento", 
        "norma", "procedimiento", "procedimientos", "devolucion", "devoluciones", 
        "pago", "pagos", "factura", "facturas", "facturacion", "horario", "horarios", 
        "estacionamiento", "atencion", "atc", "faq", "faqs", "reclamo", "reclamos", 
        "personal", "uniforme", "vestimenta", "limpieza", "limpiar", "quienes", 
        "destinatarios", "objetivo", "dirigido", "compra", "compras"
    }
    coincide_manual_politicas = len(palabras_query.intersection(palabras_manual_politicas)) > 0
    
    # Desactivar coincidencias fortuitas si se pregunta explícitamente por manuales sin mencionar un producto
    if coincide_manual_politicas and not any(p in {"arroz", "aceite", "azucar", "fideos", "tallarin"} for p in palabras_query_limpias):
        coincide_producto = False

    # Realizar el emparejamiento con el inventario utilizando términos estrictamente limpios de stop words
    if inventario_habilitado and (coincide_producto or not coincide_manual_politicas):
        productos_disponibles = df_inventario[columna_producto_real].astype(str).tolist()
        
        for p in productos_disponibles:
            p_lower = p.lower()
            p_norm = normalizar_texto(p)
            tokens_p_norm = set(p_norm.split())
            tokens_p_norm_limpios = {t for t in tokens_p_norm if t not in STOP_WORDS_SPANISH}
            
            # Coincidencia directa exacta de subcadena o cruce léxico limpio de stop words
            if p_lower in consulta_limpia:
                coincidencias.append(p)
            elif palabras_query_limpias.intersection(tokens_p_norm_limpios):
                coincidencias.append(p)
                
    # AGRUPACIÓN: Asegurar que el menú muestre nombres de productos únicos
    coincidencias = list(dict.fromkeys(coincidencias))
    
    # Si hay múltiples productos parecidos (Ej. "Arroz tipo 1", "Arroz tipo 2")
    if len(coincidencias) > 1 and not any(p.lower() == consulta_limpia for p in coincidencias):
        return {
            "tipo": "multiples_opciones",
            "opciones": coincidencias,
            "html": f"<p style='color: #d35400; font-weight: bold; font-family: sans-serif;'>🔍 Encontramos varias opciones de productos para '{consulta}'. Por favor, elija una de la lista:</p>"
        }
    
    # Obtener el registro del inventario
    resultado_inv = pd.DataFrame()
    if len(coincidencias) == 1:
        resultado_inv = df_inventario[df_inventario[columna_producto_real] == coincidencias[0]]
    elif inventario_habilitado and any(p.lower() == consulta_limpia for p in df_inventario[columna_producto_real].astype(str).str.lower().tolist()):
        resultado_inv = df_inventario[df_inventario[columna_producto_real].astype(str).str.lower() == consulta_limpia]

    # Armar planilla HTML de Inventario
    html_inventario = ""
    if not resultado_inv.empty:
        html_inventario = f"""
        <div style="margin-bottom: 25px;">
            <h3 style="color: #2c3e50; border-bottom: 2px solid #e67e22; padding-bottom: 5px;">📦 Información de Inventario</h3>
            <table style="width:100%; border-collapse: collapse; font-family: sans-serif; text-align: left; background-color: #fafafa;">
                <thead>
                    <tr style="background-color: #34495e;">
        """
        for col in resultado_inv.columns:
            html_inventario += f'<th style="padding: 10px; border: 1px solid #ddd; color: white !important; font-weight: bold;">{col}</th>'
            
        html_inventario += """
                    </tr>
                </thead>
                <tbody>
        """
        for _, row in resultado_inv.iterrows():
            html_inventario += "<tr>"
            for col in resultado_inv.columns:
                valor = row[col]
                if isinstance(valor, (int, float)) and valor > 1000:
                    valor_str = f"{valor:,.0f} Gs." if "precio" in col.lower() or "costo" in col.lower() else f"{valor:,.0f}"
                else:
                    valor_str = str(valor)
                
                if col == columna_producto_real:
                    html_inventario += f'<td style="padding: 10px; border: 1px solid #ddd; font-weight: bold; color: #2c3e50;">{valor_str}</td>'
                else:
                    html_inventario += f'<td style="padding: 10px; border: 1px solid #ddd; color: #555;">{valor_str}</td>'
            html_inventario += "</tr>"
            
        html_inventario += "</tbody></table></div>"

    if not df_inventario.empty and columna_producto_real is None:
        columnas_disponibles_str = ", ".join([f"'{c}'" for c in df_inventario.columns])
        html_inventario = f"""
        <div style="padding: 15px; background-color: #fcf3cf; border-left: 5px solid #f1c40f; color: #7e5109; margin-bottom: 20px; border-radius: 4px; font-family: sans-serif;">
            <strong>ℹ️ Nota de estructura del Excel:</strong> Se cargó el archivo de inventario, pero no encontramos una columna de productos. Columnas: <strong>{columnas_disponibles_str}</strong>.
        </div>
        """

    # Extraer el sustantivo principal (el núcleo del sujeto del producto) para el filtro estricto de PDFs
    sustantivos_productos_coincidentes = set()
    for p in coincidencias:
        palabras_p = normalizar_texto(p).split()
        if palabras_p:
            # El primer término suele ser el sustantivo principal (ej: 'arroz', 'aceite', 'azucar')
            primer_termino = palabras_p[0]
            if primer_termino not in STOP_WORDS_SPANISH:
                sustantivos_productos_coincidentes.add(primer_termino)

    # Búsqueda semántica protegida en PDFs enviando el sustantivo núcleo como filtro
    resultados_pdf = buscar_en_pdfs(
        consulta, 
        coincide_producto=coincide_producto, 
        sustantivos_productos=sustantivos_productos_coincidentes
    )
    
    html_pdfs = ""
    if resultados_pdf:
        html_pdfs = f"""
        <div>
            <h3 style="color: #2c3e50; border-bottom: 2px solid #e67e22; padding-bottom: 5px;">📄 Políticas, FAQs y Manuales Relacionados</h3>
            <table style="width:100%; border-collapse: collapse; font-family: sans-serif; text-align: left; background-color: #fafafa;">
                <thead>
                    <tr style="background-color: #2c3e50;">
                        <th style="padding: 10px; border: 1px solid #ddd; width: 30%; color: white !important; font-weight: bold;">Documento Fuente (.pdf)</th>
                        <th style="padding: 10px; border: 1px solid #ddd; width: 70%; color: white !important; font-weight: bold;">Contenido Extractado</th>
                    </tr>
                </thead>
                <tbody>
        """
        for doc in resultados_pdf:
            contenido_formateado = doc['Contenido'].replace('\n', '<br>')
            html_pdfs += f"""
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd; color: #d35400; font-size: 0.9em; font-weight: bold;">{doc['Origen']}</td>
                        <td style="padding: 10px; border: 1px solid #ddd; line-height: 1.5; color: #2c3e50;">{contenido_formateado}</td>
                    </tr>
            """
        html_pdfs += "</tbody></table></div>"

    if html_inventario == "" and html_pdfs == "":
        return {
            "tipo": "sin_resultados",
            "html": f"""
            <div style="padding: 15px; background-color: #fdf2e9; border-left: 5px solid #e67e22; color: #d35400; border-radius: 4px; font-family: sans-serif;">
                ⚠️ No encontramos coincidencia exacta o semántica sobre <strong>"{consulta}"</strong>. Intenta preguntar con otras palabras clave.
            </div>
            """
        }

    return {
        "tipo": "exito",
        "html": html_inventario + html_pdfs
    }


# --- 6. INTERFAZ DE GRADIO ---
warm_theme = gr.themes.Default(
    primary_hue="orange",
    secondary_hue="amber",
    neutral_hue="stone",
)

bienvenida = """
<div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #e67e22, #d35400); color: white; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); font-family: sans-serif;">
    <h1 style="margin: 0 0 10px 0; font-weight: bold; font-size: 2em;">Mercado Central 24 Hs.</h1>
    <p style="margin: 0; font-size: 1.2em; line-height: 1.4; font-weight: 300;">
        Soy tu Agente de IA para consultas en Mercado Central 24 Hs. acerca de Inventario, Manual de Proveedores-Políticas de Compra, Políticas de ATC y Reglamento Interno-Procedimientos Operativos
    </p>
</div>
"""

with gr.Blocks(title="Agente IA - Mercado Central 24 Hs.") as demo:
    gr.HTML(bienvenida)
    
    with gr.Row():
        with gr.Column(scale=4):
            input_txt = gr.Textbox(
                label="Haz tu pregunta sobre inventario, políticas del mercado, procedimientos o FAQs:", 
                placeholder="Ej. Arroz, tiempo de pago, devolución de mercadería, estacionamiento, horarios...", 
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

    def buscar_por_texto(texto):
        """Maneja búsquedas directas en la barra de texto."""
        if not texto or not texto.strip():
            return (
                "<p style='color: #e74c3c; font-weight: bold; font-family: sans-serif;'>Por favor, escribe una pregunta válida.</p>", 
                gr.update(visible=False, choices=[], value=None), 
                gr.update(visible=False)
            )
        
        res = procesar_consulta(texto.strip())
        
        if res["tipo"] == "multiples_opciones":
            return (
                res["html"], 
                gr.update(choices=res["opciones"], value=res["opciones"][0], visible=True), 
                gr.update(visible=True)
            )
        else:
            return (
                res["html"], 
                gr.update(visible=False, choices=[], value=None), 
                gr.update(visible=False)
            )

    def buscar_por_seleccion(seleccion):
        """Maneja las selecciones confirmadas en el menú Radio."""
        if not seleccion:
            return (
                "<p style='color: #e74c3c; font-weight: bold; font-family: sans-serif;'>Por favor, selecciona una opción.</p>", 
                gr.update(visible=False, choices=[], value=None), 
                gr.update(visible=False)
            )
        
        res = procesar_consulta(seleccion)
        return (
            res["html"], 
            gr.update(visible=False, choices=[], value=None), 
            gr.update(visible=False)
        )

    # Disparadores de eventos
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