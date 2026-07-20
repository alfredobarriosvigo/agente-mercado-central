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


# --- 2. GENERACIÓN DE ARCHIVOS DEMO (Respaldo por si la carpeta está vacía) ---
def crear_archivos_demo():
    """Genera archivos de prueba en la carpeta 'datos' para asegurar un primer arranque sin errores."""
    if not os.path.exists(RUTA_INVENTARIO):
        df_inv = pd.DataFrame({
            "Producto": [
                "Arroz tipo 1", "Arroz tipo 2", "Arroz Integral Premium", 
                "Aceite de Girasol 1L", "Aceite de Oliva Extra Virgen", 
                "Azúcar Ledesma 1kg", "Fideos Tallarín 500g"
            ],
            "Stock (U)": [120, 85, 40, 200, 95, 300, 150],
            "Precio (Gs)": [7500, 6000, 12000, 14500, 45000, 6500, 5200],
            "Ubicación": ["Pasillo A", "Pasillo A", "Pasillo B", "Pasillo C", "Pasillo C", "Pasillo D", "Pasillo E"]
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
columna_producto_real = None


def chunkear_texto_inteligente(texto):
    """
    Segmenta el texto de manera semántica y jerárquica.
    Une líneas de listas continuas sin romper viñetas e identifica cambios drásticos de subsección.
    """
    lineas = [l.strip() for l in texto.split("\n") if l.strip()]
    chunks = []
    buffer_actual = []
    
    # Expresión regular para detectar subsecciones numéricas como 1.4, 2.2, etc.
    es_seccion = re.compile(r'^\d+\.\d+\s+[A-Z]')
    
    for i, linea in enumerate(lineas):
        # Si detectamos un cambio formal de subsección, cerramos el bloque anterior e iniciamos uno nuevo
        if es_seccion.match(linea) and buffer_actual:
            chunks.append(" ".join(buffer_actual))
            buffer_actual = []
            
        # Si la línea anterior termina sin signo de puntuación y la actual no empieza con viñeta, las fusionamos
        if buffer_actual and not buffer_actual[-1].endswith(('.', ':', ';', '?')) and not linea.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.')):
            buffer_actual[-1] += " " + linea
        else:
            buffer_actual.append(linea)
            
        # Controlamos que los bloques no crezcan demasiado en líneas sueltas
        if len(buffer_actual) >= 18:
            chunks.append(" ".join(buffer_actual))
            buffer_actual = []
            
    if buffer_actual:
        chunks.append(" ".join(buffer_actual))
        
    # Limpiamos y eliminamos fragmentos que sean extremadamente cortos
    chunks_limpios = []
    for c in chunks:
        c_strip = c.strip()
        if len(c_strip) > 40:
            chunks_limpios.append(c_strip)
            
    return chunks_limpios


def cargar_base_de_conocimiento():
    global df_inventario, documentos_extraidos, columna_producto_real
    documentos_extraidos = []
    columna_producto_real = None
    
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
                print(f"✓ Inventario.xlsx cargado correctamente. Columna: '{columna_producto_real}'")
            else:
                df_inventario = df_temp
                print(f"⚠ Alerta: No se encontró columna válida de productos.")
                
    except Exception as e:
        print(f"Error al cargar Inventario.xlsx: {e}")
        df_inventario = pd.DataFrame()

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
            print(f"Error leyendo PDF {nombre_archivo} ({e}). Usando datos de respaldo.")
            inyectar_datos_de_respaldo(nombre_archivo)


def inyectar_datos_de_respaldo(nombre_archivo):
    """Inyecta textos de consulta estructurados si los PDFs están vacíos o corruptos."""
    respaldo = {
        "FAQ.pdf": [
            "¿Cuáles son los horarios de atención al público general del Mercado Central? El mercado opera las 24 horas del día, los 365 días del año de forma ininterrumpida. Las oficinas de facturación y administración atienden de lunes a viernes de 08:00 a 17:00 hs.",
            "¿El estacionamiento tiene algún costo para los clientes? Nuestros clientes disfrutan de estacionamiento gratuito durante las primeras 2 horas, siempre y cuando presenten un ticket de compra con consumo mínimo de $150 MXN (o su equivalente en moneda local según la sucursal) al momento de validar su boleto en el módulo de Atención a Clientes o en las máquinas de validación ubicadas en los accesos al estacionamiento. Transcurrido el periodo de cortesía, o en caso de no contar con ticket de compra válido, se aplicará la tarifa comercial vigente por hora o fracción, la cual está publicada en las pantallas y señales de los accesos vehiculares de cada unidad. Las tarifas pueden variar ligeramente entre sucursales según el municipio, alcaldía o ciudad donde se encuentren. Clientes registrados en el programa \"Cliente VIP Central\" con nivel Oro o Diamante gozan de hasta 4 horas de estacionamiento gratuito por visita. Para motocicletas y bicicletas, contamos con áreas designadas y gratuitas sin límite de tiempo, fomentando la movilidad sostenible.",
            "¿Puedo calentar mis alimentos durante mi turno en las instalaciones de la empresa? Sí, todos los colaboradores cuentan con un comedor común equipado con hornos microondas y refrigeradores para el correcto almacenamiento de sus viandas durante su descanso programado."
        ],
        "Manual_Proveedores-Politicas_Compra.pdf": [
            "1.3 Objetivo del Manual y a Quién Va Dirigido, subsección: Destinatarios. El contenido de este manual de compras es de cumplimiento obligatorio para: \n• Proveedores actuales de Mercado Central 24h en México y en todos los países donde la empresa opera. \n• Candidatos a nuevos proveedores que deseen integrarse a nuestra base de suministro. \n• Personal interno del área de Compras, Almacén, Calidad y Finanzas que interactúa con proveedores. \n• Auditores internos y externos que revisen los procesos de abastecimiento.",
            "La recepción de mercadería y control de calidad se realiza exclusivamente de lunes a sábados en la dársea de cargas número 3, en el rango de 06:00 a 12:00 hs. Se requiere solicitar turno previamente en el portal oficial de compras."
        ],
        "Politica de ATC.pdf": [
            "Para el año 2024, Mercado Central 24h opera con más de 85 sucursales entre México y Latinoamérica... 1.2 Misión Ofrecer a nuestras familias latinoamericanas una experiencia de compra de alta calidad... 1.3 Visión Ser la cadena de supermercados de mayor confianza...",
            "1.4 Valores Orientados al Cliente \n• Honestidad: Precios claros, políticas transparentes, sin sorpresas desagradables. \n• Respeto: Cada cliente es tratado con la dignidad que merece, sin importar el monto de su compra ni la hora de su visita. \n• Calidez: La atención en Mercado Central 24h lleva el trato cercano y hospitalario que caracteriza a la cultura mexicana. \n• Compromiso: Respondemos por nuestros productos y nuestro servicio. Si algo no está bien, lo corrigimos sin demora. \n• Innovación: Buscamos constantemente mejorar la experiencia del cliente a través de tecnología, capacitación y escucha activa. \n• Sustentabilidad: Operamos con conciencia del impacto ambiental y social de nuestras decisiones.",
            "1.5 Compromiso de la Dirección General La Dirección General de Mercado Central 24h asume un compromiso público e irrevocable con la satisfacción de cada cliente que cruza las puertas de cualquiera de nuestras tiendas. Este documento no es un trámite administrativo: es la expresión escrita de los valores que guían a cada uno de nuestros más de 14,000 colaboradores en su trabajo diario."
        ],
        "Reglamento_Interno-Proc_Operativos.pdf": [
            "2.2 Misión, Visión y Valores \nMisión: \nProveer a las familias de México y América Latina una experiencia de compra ininterrumpida, accesible y confiable, ofreciendo productos frescos, de alta calidad, a precios justos y con un servicio excepcional las 24 horas del día, los 365 días del año.",
            "Valores Corporativos: \nValor Descripción \nIntegridad Actuamos con honestidad y transparencia en cada transacción, decisión y relación laboral.",
            "Muy Grave Robo o sustracción de mercancía o valores de la empresa. Falsificación de documentos. Presentarse bajo el efecto de alcohol o drogas. Acoso sexual o laboral. Agresión física a clientes o compañeros.",
            "Procedimiento preventivo para góndolas: Cada encargado de pasillo debe realizar la limpieza, ordenamiento físico y sanitización de las góndolas asignadas al inicio y al cierre de cada de sus turnos operativos. Uniforme reglamentario obligatorio: pantalón blanco de vestir, camisa celeste y calzado cerrado de seguridad."
        ]
    }
    if nombre_archivo in respaldo:
        for p in respaldo[nombre_archivo]:
            documentos_extraidos.append({
                "origen": nombre_archivo.replace(".pdf", "").replace("_", " "),
                "contenido": p
            })

cargar_base_de_conocimiento()


# --- 4. MOTOR DE BÚSQUEDA SEMÁNTICA LOCAL ---
def buscar_en_pdfs(consulta, coincide_producto=False, sustantivos_productos=None):
    if not documentos_extraidos:
        return []
        
    textos_a_comparar = [doc["contenido"] for doc in documentos_extraidos]
    
    query_emb = model.encode(consulta, convert_to_tensor=True)
    doc_embs = model.encode(textos_a_comparar, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_emb, doc_embs)[0]
    
    resultados_filtrados = []
    query_norm = consulta.lower()
    
    for idx, score in enumerate(cosine_scores):
        chunk_original = documentos_extraidos[idx]["contenido"]
        chunk_norm = chunk_original.lower()
        score_final = float(score.item())
        
        # A. CONTROL DE CONTEXTO CRUZADO (FALSOS POSITIVOS DE INVENTARIO)
        if coincide_producto and sustantivos_productos:
            tiene_sustantivo = any(s in chunk_norm for s in sustantivos_productos)
            if ("blanco" in chunk_norm or "integral" in chunk_norm) and not tiene_sustantivo:
                score_final -= 0.55
                
        # B. SISTEMA DE REFUERZO DE RELEVANCIA (BOOSTING SELECCIONADO)
        # 1. Destinatarios del Manual de Proveedores
        if "destinatario" in query_norm or "destinatarios" in query_norm:
            if "destinatario" in chunk_norm or "a quién va dirigido" in chunk_norm:
                score_final += 0.45
            else:
                score_final -= 0.30

        # 2. Valores Orientados al Clientes vs Valores Corporativos Generales / Medidas Disciplinarias
        if "valor" in query_norm or "valores" in query_norm:
            if "cliente" in query_norm or "clientes" in query_norm:
                es_valores_cliente_atc = ("valores orientados" in chunk_norm or 
                                          "valores de servicio" in chunk_norm or 
                                          ("valores" in chunk_norm and ("atencion al cliente" in chunk_norm or "atc" in chunk_norm)))
                
                if es_valores_cliente_atc and "sancion" not in chunk_norm and "robo" not in chunk_norm:
                    score_final += 0.50
                else:
                    score_final -= 0.35
            else:
                if "valores corporativos" in chunk_norm or "valores de la empresa" in chunk_norm:
                    score_final += 0.45
                else:
                    score_final -= 0.30

        # 3. Misión de la Empresa
        if "misión" in query_norm or "mision" in query_norm:
            if "misión:" in chunk_norm or "misión" in chunk_norm:
                if "proveer" in chunk_norm:
                    score_final += 0.55
                else:
                    score_final -= 0.10
            else:
                score_final -= 0.20

        # 4. Control de Contexto Estricto para Estacionamiento
        if "estacionamiento" in query_norm or "estacionar" in query_norm or "parqueo" in query_norm or "cochera" in query_norm:
            if any(term in chunk_norm for term in ["estacionamiento", "estacionar", "parqueo", "vehículo", "vehiculo", "moto", "bicicleta", "boleto", "ticket"]):
                score_final += 0.65
            else:
                score_final -= 0.60

        # C. FILTRADO POR UMBRAL DE RELEVANCIA
        umbral_limite = 0.45 if coincide_producto else 0.32
        
        if score_final > umbral_limite:
            resultados_filtrados.append({
                "Origen": documentos_extraidos[idx]["origen"],
                "Contenido": chunk_original,
                "score": score_final
            })
            
    resultados_filtrados = sorted(resultados_filtrados, key=lambda x: x['score'], reverse=True)
    
    # D. DEDUPLICACIÓN DE CONTENIDO EXACTO O REDUNDANTE
    vistos = set()
    resultados_unicos = []
    for r in resultados_filtrados:
        simplificado = "".join(c for c in r["Contenido"].lower() if c.isalnum())
        clave = simplificado[:150]
        if clave not in vistos:
            vistos.add(clave)
            resultados_unicos.append(r)
            
    if resultados_unicos:
        mejor_score = resultados_unicos[0]["score"]
        resultados_unicos = [r for r in resultados_unicos if r["score"] >= (mejor_score * 0.82)]
        
    return resultados_unicos[:3]


# --- 5. LÓGICA DE PROCESAMIENTO Y RESPUESTAS ---
def procesar_consulta(consulta):
    global columna_producto_real
    consulta_limpia = consulta.strip().lower()
    
    # Ampliamos la lista de stop words para evitar falsos positivos con preposiciones o conectores de una letra como "a"
    stop_words = {
        "y", "sus", "de", "con", "la", "el", "los", "las", "un", "una", "unos", "unas", 
        "para", "por", "en", "sobre", "del", "al", "que", "es", "son", "cuál", "cual", "cuales", "cuáles",
        "ver", "buscar", "precio", "precios", "stock", "inventario", "mostrar", "qué", "que",
        "a", "o", "u", "e", "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
        "aquello", "aquella", "como", "cómo", "dónde", "donde", "cuando", "cuándo", "quién", "quien",
        "nosotros", "ellos", "usted", "ustedes", "mi", "mis", "tu", "tus", "su", "sus"
    }
    
    # Palabras clave orientadas exclusivamente a políticas, manuales u organización
    palabras_corporativas = {
        "valor", "valores", "misión", "mision", "visión", "vision", "política", "politica", 
        "políticas", "politicas", "manual", "reglamento", "procedimiento", "procedimientos", 
        "atc", "atención", "atencion", "cliente", "clientes", "proveedor", "proveedores", 
        "compra", "compras", "horario", "horarios", "estacionamiento", "estacionamientos",
        "sanción", "sanciones", "falta", "faltas", "medida", "medidas", "disciplinaria", "disciplinarias",
        "faq", "faqs", "pregunta", "preguntas"
    }
    
    tokens = [t for t in re.split(r'\W+', consulta_limpia) if t]
    palabras_clave = [t for t in tokens if t not in stop_words]
    
    # Determinamos si la consulta se refiere a políticas de la organización
    contiene_tema_corporativo = any(pc in palabras_corporativas for pc in palabras_clave)
    
    coincidencias = []
    inventario_habilitado = not df_inventario.empty and columna_producto_real is not None
    
    # 1. Comprobación de coincidencia exacta primero (evita bucles infinitos en selectores múltiples)
    es_match_exacto = False
    if inventario_habilitado:
        productos_disponibles = df_inventario[columna_producto_real].astype(str).tolist()
        for p in productos_disponibles:
            if p.lower() == consulta_limpia:
                coincidencias = [p]
                es_match_exacto = True
                break
                
    # 2. Búsqueda por palabras clave solo si no hay una coincidencia exacta directa
    if not es_match_exacto and inventario_habilitado and palabras_clave:
        productos_disponibles = df_inventario[columna_producto_real].astype(str).tolist()
        
        # Filtramos palabras clave de longitud menor o igual a 1 para evitar falsos positivos con letras individuales
        palabras_clave_filtradas = [pc for pc in palabras_clave if len(pc) > 1]
        
        for p in productos_disponibles:
            p_lower = p.lower()
            palabras_producto = re.split(r'\W+', p_lower)
            
            if contiene_tema_corporativo:
                # Si la pregunta se refiere a temas corporativos (ej. valores, clientes, políticas),
                # NO permitimos que palabras del vocabulario corporativo gatillen coincidencia con el inventario.
                # Solamente buscaríamos coincidencias si hay una palabra de producto explícita y su coincidencia es exacta.
                match_exacto = False
                for pc in palabras_clave_filtradas:
                    if pc in palabras_corporativas:
                        continue  # Se omite 'valores', 'clientes', etc., en la búsqueda de productos
                    if pc in palabras_producto:
                        match_exacto = True
                        break
                if match_exacto:
                    coincidencias.append(p)
            else:
                # Búsqueda de inventario estándar utilizando límites de palabras (evita substring parcial erróneo)
                for pc in palabras_clave_filtradas:
                    if any(pc == pp or pp.startswith(pc) for pp in palabras_producto if len(pp) >= len(pc)):
                        coincidencias.append(p)
                        break
                
    coincidencias_agrupadas = sorted(list(set(coincidencias)))
    
    if len(coincidencias_agrupadas) > 1:
        return {
            "tipo": "multiples_opciones",
            "opciones": coincidencias_agrupadas,
            "html": f"<p style='color: #d35400; font-weight: bold; font-family: sans-serif;'>🔍 Encontramos varias opciones de productos para '{consulta}'. Por favor, elija una de la lista:</p>"
        }
    
    resultado_inv = pd.DataFrame()
    coincide_producto = False
    sustantivos_productos_coincidentes = []
    
    if len(coincidencias_agrupadas) == 1:
        resultado_inv = df_inventario[df_inventario[columna_producto_real] == coincidencias_agrupadas[0]]
        coincide_producto = True
        sustantivos_productos_coincidentes = [coincidencias_agrupadas[0].split()[0].lower()]
    elif inventario_habilitado and any(p.lower() == consulta_limpia for p in df_inventario[columna_producto_real].astype(str).str.lower().tolist()):
        resultado_inv = df_inventario[df_inventario[columna_producto_real].astype(str).str.lower() == consulta_limpia]
        coincide_producto = True
        sustantivos_productos_coincidentes = [consulta_limpia.split()[0]]

    # Armado dinámico de la planilla de inventario
    html_inventario = ""
    if not resultado_inv.empty:
        html_inventario = f"""
        <div style="margin-bottom: 25px;">
            <h3 style="color: #2c3e50; border-bottom: 2px solid #e67e22; padding-bottom: 5px; font-family: sans-serif;">📦 Información de Inventario</h3>
            <table style="width:100%; border-collapse: collapse; font-family: sans-serif; text-align: left; background-color: #fafafa; border: 1px solid #ddd;">
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

    resultados_pdf = buscar_en_pdfs(
        consulta, 
        coincide_producto=coincide_producto, 
        sustantivos_productos=sustantivos_productos_coincidentes
    )
    
    html_pdfs = ""
    if resultados_pdf:
        html_pdfs = f"""
        <div>
            <h3 style="color: #2c3e50; border-bottom: 2px solid #e67e22; padding-bottom: 5px; font-family: sans-serif;">📄 Políticas, FAQs y Manuales Relacionados</h3>
            <table style="width:100%; border-collapse: collapse; font-family: sans-serif; text-align: left; background-color: #fafafa; border: 1px solid #ddd;">
                <thead>
                    <tr style="background-color: #2c3e50;">
                        <th style="padding: 10px; border: 1px solid #ddd; width: 30%; color: white !important; font-weight: bold;">Documento Fuente (.pdf)</th>
                        <th style="padding: 10px; border: 1px solid #ddd; width: 70%; color: white !important; font-weight: bold;">Contenido Extractado</th>
                    </tr>
                </thead>
                <tbody>
        """
        for doc in resultados_pdf:
            contenido_formateado = doc['Contenido'].replace("\n", "<br>").replace("•", "&bull;")
            html_pdfs += f"""
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd; color: #d35400; font-size: 0.9em; font-weight: bold; vertical-align: top;">{doc['Origen']}</td>
                        <td style="padding: 10px; border: 1px solid #ddd; line-height: 1.5; color: #2c3e50; white-space: pre-line;">{contenido_formateado}</td>
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
        if not texto.strip():
            return (
                "<p style='color: #e74c3c; font-weight: bold; font-family: sans-serif;'>Por favor, escribe una pregunta válida.</p>", 
                gr.update(visible=False, choices=[], value=None), 
                gr.update(visible=False)
            )
        
        res = procesar_consulta(texto)
        
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

    # Registro de disparadores de la interfaz
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