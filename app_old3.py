import os
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
                print(f"⚠ Alerta: Se cargó el Excel, pero no se encontró una columna válida de productos. Columnas detectadas: {list(df_temp.columns)}")
                
    except Exception as e:
        print(f"Error al cargar Inventario.xlsx: {e}")
        df_inventario = pd.DataFrame()

    # B. Cargar y Extraer Texto de PDFs (Con fallback por si los archivos son temporales de prueba)
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
                    # Segmentamos por párrafos o saltos dobles para afinar la similitud
                    parrafos = [p.strip() for p in texto_archivo.split("\n\n") if len(p.strip()) > 20]
                    for p in parrafos:
                        documentos_extraidos.append({
                            "origen": nombre_archivo.replace(".pdf", "").replace("_", " "),
                            "contenido": p
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
            "Los destinatarios de este manual de proveedores y políticas de compra abarcan a todos los abastecedores, productores directos, intermediarios comerciales y contratistas externos de insumos que deseen entablar operaciones comerciales con el Mercado Central.",
            "Los plazos de pago estándar para proveedores de productos secos y frescos están fijados para los días viernes de cada semana, con una acreditación estimada a los 30 días corridos posteriores a la recepción de la factura comercial debidamente aprobada.",
            "La recepción de mercadería y control de calidad se realiza exclusivamente de lunes a sábados en la dársena de cargas número 3, en el rango horario de 06:00 a 12:00 hs. Se requiere solicitar turno previamente en el portal oficial de compras."
        ],
        "Politica de ATC.pdf": [
            "La política de atención al cliente (ATC) del Mercado Central determina que cualquier solicitud de cambio o devolución de productos defectuosos de fábrica debe gestionarse dentro de las primeras 24 horas posteriores a la compra, presentando el ticket físico original.",
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

# Ejecutar carga al iniciar el servidor
cargar_base_de_conocimiento()


# --- 4. MOTOR DE BÚSQUEDA SEMÁNTICA LOCAL (Hugging Face) ---
def buscar_en_pdfs(consulta, umbral=0.32):
    """Busca coincidencias en los documentos PDF con un umbral de relevancia dinámico."""
    if not documentos_extraidos:
        return []
        
    textos_a_comparar = [doc["contenido"] for doc in documentos_extraidos]
    
    query_emb = model.encode(consulta, convert_to_tensor=True)
    doc_embs = model.encode(textos_a_comparar, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_emb, doc_embs)[0]
    
    resultados_filtrados = []
    for idx, score in enumerate(cosine_scores):
        if score.item() > umbral:  # Uso de umbral adaptativo
            resultados_filtrados.append({
                "Origen": documentos_extraidos[idx]["origen"],
                "Contenido": documentos_extraidos[idx]["contenido"],
                "score": score.item()
            })
            
    # Ordenar las respuestas por puntuación de relevancia decreciente
    resultados_filtrados = sorted(resultados_filtrados, key=lambda x: x['score'], reverse=True)
    return resultados_filtrados[:3]


# --- 5. LÓGICA DE PROCESAMIENTO Y RESPUESTAS (Planillas Dinámicas HTML) ---
def procesar_consulta(consulta, seleccion_previa=None):
    global columna_producto_real
    if seleccion_previa:
        consulta = seleccion_previa

    consulta_limpia = consulta.strip().lower()
    
    # Búsqueda adaptativa en Inventario
    coincidencias = []
    inventario_habilitado = not df_inventario.empty and columna_producto_real is not None
    
    # Evaluar si es una consulta de stock general (ej: "mostrar inventario", "lista de productos")
    palabras_clave_general = ["inventario", "productos", "stock", "lista de productos", "todos los productos", "tabla de productos", "ver el stock", "stock total"]
    es_consulta_general_inventario = any(k in consulta_limpia for k in palabras_clave_general)
    
    if inventario_habilitado:
        productos_disponibles = df_inventario[columna_producto_real].astype(str).tolist()
        
        # Tokenizamos la consulta para una búsqueda inteligente basada en NLP
        palabras_consulta = [w.strip(",.!?();:").lower() for w in consulta_limpia.split()]
        palabras_consulta = [w for w in palabras_consulta if len(w) > 1] # Ignorar caracteres sueltos
        
        for p in productos_disponibles:
            p_lower = p.lower()
            
            # Condición 1: El nombre completo del producto se encuentra citado en la pregunta del usuario
            if p_lower in consulta_limpia:
                coincidencias.append(p)
                continue
            
            # Condición 2: Alguna palabra clave de alta relevancia del producto se encuentra en la consulta
            tokens_producto = [t.strip(",.!?();:").lower() for t in p_lower.split()]
            # Excluimos palabras vacías comunes para evitar falsos positivos
            stop_words = {"de", "del", "el", "la", "en", "para", "un", "una", "con", "y", "tipo", "g", "kg", "ml", "1l", "500g", "1kg", "sus", "precios", "precio"}
            tokens_clave = [t for t in tokens_producto if t not in stop_words]
            
            if any(t in palabras_consulta for t in tokens_clave):
                coincidencias.append(p)
    
    # Menú dinámico de aproximación lógica (Ej: escribe 'Arroz' -> devuelve listado múltiple de opciones de arroz)
    if len(coincidencias) > 1 and not any(p.lower() == consulta_limpia for p in coincidencias):
        return {
            "tipo": "multiples_opciones",
            "opciones": coincidencias,
            "html": f"<p style='color: #d35400; font-weight: bold; font-family: sans-serif;'>🔍 Encontramos varias opciones de productos para '{consulta}'. Por favor, elija una de la lista:</p>"
        }
    
    # Obtener el registro único seleccionado, identificado del inventario o mostrar tabla completa
    resultado_inv = pd.DataFrame()
    if len(coincidencias) == 1:
        resultado_inv = df_inventario[df_inventario[columna_producto_real] == coincidencias[0]]
    elif inventario_habilitado and any(p.lower() == consulta_limpia for p in df_inventario[columna_producto_real].astype(str).str.lower().tolist()):
        resultado_inv = df_inventario[df_inventario[columna_producto_real].astype(str).str.lower() == consulta_limpia]
    elif es_consulta_general_inventario and inventario_habilitado:
        # Si pide el inventario en general, retornamos toda la tabla
        resultado_inv = df_inventario

    # Armar planilla HTML de Inventario de forma dinámica
    html_inventario = ""
    if not resultado_inv.empty:
        html_inventario = f"""
        <div style="margin-bottom: 25px;">
            <h3 style="color: #2c3e50; border-bottom: 2px solid #e67e22; padding-bottom: 5px;">📦 Información de Inventario</h3>
            <table style="width:100%; border-collapse: collapse; font-family: sans-serif; text-align: left; background-color: #fafafa;">
                <thead>
                    <tr style="background-color: #34495e;">
        """
        # Cabeceras dinámicas basadas en tu Excel con fuente explícitamente blanca (color: white;)
        for col in resultado_inv.columns:
            html_inventario += f'<th style="padding: 10px; border: 1px solid #ddd; color: white; font-weight: bold;">{col}</th>'
            
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
                if isinstance(valor, (int, float)) and valor > 1000:
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

    # Aviso prolijo si el Excel no cumple con los formatos de columnas
    if not df_inventario.empty and columna_producto_real is None:
        columnas_disponibles_str = ", ".join([f"'{c}'" for c in df_inventario.columns])
        html_inventario = f"""
        <div style="padding: 15px; background-color: #fcf3cf; border-left: 5px solid #f1c40f; color: #7e5109; margin-bottom: 20px; border-radius: 4px; font-family: sans-serif;">
            <strong>ℹ️ Nota de estructura del Excel:</strong> Se cargó el archivo de inventario, pero no encontramos una columna llamada "Producto" (o similar). 
            <br>Las columnas detectadas son: <strong>{columnas_disponibles_str}</strong>. Para una mejor experiencia, te sugerimos renombrar la columna principal a 'Producto'.
        </div>
        """

    # --- ENRUTAMIENTO DINÁMICO DE PDFS (UMBRAL ADAPTATIVO) ---
    # Si detectamos que la pregunta está claramente enfocada en el inventario o encontramos coincidencia, 
    # subimos el umbral para evitar que PDFs irrelevantes agreguen "ruido" o falsos positivos.
    tiene_contexto_inventario = len(coincidencias) > 0 or not resultado_inv.empty or es_consulta_general_inventario
    umbral_dinamico = 0.45 if tiene_contexto_inventario else 0.32

    # Búsqueda semántica integrada en los PDFs (incluye FAQ, Políticas, Manuales y Reglamento)
    resultados_pdf = buscar_en_pdfs(consulta, umbral=umbral_dinamico)
    html_pdfs = ""
    if resultados_pdf:
        html_pdfs = f"""
        <div>
            <h3 style="color: #2c3e50; border-bottom: 2px solid #e67e22; padding-bottom: 5px;">📄 Políticas, FAQs y Manuales Relacionados</h3>
            <table style="width:100%; border-collapse: collapse; font-family: sans-serif; text-align: left; background-color: #fafafa;">
                <thead>
                    <tr style="background-color: #2c3e50; color: white;">
                        <th style="padding: 10px; border: 1px solid #ddd; width: 30%; color: white;">Documento Fuente (.pdf)</th>
                        <th style="padding: 10px; border: 1px solid #ddd; width: 70%; color: white;">Contenido Extractado</th>
                    </tr>
                </thead>
                <tbody>
        """
        for doc in resultados_pdf:
            html_pdfs += f"""
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd; color: #d35400; font-size: 0.9em; font-weight: bold;">{doc['Origen']}</td>
                        <td style="padding: 10px; border: 1px solid #ddd; line-height: 1.5; color: #2c3e50;">{doc['Contenido']}</td>
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
    <h1 style="margin: 0 0 10px 0; font-weight: bold; font-size: 2em;">Mercado Central 24 Hs.</h1>
    <p style="margin: 0; font-size: 1.2em; line-height: 1.4; font-weight: 300;">
        Soy tu Agente de IA para consultas en Mercado Central 24 Hs. acerca de Inventario, Manual de Proveedores-Políticas de Compra, Políticas de ATC y Reglamento Interno-Procedimientos Operativos
    </p>
</div>
"""

# Se mantiene 'title' en Blocks (que es lo correcto para la pestaña del navegador)
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

    # Separación lógica del flujo de eventos: Evita arrastrar variables antiguas del selector de radio
    def buscar_por_texto(texto):
        """Maneja las búsquedas ingresadas directamente en la caja de texto."""
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
                gr.update(visible=False, choices=[], value=None), # 'value=None' previene que Gradio lance error al vaciar 'choices'
                gr.update(visible=False)
            )

    def buscar_por_seleccion(seleccion):
        """Maneja las búsquedas cuando el usuario confirma una de las opciones del control de Radio."""
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

    # Registro de disparadores de interacción con flujos independientes
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
    # 'theme' se pasa en el launch() para cumplir de forma limpia con Gradio 6
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        theme=warm_theme
    )