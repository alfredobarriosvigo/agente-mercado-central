<!-- ... existing code ... -->
        # 2. Valores Orientados al Clientes vs Valores Corporativos Generales / Medidas Disciplinarias
        if "valor" in query_norm or "valores" in query_norm:
            if "cliente" in query_norm or "clientes" in query_norm:
                # Es una búsqueda específica sobre la política de ATC
                es_valores_cliente_atc = ("valores orientados" in chunk_norm or 
                                          "valores de servicio" in chunk_norm or 
                                          ("valores" in chunk_norm and ("atencion al cliente" in chunk_norm or "atc" in chunk_norm)))
                
                # Descartar falsos positivos de sanciones del reglamento (como robo de valores, agresión a clientes)
                if es_valores_cliente_atc and "sancion" not in chunk_norm and "robo" not in chunk_norm:
                    score_final += 0.50
                else:
                    score_final -= 0.35  # Penalización agresiva para descartar ruido del reglamento interno
            else:
                # Valores corporativos de misión
<!-- ... existing code ... -->
```
eof

Con esta sintonía fina en el algoritmo de búsqueda, al consultar *"cuales son los valores orientados a clientes"*, el sistema aislará de forma estricta la sección 1.4 del documento de ATC y mantendrá la respuesta en una única y prolija fila.