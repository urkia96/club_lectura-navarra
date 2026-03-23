import streamlit as st
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os
import unicodedata
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import json
import gc
import numpy as np


# --- 0. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Clubes de Lectura de Navarra", layout="wide")

# --- 1. INICIALIZAR ESTADO TEMPRANO ---
if "idioma" not in st.session_state:
    st.session_state.idioma = "Castellano"
if "auth" not in st.session_state:
    st.session_state.auth = False




# --- 1. CONFIGURACIÓN E IDIOMAS ---

PATH_RECO = "recomendador"
URL_LOGO = f"{PATH_RECO}/logo_B. Navarra.jpg"
URL_SERENDIPIA = f"{PATH_RECO}/serendipia.png" 
RUTA_PORTADAS = "portadas"

def normalizar_texto(texto):
    if not isinstance(texto, str): return ""
    texto = "".join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    return texto.lower().strip()



col_main, col_lang = st.columns([12, 1])
with col_lang:
    idioma_actual = st.selectbox("🌐", ["Castellano", "Euskera"], index=0 if st.session_state.idioma == "Castellano" else 1, key="selector_global")
    st.session_state.idioma = idioma_actual

texts = {
    "Castellano": {
        "titulo": "Clubes de Lectura de Navarra", "subtitulo": "Nafarroako Irakurketa Klubak",
        "sidebar_tit": "🎯 Panel de Control", 
        "exp_gral": "⚙️ Filtros generales", "exp_cont": "📖 Filtros de contenido",
        "f_idioma": "🌍 Idioma", "f_publico": "👥 Público",
        "f_genero_aut": "👤 Género Autor/a", "f_editorial": "📚 Editorial", "f_paginas": "📄 Número de páginas",
        "f_local": "🏠 Autores locales", "f_ia_gen": "📂 Género", "f_ia_sub": "🏷️ Subgénero",
        "tab1": "📖 Búsqueda por autor/título", "tab2": "✨ Búsqueda libre", "tab3": "🔍 Lotes similares", "tab4": "🎲 Búsqueda aleatoria",
        "placeholder": "Ej: Novelas sobre la historia de Navarra", "input_query": "Puedes escribir lo que quieras",
        "lote_input": "Introduce el código del lote:", "busq_titulo": "Buscar por Título:", "busq_autor": "Buscar por Autor:",
        "resumen_btn": "Ver resumen", "pags_label": "págs", "thanks": "✅ Voto registrado", "ask": "¿Te gusta esta recomendación?",
        "boton_txt": "¡Sorpréndeme!", "no_results": "Sin resultados con esos filtros."
    },
    "Euskera": {
        "titulo": "Nafarroako Irakurketa Klubak", "subtitulo": "Clubes de Lectura de Navarra",
        "sidebar_tit": "🎯 Kontrol Panela", 
        "exp_gral": "⚙️ Iragazki orokorrak", "exp_cont": "📖 Edukiaren iragazkiak",
        "f_idioma": "🌍 Hizkuntza", "f_publico": "👥 Publikoa",
        "f_genero_aut": "👤 Egilearen generoa", "f_editorial": "📚 Argitaletxea", "f_paginas": "📄 Orrialde kopurua",
        "f_local": "🏠 Bertako autoreak", "f_ia_gen": "📂 Generoa", "f_ia_sub": "🏷️ Azpigeneroa",
        "tab1": "📖 Izenburu / Idazle bilaketa", "tab2": "✨ Bilaketa librea", "tab3": "🔍 Lote antzekoak", "tab4": "🎲 Zorizko bilaketa",
        "placeholder": "Adibidez: Nafarroako historiaren inguruko eleberriak", "input_query": "Nahi duzuna idatzi dezakezu",
        "lote_input": "Sartu lote kodea:", "busq_titulo": "Izenburuaren arabera bilatu:", "busq_autor": "Egilearen arabera bilatu:",
        "resumen_btn": "Ikusi laburpena", "pags_label": "orr", "thanks": "✅ Iritzia gordeta", "ask": "Gogoko duzu?",
        "boton_txt": "Harritu nazazu!", "no_results": "Ez da emaitzarik aurkitu iragazki hauekin."
    }
}
t = texts[st.session_state.idioma]

# --- 2. CARGA DE RECURSOS ---
@st.cache_resource
def load_resources():
    excel_path = f"{PATH_RECO}/CATALOGO_PROCESADO_version3.xlsx"
    if not os.path.exists(excel_path):
        st.error(f"Archivo crítico no encontrado: {excel_path}")
        st.stop()
    
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()
    df['Nº lote'] = df['Nº lote'].astype(str).str.strip()
    
    df['Páginas'] = pd.to_numeric(df['Páginas'], errors='coerce').fillna(0).astype(int)
    cols_check = ['Idioma', 'Público', 'genero_fix', 'Editorial', 'Geografia_Autor', 'Genero_Principal_IA', 'Subgeneros_Limpios_IA']
    for col in cols_check:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'None', '<NA>'], "Desconocido")
        else:
            df[col] = "Desconocido"
            
    df['titulo_norm'] = df['Título'].apply(normalizar_texto)
    df['autor_norm'] = df['Autor'].apply(normalizar_texto)
    
    with open(f"{PATH_RECO}/metadatos_promptss_infloat_ponderado_small.pkl", "rb") as f:
        df_ia_meta = pickle.load(f)
    df_ia_meta['Nº lote'] = df_ia_meta['Nº lote'].astype(str).str.strip()
    
    index = faiss.read_index(f"{PATH_RECO}/biblioteca_prompts_infloat_ponderado_small.index")
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    
    gc.collect() 
    return df, df_ia_meta, index, model

df, df_ia_meta, index, model = load_resources()


# --- 3. FUNCIONES AUXILIARES ---
def conectar_sheets():
    try:
        if "GCP_SERVICE_ACCOUNT" in os.environ and "GSHEET_URL" in os.environ:
            creds_info = json.loads(os.environ["GCP_SERVICE_ACCOUNT"])
            sheet_url = os.environ["GSHEET_URL"]
            creds = Credentials.from_service_account_info(
                creds_info,
                scopes=["https://www.googleapis.com/auth/spreadsheets"]
            )
            gc_client = gspread.authorize(creds)
            sheet = gc_client.open_by_url(sheet_url).sheet1
            st.success("✅ Conectado a Google Sheets correctamente")
            return sheet
        else:
            st.error("❌ Faltan las variables de entorno GCP_SERVICE_ACCOUNT o GSHEET_URL")
    except Exception as e:
        st.error(f"❌ Error conectando a Sheets: {e}")
        return None

def guardar_voto(lote, titulo, valor, query):
    sheet = conectar_sheets()
    if sheet:
        try:
            val_txt = "👍" if valor == 1 else "👎"
            row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(lote), str(titulo), val_txt, str(query)]
            st.write(f"Intentando guardar fila: {row}") 
            sheet.append_row(row)
            st.success(f"{val_txt} ¡Voto registrado!")
        except Exception as e:
            st.error(f"❌ No se pudo guardar el voto: {e}")
    else:
        st.error("❌ No se pudo obtener la hoja de cálculo")

# 4. Mostrar tarjeta
@st.fragment
def mostrar_card(r, context):
    IMG_WIDTH = 160  
    lote_id = str(r.get('Nº lote', '')).strip()
    
    with st.container(border=True):
        # Tres columnas: Imagen | Contenido | Botones
        col_img, col_content, col_vote = st.columns([1, 3, 0.5])

        # --- COLUMNA 1: IMAGEN ---
        with col_img:
            foto_path = None
            if os.path.exists(RUTA_PORTADAS):
                for f in os.listdir(RUTA_PORTADAS):
                    if os.path.splitext(f)[0] == lote_id:
                        foto_path = f"{RUTA_PORTADAS}/{f}"
                        break

            # Imagen escalada, más pequeña
            if foto_path:
                st.image(foto_path, width=IMG_WIDTH)
            else:
                st.markdown("<p style='font-size:30px; text-align:center;'>📖</p>", unsafe_allow_html=True)

            st.caption(f"Lote {lote_id}")

        # --- COLUMNA 2: CONTENIDO ---
        with col_content:
            st.markdown(f"### {r.get('Título','Sin título')}")
            st.write(f"**{r.get('Autor','Autor desconocido')}**")

            # Info adicional
            pags_val = r.get('Páginas', r.get('Páginas_ex','--'))
            try:
                pags_display = str(int(float(pags_val))) if pd.notnull(pags_val) and str(pags_val).replace('.','',1).isdigit() else str(pags_val)
            except:
                pags_display = str(pags_val)
            st.caption(f"{r.get('Editorial','--')} | {r.get('Idioma','--')} | {pags_display} {t['pags_label']} | {r.get('Público','--')}")

            # Subgéneros
            if pd.notnull(r.get('Subgeneros_Limpios_IA')):
                st.write(f"**{r.get('Genero_Principal_IA')}**: {r.get('Subgeneros_Limpios_IA')}")

            # Resumen con expander
            with st.expander(t["resumen_btn"], expanded=False):
                st.write(r.get('Resumen_navarra','No hay resumen disponible.'))
                tags = r.get('IA_Tags','')
                if pd.notnull(tags) and str(tags).strip() != "":
                    st.divider()
                    label = t.get('kw_label', 'Palabras clave')
                    st.write(f"**{label}:** {tags}")

        # --- COLUMNA 3: BOTONES DE VOTO ---
        with col_vote:
            st.markdown("<p style='font-weight:bold;'>¿Es relevante?</p>", unsafe_allow_html=True)
            ctx_id = str(context)[:10].replace(" ","_")
            kv = f"v_{lote_id}_{ctx_id}"

            if kv not in st.session_state:
                if st.button("👍", key=f"u_{lote_id}_{ctx_id}", use_container_width=True):
                    guardar_voto(lote_id, r.get('Título','S/T'), 1, context)
                    st.session_state[kv] = 1
                    st.rerun()
                if st.button("👎", key=f"d_{lote_id}_{ctx_id}", use_container_width=True):
                    guardar_voto(lote_id, r.get('Título','S/T'), 0, context)
                    st.session_state[kv] = 0
                    st.rerun()
            else:
                st.success("✅ Votado")
                
# --- 5. PANEL DE CONTROL ---
st.sidebar.title(t["sidebar_tit"])

# 5.1 FILTROS GENERALES
with st.sidebar.expander(t["exp_gral"], expanded=False):
    f_idioma = st.multiselect(t["f_idioma"], sorted(df['Idioma'].unique()))
    f_publico = st.multiselect(t["f_publico"], sorted(df['Público'].unique()))
    f_gen_aut = st.multiselect(t["f_genero_aut"], sorted(df['genero_fix'].unique()))
    f_editorial = st.multiselect(t["f_editorial"], sorted([e for e in df['Editorial'].unique() if e != "Desconocido"]))
    f_local = st.checkbox(t["f_local"])
    f_paginas = st.slider(t["f_paginas"], 50, 1500, 1500)

# 5.2 FILTROS DE CONTENIDO
with st.sidebar.expander(t["exp_cont"], expanded=False):
    f_ia_gen = st.multiselect(t["f_ia_gen"], sorted([g for g in df['Genero_Principal_IA'].unique() if g != "Desconocido"]))
    f_ia_sub = []
    if f_ia_gen:
        subs = set()
        df[df['Genero_Principal_IA'].isin(f_ia_gen)]['Subgeneros_Limpios_IA'].str.split(',').dropna().apply(lambda x: subs.update([s.strip() for s in x]))
        f_ia_sub = st.multiselect(t["f_ia_sub"], sorted([s for s in list(subs) if s != "Desconocido"]))

def filtrar(dataframe):
    temp = dataframe.copy()
    if f_idioma: temp = temp[temp['Idioma'].isin(f_idioma)]
    if f_publico: temp = temp[temp['Público'].isin(f_publico)]
    if f_gen_aut: temp = temp[temp['genero_fix'].isin(f_gen_aut)]
    if f_local: temp = temp[temp['Geografia_Autor'] == "Local"]
    if f_paginas < 1500: temp = temp[temp['Páginas'] <= f_paginas]
    if f_editorial: temp = temp[temp['Editorial'].isin(f_editorial)]
    if f_ia_gen: temp = temp[temp['Genero_Principal_IA'].isin(f_ia_gen)]
    if f_ia_sub: temp = temp[temp['Subgeneros_Limpios_IA'].apply(lambda x: any(s in str(x) for s in f_ia_sub) if pd.notnull(x) else False)]
    return temp

# --- 6. INTERFAZ ---
col_logo, col_tit = st.columns([1,6])
with col_logo:
    if os.path.exists(URL_LOGO): st.image(URL_LOGO, width=150)
with col_tit:
    st.title(t["titulo"])
    st.caption(t["subtitulo"])

tab1, tab2, tab3, tab4 = st.tabs([t["tab1"], t["tab2"], t["tab3"], t["tab4"]])


# --- TAB1: búsqueda por título/autor ---
with tab1:
    c1,c2 = st.columns(2)
    b_tit = c1.text_input(t["busq_titulo"], key="busq_t_input")
    b_aut = c2.text_input(t["busq_autor"], key="busq_a_input")
    
    if b_tit or b_aut:
        texto_buscado = f"Tit: {b_tit} | Aut: {b_aut}".strip(" | ")
        
        res = filtrar(df)
        if b_tit: res = res[res['titulo_norm'].str.contains(normalizar_texto(b_tit), na=False)]
        if b_aut: res = res[res['autor_norm'].str.contains(normalizar_texto(b_aut), na=False)]
        res = res.dropna(subset=['Título']).reset_index(drop=True)
        
        for _, r in res.head(10).iterrows(): 
            mostrar_card(r, texto_buscado)

# --- TAB2: búsqueda libre con FAISS ---
with tab2:
    q = st.text_input(t["input_query"], placeholder=t["placeholder"], key="txt_libre_80")
    if q:
        vec = model.encode([f"query: {q}"], normalize_embeddings=True).astype('float32')
        D, I = index.search(vec, 50)
        
        # Filtramos los índices donde la distancia (similitud) sea >= 0.8
        indices_validos = I[0][D[0] >= 0.8]
        
        if len(indices_validos) > 0:
            lotes_ia = df_ia_meta.iloc[indices_validos]['Nº lote'].tolist()
            df_base = filtrar(df)
            res_final = df_base[df_base['Nº lote'].isin(lotes_ia)]
            res_final = res_final.set_index('Nº lote').reindex(lotes_ia).dropna(subset=['Título']).reset_index()
            
            for _, r in res_final.head(10).iterrows(): 
                 mostrar_card(r, q)
        else:
            st.warning("No se encontraron resultados con una similitud superior al 80%.")

# --- TAB3: lotes similares (Punto Medio / Multi-lote) ---
with tab3:
    # Ahora permitimos varios lotes separados por comas
    lid_input = st.text_input(t["lote_input"] + " (puedes poner varios lotes separados por comas: 001N,002N...)", key="txt_sim_lote_multi")
    
    if lid_input:
        # 1. Limpiamos y obtenemos la lista de lotes
        lotes_solicitados = [l.strip().upper() for l in lid_input.split(",") if l.strip()]
        
        vectores_para_promediar = []
        lotes_encontrados = []

        # 2. Extraemos los vectores de cada lote del índice
        for lid_clean in lotes_solicitados:
            ref_ia = df_ia_meta[df_ia_meta['Nº lote'] == lid_clean]
            if not ref_ia.empty:
                idx_ia = ref_ia.index[0]
                # reconstruct nos da el vector original del lote
                v_lote = index.reconstruct(int(idx_ia))
                vectores_para_promediar.append(v_lote)
                lotes_encontrados.append(lid_clean)
            else:
                st.warning(f"El lote {lid_clean} no se encuentra en el sistema.")

        if vectores_para_promediar:
            # --- 3. CÁLCULO DEL PUNTO MEDIO (CENTROIDE) ---
            # Promediamos todos los vectores encontrados
            v_ref = np.mean(vectores_para_promediar, axis=0).astype('float32').reshape(1, -1)
            
            # 4. Buscamos en el índice usando el vector "mezclado"
            D, I = index.search(v_ref, 30) # Pedimos más para filtrar los originales
            
            # Filtramos por el umbral del 80% (0.8)
            indices_validos = I[0][D[0] >= 0.8]
            lotes_sim = df_ia_meta.iloc[indices_validos]['Nº lote'].tolist()
            
            df_base = filtrar(df)
            
            # 5. Quitamos los lotes que el usuario ya usó para la búsqueda
            res_sim = df_base[df_base['Nº lote'].isin(lotes_sim) & (~df_base['Nº lote'].isin(lotes_encontrados))]
            
            # Reindexamos para mantener el orden de similitud respecto al punto medio
            lotes_ordenados = [l for l in lotes_sim if l not in lotes_encontrados]
            res_sim = res_sim.set_index('Nº lote').reindex(lotes_ordenados).dropna(subset=['Título']).reset_index()
            
            contexto_voto = f"Punto medio de: {', '.join(lotes_encontrados)}"
            
            if not res_sim.empty:
                st.info(f"Mostrando libros similares a: {', '.join(lotes_encontrados)}")
                for _, r in res_sim.head(10).iterrows(): 
                    mostrar_card(r, contexto_voto)
            else:
                st.warning("No hay otros lotes con más del 80% de similitud con esa combinación.")

# --- TAB4: búsqueda aleatoria ---
with tab4:
    if os.path.exists(URL_SERENDIPIA):
        st.image(URL_SERENDIPIA, use_container_width=False, width=150)
    if st.button(t["boton_txt"]):
        posibles = filtrar(df)
        if not posibles.empty: st.session_state.azar = posibles.sample(1).iloc[0]
    if 'azar' in st.session_state: mostrar_card(st.session_state.azar, "Seren")
