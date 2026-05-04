import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from io import BytesIO
import plotly.express as px
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y ESTILO
# ==========================================
st.set_page_config(page_title="GIG - Simulador UNAD", layout="wide")

st.markdown("""
    <style>
        .stApp { background-color: #FFFFFF; }
        [data-testid="stSidebar"] { background-color: #F8F9FA; border-right: 1px solid #E0E0E0; }
        h1, h2, h3, h4, .stSubheader { color: #005088 !important; font-family: 'Segoe UI', sans-serif; }
        .stButton>button { background-color: #005088; color: white; border-radius: 5px; width: 100%; }
        .stButton>button:hover { background-color: #F8B133; color: #005088; }
    </style>
    """, unsafe_allow_html=True)

# --- RUTA DEL LOGO LOCAL ---
LOGO_PATH = r"D:\Documents\Roger\Universidad\Proyecto de grado ll\DATA GIG\Proyecto aplicado\logo-unad-acreditacion-min.png"

# ==========================================
# 2. INICIALIZACIÓN DE ESTADOS (PERSISTENCIA)
# ==========================================
if 'resultado_prediccion' not in st.session_state:
    st.session_state.resultado_prediccion = None
    st.session_state.historico = None
    st.session_state.metricas = None

# ==========================================
# 3. BARRA LATERAL (SIDEBAR)
# ==========================================
with st.sidebar:
    try:
        st.image(LOGO_PATH, use_container_width=True)
    except:
        st.warning("Logo no encontrado en la ruta local.")
        
    st.markdown("---")
    st.header("⚙️ Configuración")
    uploaded_file = st.file_uploader("Subir Excel data Histórica", type=["xlsx"])
    
    marcas_seleccionadas = []
    if uploaded_file:
        # Carga rápida para extraer marcas disponibles
        df_info_temp = pd.read_excel(uploaded_file, sheet_name='Info_Pdv')
        df_info_temp['marca_id'] = pd.to_numeric(df_info_temp['marca_id'].astype(str).str.strip().str[1:], errors='coerce').fillna(0).astype(int)
        lista_marcas = sorted(df_info_temp['marca_id'].unique())
        marcas_seleccionadas = st.multiselect("Seleccione Marcas a Predecir:", options=lista_marcas, default=lista_marcas[:2])
    
    fecha_proy = st.date_input("Mes a Proyectar", value=pd.to_datetime("2026-03-01"))
    fecha_proy = fecha_proy.replace(day=1)

# ==========================================
# 4. FUNCIONES TÉCNICAS (MOTOR XGBOOST)
# ==========================================
def procesar_proyeccion(file, marcas_a_predecir, fecha_proyeccion):
    # Carga de pestañas
    df_info = pd.read_excel(file, sheet_name='Info_Pdv')
    df_ventas = pd.read_excel(file, sheet_name='Data_Historica')
    df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'], format='%d/%m/%Y', errors='coerce')
    
    # Merge y limpieza de IDs (Quitando la letra inicial si existe)
    df_master = pd.merge(df_ventas, df_info, on='punto_de_venta_id', how='left')
    columnas_id = ['punto_de_venta_id', 'ubicacion_id', 'tipo_pdv_id', 'marca_id', 'categoria_id']
    for col in columnas_id:
        df_master[col] = pd.to_numeric(df_master[col].astype(str).str.strip().str[1:], errors='coerce').fillna(0).astype(int)
    
    # Cálculo de Ticket y Filtro de Robustez (> $5M y > 10 TX)
    df_master['ticket'] = np.where(df_master['transacciones'] > 0, df_master['ventas_monto'] / df_master['transacciones'], 0)
    df_clean = df_master[(df_master['ventas_monto'] > 5000000) & (df_master['transacciones'] > 10)].copy()
    
    # Ingeniería de Features (Lags)
    df_clean = df_clean.sort_values(by=['punto_de_venta_id', 'fecha'])
    df_clean['Mes'] = df_clean['fecha'].dt.month
    for prefijo, col_base in [('ventas', 'ventas_monto'), ('transacciones', 'transacciones'), ('ticket', 'ticket')]:
        df_clean[f'{prefijo}_lag_1'] = df_clean.groupby('punto_de_venta_id')[col_base].shift(1).fillna(0)
    df_clean['ventas_roll_mean_3'] = df_clean.groupby('punto_de_venta_id')['ventas_monto'].transform(lambda x: x.shift(1).rolling(3).mean()).fillna(0)
    
    # Entrenamiento
    df_train = df_clean[df_clean['fecha'] < fecha_proyeccion].dropna()
    df_target = df_clean[df_clean['fecha'] == (pd.to_datetime(fecha_proyeccion) - pd.DateOffset(months=1))].copy()
    df_target['fecha'] = fecha_proyeccion
    df_target = df_target[df_target['marca_id'].isin(marcas_a_predecir)]

    features = ['marca_id', 'categoria_id', 'tipo_pdv_id', 'ubicacion_id', 'Mes', 'ventas_lag_1', 'transacciones_lag_1', 'ticket_lag_1', 'ventas_roll_mean_3']
    targets = {'Ventas': 'ventas_monto', 'Transacciones': 'transacciones', 'Ticket': 'ticket'}
    
    metricas_res = {}
    for nombre, col in targets.items():
        model = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.07, random_state=42)
        model.fit(df_train[features], df_train[col])
        p_train = model.predict(df_train[features])
        metricas_res[nombre] = {
            'MAPE': f"{mean_absolute_percentage_error(df_train[col], p_train):.2%}",
            'R2': f"{r2_score(df_train[col], p_train):.2f}"
        }
        df_target[f'Pred_{nombre}'] = np.maximum(0, model.predict(df_target[features]))
        
    return df_target, df_clean, metricas_res

# ==========================================
# 5. CUERPO PRINCIPAL Y DASHBOARD
# ==========================================
st.title("🚀 Motor de Inferencia: Trinidad Operativa")
st.subheader("Proyecto de Grado - Dashboard Estratégico de Retail")

if uploaded_file and marcas_seleccionadas:
    if st.button("Generar Proyección y Análisis 📈"):
        with st.spinner("Entrenando modelos y calculando crecimientos..."):
            res, hist, mets = procesar_proyeccion(uploaded_file, marcas_seleccionadas, pd.to_datetime(fecha_proy))
            st.session_state.resultado_prediccion = res
            st.session_state.historico = hist
            st.session_state.metricas = mets

    if st.session_state.resultado_prediccion is not None:
        res = st.session_state.resultado_prediccion
        hist = st.session_state.historico
        mets = st.session_state.metricas

        # --- FILTROS GLOBALES DEL DASHBOARD ---
        st.markdown("### 🛠️ Filtros de Visualización")
        f1, f2, f3 = st.columns(3)
        with f1:
            var_viz = st.selectbox("Métrica:", ["Ventas", "Transacciones", "Ticket"])
            col_h = "ventas_monto" if var_viz == "Ventas" else ("transacciones" if var_viz == "Transacciones" else "ticket")
            col_p = f"Pred_{var_viz}"
            agg_f = "mean" if var_viz == "Ticket" else "sum"
        with f2:
            m_ver = st.selectbox("Marca:", options=marcas_seleccionadas)
        with f3:
            dim_seg = st.selectbox("Segmentar por:", ["categoria_id", "ubicacion_id", "tipo_pdv_id"])

# --- BLOQUE 1: CRECIMIENTO Y PERFORMANCE ---
        st.markdown("---")
        
        # Cálculos de Crecimiento
        f_ant = pd.to_datetime(fecha_proy) - pd.DateOffset(months=1)
        f_yoy = pd.to_datetime(fecha_proy) - pd.DateOffset(years=1)
        
        v_actual = res[res['marca_id'] == m_ver][col_p].agg(agg_f)
        v_ant = hist[(hist['marca_id'] == m_ver) & (hist['fecha'] == f_ant)][col_h].agg(agg_f)
        v_yoy = hist[(hist['marca_id'] == m_ver) & (hist['fecha'] == f_yoy)][col_h].agg(agg_f)

        def get_delta(cur, prev):
            return f"{((cur-prev)/prev)*100:.2f}%" if prev and prev > 0 else "N/A"

        # Fila A: Valores de Negocio (Grandes)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric(f"Predicción {var_viz}", f"{v_actual:,.0f}")
        with c2: st.metric("Crecimiento MoM", get_delta(v_actual, v_ant), delta=get_delta(v_actual, v_ant))
        with c3: st.metric("Crecimiento YoY", get_delta(v_actual, v_yoy), delta=get_delta(v_actual, v_yoy))

        # Fila B: Métricas Técnicas (Pequeñas y discretas - ESTO ES LO NUEVO)
        st.markdown(f"<p style='color: #7f8c8d; font-size: 0.8rem; margin-bottom: -10px;'>Fiabilidad del Modelo para {var_viz}:</p>", unsafe_allow_html=True)
        t1, t2, t3, t4 = st.columns([1, 1, 1, 1])
        t1.caption(f"**MAPE:** {mets[var_viz]['MAPE']}")
        t2.caption(f"**R² Score:** {mets[var_viz]['R2']}")

        # --- BLOQUE 2: TENDENCIA HISTÓRICA ---
        st.markdown(f"#### 📈 Tendencia Histórica vs Predicción ({var_viz})")
        df_h_plot = hist[hist['marca_id'] == m_ver].groupby('fecha')[col_h].agg(agg_f).reset_index()
        df_p_plot = pd.DataFrame({'fecha': [pd.to_datetime(fecha_proy)], col_h: [v_actual]})
        
        fig_line = px.line(pd.concat([df_h_plot, df_p_plot]), x='fecha', y=col_h, markers=True, template="plotly_white")
        fig_line.add_scatter(x=df_p_plot['fecha'], y=df_p_plot[col_h], mode='markers', marker=dict(size=15, color='gold'), name='Predicción')
        st.plotly_chart(fig_line, use_container_width=True)

        # --- BLOQUE 3: TRAYECTORIA SEGMENTADA CON FILTRO DINÁMICO ---
        st.markdown(f"#### 🔍 Análisis Detallado por {dim_seg}")
        
        df_dim_h = hist[hist['marca_id'] == m_ver].groupby(['fecha', dim_seg])[col_h].agg(agg_f).reset_index()
        df_dim_p = res[res['marca_id'] == m_ver].groupby(['fecha', dim_seg])[col_p].agg(agg_f).reset_index()
        df_dim_p.rename(columns={col_p: col_h}, inplace=True)
        df_total_dim = pd.concat([df_dim_h, df_dim_p]).sort_values(['fecha', dim_seg])

        # Filtro de selección para no saturar la gráfica
        opciones_dim = sorted(df_total_dim[dim_seg].unique())
        seleccion = st.multiselect(f"Seleccione {dim_seg} para visualizar:", options=opciones_dim, default=opciones_dim[:5])

        if seleccion:
            df_filt = df_total_dim[df_total_dim[dim_seg].isin(seleccion)]
            fig_dim = px.line(df_filt, x='fecha', y=col_h, color=str(dim_seg), markers=True, title=f"Trayectoria de {var_viz}")
            st.plotly_chart(fig_dim, use_container_width=True)
        
        # Tabla de exportación
        st.markdown("#### 📋 Datos Predichos por Punto de Venta")
        st.dataframe(res[res['marca_id'] == m_ver][['punto_de_venta_id', dim_seg, col_p]])
        
        # Botón de descarga
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            res.to_excel(writer, index=False, sheet_name='Proyeccion')
        st.download_button("📥 Descargar Excel", output.getvalue(), f"Proyeccion_{m_ver}_{fecha_proy.strftime('%Y%m')}.xlsx")

else:
    st.info("👋 Bienvenida/o. Por favor carga el archivo Excel y selecciona las marcas para iniciar.")