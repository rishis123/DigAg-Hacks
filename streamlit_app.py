import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import os
from dotenv import load_dotenv

# bring in ML system
from yield_forecast import load_system, make_reading

load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Terra | Early Warning System", layout="wide", page_icon="🌿")

# --- CUSTOM "FARM-TECH" STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f9fbf9; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 20px; border: none; }
    h1 { color: #2E7D32; font-family: 'Segoe UI', sans-serif; }
    .metric-card { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- GEMINI SETUP ---
api_key = os.getenv("API_KEY") # Ensure your .env has API_KEY=...
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('models/gemini-flash-latest')
    st.info("🔌 Gemini API configured")
else:
    st.error("API Key missing! Please set API_KEY in your environment variables.")
    model = None

# --- HEADER ---
col1, col2 = st.columns([1, 4])
with col2:
    st.title("Terra")
    st.subheader("Your AI Agronomist & Early Warning System")

st.divider()

# --- SIDEBAR ---
with st.sidebar:
    st.header("🌿 Farm Settings")
    crop_type = st.selectbox("Select Crop Type", ["Bibb Lettuce", "Spinach", "Roma Tomatoes", "Sweet Basil"])
    
    st.divider()
    st.header("📂 Data Upload")
    uploaded_file = st.file_uploader("Upload ISE Sensor CSV", type="csv")
    
    with st.expander("CSV Format Help"):
        st.write("Required columns for basic dashboard:")
        st.code("Growth Day, Inst [N], Inst [P], Inst [K], Best Fit Plant DM")
        st.write("Additional sensor columns accepted (will be passed to ML models):")
        st.code("Ca_mgl, Mg_mgl, S_mgl, Fe_mgl, pH, temp_air, EC_estimated, etc.")
        st.write("You can load `sample_input.csv` from the repository as an example.")

    st.divider()
    st.header("🧠 Model Configuration")
    short_path = st.text_input("Short horizon model path", "short_horizon_model.pkl")
    long_path = st.text_input("Long horizon model path", "long_horizon_model.pkl")

# --- MAIN CONTENT ---
if not uploaded_file:
    st.markdown("""
        <div style="text-align: center; padding: 40px;">
            <h2 style="color: #2E7D32;">👋 Welcome to Terra</h2>
            <p style="font-size: 1.2rem; color: #555;">Upload your Ion-Selective Electrode (ISE) sensor data to begin.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.image("https://images.squarespace-cdn.com/content/v1/63064607eb816a4d50027fd1/1688398925707-J1DZKYEDFAAPESGIT1KD/eden-green-vertical-produce.jpg", use_container_width=True)
else:
    # Read the data
    df = pd.read_csv(uploaded_file)

    # Simple Validation
    required_cols = ['Growth Day', 'Inst [N]', 'Inst [P]', 'Inst [K]', 'Best Fit Plant DM']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV missing required columns: {', '.join(required_cols)}")
    else:
        # attempt to load ML models
        try:
            system = load_system(short_path, long_path)
            ml_available = True
            st.success("✅ ML models loaded successfully")
        except Exception as e:
            st.warning(f"⚠️ could not load ML models: {e}")
            ml_available = False

        # helper to detect >20% day-over-day changes in numeric columns
        def pct_change_alert(dataframe, threshold=0.2):
            df2 = dataframe.reset_index(drop=True)
            for i in range(1, len(df2)):
                prev = df2.iloc[i-1]
                curr = df2.iloc[i]
                for col in df2.columns:
                    if col == 'Growth Day':
                        continue
                    try:
                        pv = float(prev[col])
                        cv = float(curr[col])
                    except Exception:
                        continue
                    if pv != 0 and abs(cv - pv) / abs(pv) > threshold:
                        return True
            return False

        pct_alert = pct_change_alert(df)
        # specifically check potassium drops
        def find_k_drop(dataframe, threshold=0.2):
            max_drop = 0.0
            triggered = False
            df2 = dataframe.reset_index(drop=True)
            for i in range(1, len(df2)):
                try:
                    prev = float(df2.loc[i-1, 'Inst [K]'])
                    curr = float(df2.loc[i, 'Inst [K]'])
                except Exception:
                    continue
                if prev != 0:
                    drop = (prev - curr) / prev
                    if drop > threshold:
                        triggered = True
                        if drop > max_drop:
                            max_drop = drop
            return triggered, max_drop
        k_alert_trigger, k_alert_drop = find_k_drop(df)

        # show model availability banner up‑front
        status_lines = []
        status_lines.append(f"**ML models:** {'loaded' if ml_available else 'not loaded'}")
        status_lines.append(f"**Gemini:** {'configured' if model is not None else 'not available'}")
        st.markdown('  '.join(status_lines))
        if ml_available:
            st.markdown("**🧠 ML models will be used for all readings.**")
            system.reset()
            preds = []
            cycle_alert = False
            for _, r in df.iterrows():
                reading = make_reading(
                    growth_day=int(r['Growth Day']),
                    N_mgl=r.get('Inst [N]'),
                    P_mgl=r.get('Inst [P]'),
                    K_mgl=r.get('Inst [K]'),
                )
                res = system.process_reading(reading)
                if res.get('alert'):
                    cycle_alert = True
                preds.append(res)
            pred_df = pd.DataFrame(preds)
            df = pd.concat([df.reset_index(drop=True),
                            pred_df[['short_forecast_g','long_forecast_g','alert','threshold_warnings','recommendation']]],
                           axis=1)
            last = preds[-1]
            # field status reflects any alert or hard threshold event
            status = "ACTION REQUIRED" if (cycle_alert or pct_alert or k_alert_trigger) else "OPTIMAL"
            delta = "Warning" if (cycle_alert or pct_alert or k_alert_trigger) else "Healthy"
        else:
            st.markdown("**⚠️ Using simple fallback anomaly logic (models not loaded).**")
            # still check hard threshold across columns or potassium drop
            status = "ACTION REQUIRED" if (pct_alert or k_alert_trigger) else "OPTIMAL"
            delta = "Warning" if (pct_alert or k_alert_trigger) else "Healthy"
            last = {'alert': None}


        latest_k = df['Inst [K]'].iloc[-1]
        prev_k = df['Inst [K]'].iloc[-2]
        # k_pct_change now only used for last-day display; overall drop handled
        k_pct_change = abs(latest_k - prev_k) / prev_k if prev_k != 0 else 0

        # dashboard metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Field Status", status,
                      delta=delta, delta_color="inverse")
        with m2:
            # show red if any k alert triggered during cycle
            if k_alert_trigger:
                st.metric("Potassium (K) Level", f"{latest_k:.1f} mg/L", f"-{k_alert_drop*100:.1f}% drop", delta_color="inverse")
                st.markdown("<span style='color:red;'>⚠︎ Potassium dropped over 20% during cycle</span>", unsafe_allow_html=True)
            else:
                # fall back to last-pair change indicator
                if k_pct_change > 0.2:
                    st.metric("Potassium (K) Level", f"{latest_k:.1f} mg/L", "DANGER", delta_color="inverse")
                else:
                    st.metric("Potassium (K) Level", f"{latest_k:.1f} mg/L", "Stable")
        with m3:
            st.metric("Current Growth Day", int(df['Growth Day'].max()))

        # cycle alert indicator (ML-generated) and hard-threshold alert
        if ml_available and cycle_alert:
            st.error("🚨 ML model generated an alert at some point during this cycle")

        st.divider()

        # graphs
        tab1, tab2 = st.tabs(["📊 Nutrient Dynamics", "🌿 Biomass Growth"])

        with tab1:
            fig_nutrients = px.line(df, x='Growth Day', y=['Inst [N]', 'Inst [P]', 'Inst [K]'],
                                    title="Macronutrient Concentration (mg/L)",
                                    color_discrete_map={'Inst [N]': '#4CAF50',
                                                        'Inst [P]': '#FF9800',
                                                        'Inst [K]': '#2196F3'},
                                    markers=True)
            if ml_available:
                fig_nutrients.add_scatter(x=df['Growth Day'], y=df['short_forecast_g'],
                                          mode='lines+markers', name='Short forecast',
                                          line={'dash':'dash','color':'#9C27B0'})
                fig_nutrients.add_scatter(x=df['Growth Day'], y=df['long_forecast_g'],
                                          mode='lines+markers', name='Long forecast',
                                          line={'dash':'dot','color':'#3F51B5'})
            st.plotly_chart(fig_nutrients, use_container_width=True)

        with tab2:
            fig_mass = px.area(df, x='Growth Day', y='Best Fit Plant DM',
                               title="Daily Dry Matter Accumulation (mg/plant)",
                               color_discrete_sequence=['#81C784'])
            st.plotly_chart(fig_mass, use_container_width=True)

        # show alerts/recommendations
        if last.get('alert'):
            st.warning(f"🛑 ALERT: {last['alert']}")
            if last.get('recommendation'):
                st.info(f"💡 Recommendation: {last['recommendation']}")
        if ml_available and last.get('threshold_warnings'):
            for w in last['threshold_warnings']:
                st.error(f"THRESHOLD: {w}")


        st.divider()
        prompt = f"""
        You are a specialized AI Agronomist.
        Current Crop: {crop_type}
        Sensor Data Summary:
        - Growth Day: {df['Growth Day'].max()}
        - Latest Potassium (K): {latest_k:.2f} mg/L
        {f"- Potassium drop observed: {k_alert_drop*100:.1f}%" if k_alert_trigger else ""}
        - Nitrogen (N): {df['Inst [N]'].iloc[-1]:.2f} mg/L
        - Phosphorus (P): {df['Inst [P]'].iloc[-1]:.2f} mg/L
        Model Alert: {last['alert']}
        {('Recommendation: ' + last.get('recommendation','')) if last.get('recommendation') else ''}
        Provide a brief explanation of the alert, what crop conditions any deficiencies or surplus may cause, possible causes, and immediate corrective actions.
        """

        st.subheader("🩺 Agronomist Diagnosis")
        with st.spinner("Analyzing plant-environment signatures..."):
            if model is not None:
                try:
                    response = model.generate_content(prompt)
                    st.success("🤖 Gemini response received")
                    if response.candidates and response.candidates[0].finish_reason == 3:
                        st.warning("AI diagnosis was filtered. Please check sensor hardware.")
                    else:
                        st.markdown(f"""
                            <div style='background-color: #fff3e0; padding: 20px; border-radius: 10px; color: #5d4037; border-left: 5px solid #ffb74d;'>
                                {response.text}
                            </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Gemini API Error: {e}")
            else:
                st.info("🔒 Gemini not available (API key missing).")
