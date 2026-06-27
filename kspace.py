import streamlit as st
import numpy as np
import math
import pandas as pd
import plotly.graph_objects as go
import json

# --- 1. Base Layout & Compressed UI CSS ---
st.set_page_config(page_title="SRG DOE Waveguide Design Tool", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem !important; padding-left: 1rem !important; padding-right: 1rem !important; }
    [data-testid="stSidebar"] .st-emotion-cache-1wivap2, [data-testid="stSidebar"] .st-emotion-cache-1ob1npu { gap: 0.4rem !important; }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p, [data-testid="stSidebar"] .stMarkdown p { font-size: 13px !important; margin-bottom: 0px !important; font-weight: 600 !important; }
    [data-testid="stSidebar"] .stNumberInput input { font-size: 13px !important; height: 28px !important; padding: 0px 5px !important; }
    [data-testid="stSidebar"] .stSlider { margin-bottom: -15px !important; }
    [data-testid="stSidebar"] .stCheckbox p, [data-testid="stSidebar"] .stRadio p { font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. Configuration Management ---
def export_settings():
    settings = {}
    for key, val in st.session_state.items():
        if any(suffix in key for suffix in ["_num", "_slider", "_active"]):
            settings[key] = val
    settings.update({
        "path_choice": st.session_state.get("path_choice"),
        "coord_sys": st.session_state.get("coord_sys"),
        "single_layer_sync": st.session_state.get("single_layer_sync"),
        "m_order_select": st.session_state.get("m_order_select", 1),
        "auto_oc_angle": st.session_state.get("auto_oc_angle", False),
        "oc_find_mode": st.session_state.get("oc_find_mode", "Specular Target")
    })
    return json.dumps(settings, indent=4)

def import_settings(json_data):
    data = json.loads(json_data)
    for key, val in data.items():
        st.session_state[key] = val
    st.rerun()

# --- 3. Widget Synchronization Logic ---
def update_sync(k, val):
    if st.session_state.get("single_layer_sync"):
        parts = k.split('_')
        if len(parts) == 2 and parts[0] in ['R', 'G', 'B'] and parts[1] in ['icg', 'epe', 'oc', 'efficg', 'effepe', 'effoc']:
            for color in ['R', 'G', 'B']:
                st.session_state[f"{color}_{parts[1]}_slider"] = float(val)
                st.session_state[f"{color}_{parts[1]}_num"] = float(val)

def update_from_slider(k):
    val = st.session_state[f"{k}_slider"]
    st.session_state[f"{k}_num"] = float(val)
    update_sync(k, val)

def update_from_num(k):
    val = st.session_state[f"{k}_num"]
    st.session_state[f"{k}_slider"] = float(val)
    update_sync(k, val)

def dual_input(label, min_val, max_val, default_val, step, k, fmt=None, sidebar=False):
    if f"{k}_slider" not in st.session_state: st.session_state[f"{k}_slider"] = float(default_val)
    if f"{k}_num" not in st.session_state: st.session_state[f"{k}_num"] = float(default_val)
    target = st.sidebar if sidebar else st
    target.markdown(f"<div style='font-size:11px; margin-top:5px;'>{label}</div>", unsafe_allow_html=True)
    col1, col2 = target.columns([7, 3])
    with col1: st.slider(label, float(min_val), float(max_val), key=f"{k}_slider", step=float(step), on_change=update_from_slider, args=(k,), label_visibility="collapsed", format=fmt)
    with col2: st.number_input(label, float(min_val), float(max_val), key=f"{k}_num", step=float(step), on_change=update_from_num, args=(k,), label_visibility="collapsed", format=fmt)
    return st.session_state[f"{k}_slider"]

def dual_range_input(label, min_val, max_val, default_val, step, k):
    if f"{k}_slider" not in st.session_state:
        st.session_state[f"{k}_slider"] = (float(default_val[0]), float(default_val[1]))
        st.session_state[f"{k}_min_num"], st.session_state[f"{k}_max_num"] = float(default_val[0]), float(default_val[1])
    st.markdown(f"<div style='font-size:11px; margin-top:5px;'>{label}</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([5.4, 2.3, 2.3])
    with c1: st.slider(label, float(min_val), float(max_val), key=f"{k}_slider", step=float(step), on_change=lambda: [st.session_state.update({f"{k}_min_num":float(st.session_state[f"{k}_slider"][0]), f"{k}_max_num":float(st.session_state[f"{k}_slider"][1])})], label_visibility="collapsed")
    with c2: st.number_input("min", float(min_val), float(max_val), key=f"{k}_min_num", step=float(step), on_change=lambda: [st.session_state.update({f"{k}_slider":(float(st.session_state[f"{k}_min_num"]), float(st.session_state[f"{k}_max_num"]))})], label_visibility="collapsed", format="%.2f")
    with c3: st.number_input("max", float(min_val), float(max_val), key=f"{k}_max_num", step=float(step), on_change=lambda: [st.session_state.update({f"{k}_slider":(float(st.session_state[f"{k}_min_num"]), float(st.session_state[f"{k}_max_num"]))})], label_visibility="collapsed", format="%.2f")
    return st.session_state[f"{k}_slider"]

# --- 4. Computational Physics Engine ---
def get_refractive_index(lam, n_d, V_d):
    B = (n_d - 1) / V_d * (486.1**2 * 656.3**2) / (486.1**2 - 656.3**2) * 1e-6
    return (n_d - B / (589.3/1000)**2) + B / (lam/1000)**2

def calculate_k_space(wl_dict, n_d, V_d, m_order, h_min, h_max, v_min, v_max, t_mm, epd_limit, path_type, a_icg, a_epe, a_oc, le_tilt_x, le_tilt_y, auto_oc_flag, oc_find_mode, custom_out_x, custom_out_y):
    if not wl_dict["active"]: return None
    lam = wl_dict["lambda"]; n_lam = get_refractive_index(lam, n_d, V_d); k0 = 2 * math.pi / lam; k_wg_max = n_lam * k0
    
    # 1. ICG Grating Vector Setup
    G_ICG_x, G_ICG_y = m_order * (2*math.pi/wl_dict["Lambda_ICG"]) * np.array([math.cos(math.radians(a_icg)), math.sin(math.radians(a_icg))])
    
    # 2. EPE Grating Vector Setup
    G_EPE_x, G_EPE_y = (0.0, 0.0)
    if "Path B" in path_type and wl_dict["Lambda_EPE"]:
        G_EPE_x, G_EPE_y = (2*math.pi/wl_dict["Lambda_EPE"]) * np.array([math.cos(math.radians(a_epe)), math.sin(math.radians(a_epe))])
        
    # 부호 보정 적용 (지면 방향 = 음수 매칭 완료)
    k_x_center_in = k0 * math.sin(math.radians(0.0 - le_tilt_x))
    k_y_center_in = k0 * math.sin(math.radians(0.0 - le_tilt_y))
    
    k_x_center_epe = k_x_center_in + G_ICG_x + G_EPE_x
    k_y_center_epe = k_y_center_in + G_ICG_y + G_EPE_y
    
    if auto_oc_flag:
        if oc_find_mode == "Specular Target":
            k_x_target = -k_x_center_in
            k_y_target = -k_y_center_in
        else:
            k_x_target = k0 * math.sin(math.radians(custom_out_x))
            k_y_target = k0 * math.sin(math.radians(custom_out_y))
            
        G_OC_x = k_x_target - k_x_center_epe
        G_OC_y = k_y_target - k_y_center_epe
        G_OC_mag = math.sqrt(G_OC_x**2 + G_OC_y**2)
        if G_OC_mag > 0:
            wl_dict["Lambda_OC"] = (2 * math.pi) / G_OC_mag
            a_oc = np.degrees(math.atan2(G_OC_y, G_OC_x)) % 360.0
        else:
            G_OC_x, G_OC_y = (2*math.pi/wl_dict["Lambda_OC"]) * np.array([math.cos(math.radians(a_oc)), math.sin(math.radians(a_oc))])
    else:
        G_OC_x, G_OC_y = (2*math.pi/wl_dict["Lambda_OC"]) * np.array([math.cos(math.radians(a_oc)), math.sin(math.radians(a_oc))])
        
    H_mesh, V_mesh = np.meshgrid(np.radians(np.arange(h_min, h_max + 0.1, 0.5)), np.radians(np.arange(v_min, v_max + 0.1, 0.5)))
    
    kx_in = k0 * np.sin(H_mesh - math.radians(le_tilt_x))
    ky_in = k0 * np.sin(V_mesh - math.radians(le_tilt_y))
    kz_in = np.sqrt(np.maximum(k0**2 - kx_in**2 - ky_in**2, 0))
    
    kx_icg, ky_icg = kx_in + G_ICG_x, ky_in + G_ICG_y; kz_icg = np.sqrt(np.maximum(k_wg_max**2 - kx_icg**2 - ky_icg**2, 0))
    tir_mask_icg = ((kx_icg**2 + ky_icg**2) > k0**2) & ((kx_icg**2 + ky_icg**2) < k_wg_max**2)
    
    kx_epe, ky_epe = kx_icg + G_EPE_x, ky_icg + G_EPE_y; kz_epe = np.sqrt(np.maximum(k_wg_max**2 - kx_epe**2 - ky_epe**2, 0))
    tir_mask_epe = ((kx_epe**2 + ky_epe**2) > k0**2) & ((kx_epe**2 + ky_epe**2) < k_wg_max**2) if "Path B" in path_type else np.ones_like(tir_mask_icg, dtype=bool)
    
    kx_oc, ky_oc = kx_epe + G_OC_x, ky_epe + G_OC_y; kz_oc = np.sqrt(np.maximum(k0**2 - kx_oc**2 - ky_oc**2, 0))
    tir_mask_oc = ((kx_oc**2 + ky_oc**2) <= k0**2)
    
    hop_dist = 2 * t_mm * (np.sqrt(kx_epe**2 + ky_epe**2) / np.maximum(kz_epe, 1e-10))
    mask_0, mask_1, mask_2, mask_3 = np.ones_like(kx_in, dtype=bool), tir_mask_icg, tir_mask_icg & tir_mask_epe, tir_mask_icg & tir_mask_epe & tir_mask_oc & (hop_dist <= epd_limit)
    
    c_idx_v = np.argmin(np.abs(np.degrees(V_mesh[:, 0])))
    c_idx_h = np.argmin(np.abs(np.degrees(H_mesh[0, :])))
    
    c_ray = {
        "kx_in": kx_in[c_idx_v, c_idx_h], "ky_in": ky_in[c_idx_v, c_idx_h], "kz_in": kz_in[c_idx_v, c_idx_h],
        "kx_icg": kx_icg[c_idx_v, c_idx_h], "ky_icg": ky_icg[c_idx_v, c_idx_h], "kz_icg": kz_icg[c_idx_v, c_idx_h],
        "kx_epe": kx_epe[c_idx_v, c_idx_h], "ky_epe": ky_epe[c_idx_v, c_idx_h], "kz_epe": kz_epe[c_idx_v, c_idx_h],
        "kx_oc": kx_oc[c_idx_v, c_idx_h], "ky_oc": ky_oc[c_idx_v, c_idx_h], "kz_oc": kz_oc[c_idx_v, c_idx_h]
    }
        
    return {"color": wl_dict["color"], "k0": k0, "k_wg_max": k_wg_max, "kx_in": kx_in, "ky_in": ky_in, "kz_in": kz_in, "kx_icg": kx_icg, "ky_icg": ky_icg, "kz_icg": kz_icg, "kx_epe": kx_epe, "ky_epe": ky_epe, "kz_epe": kz_epe, "kx_oc": kx_oc, "ky_oc": ky_oc, "kz_oc": kz_oc, "mask_0": mask_0, "mask_1": mask_1, "mask_2": mask_2, "mask_3": mask_3, "hop_distance": hop_dist, "H_mesh": np.degrees(H_mesh), "V_mesh": np.degrees(V_mesh), "c_ray": c_ray, "G_ICG_x": G_ICG_x, "G_EPE_x": G_EPE_x, "G_OC_x": G_OC_x, "Lambda_ICG": wl_dict["Lambda_ICG"], "Lambda_EPE": wl_dict["Lambda_EPE"], "Lambda_OC": wl_dict["Lambda_OC"], "eff_icg": wl_dict["eff_icg"], "eff_epe": wl_dict["eff_epe"], "eff_oc": wl_dict["eff_oc"], "calculated_a_oc": a_oc}

# --- 5. Session State Initialization for Cache Sync ---
if "srg_cached_results" not in st.session_state: st.session_state["srg_cached_results"] = None
if "auto_oc_pitch_val" not in st.session_state: st.session_state["auto_oc_pitch_val"] = 300.0
if "auto_oc_vector_ang" not in st.session_state: st.session_state["auto_oc_vector_ang"] = 120.0

# --- 6. Sidebar Layout UI Panel Tree ---
st.sidebar.markdown("### 💾 Configuration Management")
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1: st.sidebar.download_button("Save Setup", data=export_settings(), file_name="waveguide_config.json", use_container_width=True)
with col_s2: 
    uploaded = st.sidebar.file_uploader("Load Setup", type="json", label_visibility="collapsed")
    if uploaded: st.sidebar.button("Apply Settings", on_click=import_settings, args=(uploaded.getvalue().decode("utf-8"),), use_container_width=True)

st.sidebar.markdown("---")
path_choice = st.sidebar.radio("Grating Path Type", ["Path A (ICG→OC)", "Path B (ICG→EPE→OC)"], index=1, horizontal=True, key="path_choice")
single_layer_sync = st.sidebar.checkbox("Single Layer Mode (RGB Sync)", value=True, key="single_layer_sync")
coord_sys = st.sidebar.radio("K-Space Coordinates System", ["Absolute Wavevector (nm⁻¹)", "Normalized Wavevector (Direction Cosine)"], index=1, horizontal=True, key="coord_sys")

st.sidebar.markdown("---")
st.sidebar.markdown("**⚙️ Light Engine Alignment Tilt Panel**")
le_tilt_x = dual_input("LE Horizontal Incident Angle θ_x (°)", -30.0, 30.0, 0.0, 0.1, "le_tilt_x", "%.1f", sidebar=True)
le_tilt_y = dual_input("LE Vertical Incident Angle θ_y (° -:Ground, +:Sky)", -30.0, 30.0, 0.0, 0.1, "le_tilt_y", "%.1f", sidebar=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**📐 Hardware Glass Properties**")
n_d_in = dual_input("Substrate Index (n at 589nm)", 1.0, 3.0, 1.75, 0.01, "n_d", "%.2f")
abbe_v_in = dual_input("Abbe Number (Vd)", 10.0, 100.0, 35.0, 0.1, "abbe_v", "%.1f")
thickness_in = dual_input("Substrate Thickness t (mm)", 0.1, 3.0, 0.40, 0.01, "thickness", "%.2f") 
epd_val_in = dual_input("Light Engine EPD Size (mm)", 1.0, 50.0, 3.5, 0.1, "epd_val", "%.1f") 
h_fov = dual_range_input("Horizontal FOV Bounds (°)", -60, 60, (-30, 30), 0.01, "h_fov")
v_fov = dual_range_input("Vertical FOV Bounds (°)", -60, 60, (-20, 20), 0.01, "v_fov")
m_ord = st.sidebar.selectbox("Diffraction Order (m)", [1, -1, 2, -2], index=1, key="m_order_select")

st.sidebar.markdown("---")
st.sidebar.markdown("**📐 Out-Coupler Aperture Form Factor**")
oc_width = dual_input("OC Aperture Width (mm)", 5.0, 100.0, 30.0, 0.1, "oc_width", "%.1f", sidebar=True)
oc_height = dual_input("OC Aperture Height (mm)", 5.0, 100.0, 20.0, 0.1, "oc_height", "%.1f", sidebar=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**🔮 Grating Vector Rotation Alignment**")
angle_icg = dual_input("ICG Vector Angle (°)", 0.0, 360.0, 0.0, 0.01, "angle_icg", "%.2f", sidebar=True)
angle_epe = dual_input("EPE Vector Angle (°)", 0.0, 360.0, 240.0, 0.01, "angle_epe", "%.2f", sidebar=True) if "Path B" in path_choice else 0.0

auto_oc_angle = st.sidebar.checkbox("Auto-find OC Angle Mode", value=False, key="auto_oc_angle")

oc_find_mode = "Specular Target"
custom_out_x = 0.0
custom_out_y = 0.0

if auto_oc_angle:
    oc_find_mode = st.sidebar.radio("OC Auto-Find Target Condition", ["Specular Target", "Custom Output Angle"], index=0, key="oc_find_mode")
    if oc_find_mode == "Custom Output Angle":
        st.sidebar.markdown("*🎯 Target Out-Coupling Angle Setup*")
        custom_out_x = dual_input("Target Horizontal Out Angle (°)", -30.0, 30.0, 0.0, 0.1, "custom_out_x", "%.1f", sidebar=True)
        custom_out_y = dual_input("Target Vertical Out Angle (°)", -30.0, 30.0, 0.0, 0.1, "custom_out_y", "%.1f", sidebar=True)
        
    angle_oc = st.session_state["auto_oc_vector_ang"]
    auto_oc_line_ang = (angle_oc + 90.0) % 180.0
    theme_color = "#4b96ff" if oc_find_mode == "Specular Target" else "#ff964b"
    st.sidebar.markdown(f"""
    <div style="background-color:rgba(75, 150, 255, 0.08); border:1px solid {theme_color}; padding:8px; border-radius:4px; margin-top:5px; margin-bottom:5px;">
        <div style="font-size:11px; color:{theme_color}; font-weight:bold;">🔒 OC Auto-Tracking Active ({oc_find_mode})</div>
        <div style="font-size:12px; font-weight:600; margin-top:3px;">• Calculated Λ_OC: {st.session_state["auto_oc_pitch_val"]:.2f} nm</div>
        <div style="font-size:12px; font-weight:600;">• Vector Angle: {angle_oc:.2f}°</div>
        <div style="font-size:12px; font-weight:600;">• Line Angle: {auto_oc_line_ang:.2f}°</div>
    </div>
    """, unsafe_allow_html=True)
else:
    angle_oc = dual_input("OC Vector Angle (°)", 0.0, 360.0, 120.0, 0.01, "angle_oc", "%.2f", sidebar=True)

st.sidebar.markdown("---")
def get_wl_inputs(name, def_l, def_p, n_d, V_d):
    act = st.sidebar.checkbox(f"Enable Wavelength {name}", value=True, key=f"{name}_active")
    if act:
        with st.sidebar.container():
            st.markdown(f"**[{name} Channel] Parameters**")
            wl = dual_input("λ Wavelength (nm)", 400.0, 750.0, def_l, 0.01, f"{name}_wl", "%.2f", sidebar=True)
            limit_p = wl / get_refractive_index(wl, n_d, V_d)
            icg_p = dual_input("Λ_ICG Period (nm)", 100.0, 1000.0, def_p, 0.01, f"{name}_icg", "%.2f", sidebar=True)
            if icg_p < limit_p: st.error(f"⚠️ Limit Error: Min {limit_p:.2f}nm required")
            else: st.caption(f"Physical Limit Grating Pitch: {limit_p:.2f}nm")
            epe_p = dual_input("Λ_EPE Period (nm)", 100.0, 1000.0, icg_p if single_layer_sync else def_p, 0.01, f"{name}_epe", "%.2f", sidebar=True) if "Path B" in path_choice else None
            
            if auto_oc_angle:
                oc_p = st.session_state["auto_oc_pitch_val"]
            else:
                oc_p = dual_input("Λ_OC Period (nm)", 100.0, 1000.0, icg_p if single_layer_sync else def_p, 0.01, f"{name}_oc", "%.2f", sidebar=True)
            
            st.markdown("*Diffraction Efficiencies (0.00 ~ 1.00)*")
            eff_icg = dual_input("ICG Efficiency", 0.00, 1.00, 0.30, 0.01, f"{name}_efficg", "%.2f", sidebar=True)
            eff_epe = dual_input("EPE Efficiency", 0.00, 1.00, 0.20, 0.01, f"{name}_effepe", "%.2f", sidebar=True) if "Path B" in path_choice else 1.00
            eff_oc = dual_input("OC Efficiency", 0.00, 1.00, 0.40, 0.01, f"{name}_effoc", "%.2f", sidebar=True)
            
            return {"active": True, "lambda": wl, "Lambda_ICG": icg_p, "Lambda_EPE": epe_p, "Lambda_OC": oc_p, "eff_icg": eff_icg, "eff_epe": eff_epe, "eff_oc": eff_oc, "color": name}
    return {"active": False}

wl_R = get_wl_inputs("R", 638.0, 300.0, n_d_in, abbe_v_in)
wl_G = get_wl_inputs("G", 520.0, 300.0, n_d_in, abbe_v_in)
wl_B = get_wl_inputs("B", 450.0, 300.0, n_d_in, abbe_v_in)

st.sidebar.markdown("---")
run_simulation_trigger = st.sidebar.button(label="▶ Run Simulation", use_container_width=True)

# --- 7. Synchronized Processing Pipeline Core Trigger ---
if run_simulation_trigger:
    results = {}
    for data in [wl_R, wl_G, wl_B]:
        res = calculate_k_space(data, n_d_in, abbe_v_in, m_ord, h_fov[0], h_fov[1], v_fov[0], v_fov[1], thickness_in, epd_val_in, path_choice, angle_icg, angle_epe, angle_oc, le_tilt_x, le_tilt_y, auto_oc_angle, oc_find_mode, custom_out_x, custom_out_y)
        if res:
            results[data["color"]] = res
            if auto_oc_angle:
                st.session_state["auto_oc_pitch_val"] = res["Lambda_OC"]
                st.session_state["auto_oc_vector_ang"] = res["calculated_a_oc"]
    st.session_state["srg_cached_results"] = results
    st.rerun()

# --- 8. Main Dashboard Visualization View ---
results = st.session_state["srg_cached_results"]

if results is not None:
    st.title("SRG DOE Waveguide Simulation Dashboard")
    viz_options = list(results.keys())
    if len(results) > 1: viz_options.append("RGB Overlap View")
    tab_xy, tab_xz, tab_sweep = st.tabs(["K-Space (XY Layout)", "K-Space (XZ Profile Cross-Section)", "Thickness (t) Margin Sweep Analysis"])
    c_map = {"R": "red", "G": "green", "B": "blue"}

    # --- XY Plane Tab Layout Panel ---
    with tab_xy:
        target = st.selectbox("XY Visualization Target", viz_options, index=len(viz_options)-1, key="xy_sel")
        fig_xy = go.Figure(); max_k = 0; common_mask = None
        plots = list(results.values()) if target == "RGB Overlap View" else [results[target]]
        
        for r in plots:
            cn, pc = r["color"], c_map[r["color"]]; sf = r["k0"] if coord_sys == "정규화 파수 (Direction Cosine)" else 1.0
            max_k = max(max_k, r["k_wg_max"]/sf)
            fig_xy.add_shape(type="circle", x0=-r["k0"]/sf, y0=-r["k0"]/sf, x1=r["k0"]/sf, y1=r["k0"]/sf, line_color=pc, line_dash="dash", opacity=0.4)
            fig_xy.add_shape(type="circle", x0=-r["k_wg_max"]/sf, y0=-r["k_wg_max"]/sf, x1=r["k_wg_max"]/sf, y1=r["k_wg_max"]/sf, line_color=pc, fillcolor=pc, opacity=0.03)
            m0, m1, m2, m3 = r["mask_0"], r["mask_1"], r["mask_2"], r["mask_3"]
            
            if common_mask is None: common_mask = m3.copy()
            else: common_mask &= m3

            fig_xy.add_trace(go.Scatter(x=r["kx_in"][m0]/sf, y=r["ky_in"][m0]/sf, mode="markers", marker=dict(size=2, color=pc, symbol="circle-open", opacity=0.1), name=f"{cn} Input FOV", hoverinfo="skip"))
            fig_xy.add_trace(go.Scatter(x=r["kx_icg"][m1]/sf, y=r["ky_icg"][m1]/sf, mode="markers", marker=dict(size=3, color=pc, symbol="square", opacity=0.3), name=f"{cn} Coupled TIR", hoverinfo="skip"))
            if "Path B" in path_choice:
                fig_xy.add_trace(go.Scatter(x=r["kx_epe"][m2]/sf, y=r["ky_epe"][m2]/sf, mode="markers", marker=dict(size=3, color=pc, symbol="diamond", opacity=0.5), name=f"{cn} EPE Extracted", hoverinfo="skip"))
            ht = [f"H:{h:.2f} V:{v:.2f}<br>Hop:{hp:.2f}mm<br>Overlap:{(epd_val_in-hp):.2f}mm" for h,v,hp in zip(r["H_mesh"][m3], r["V_mesh"][m3], r["hop_distance"][m3])]
            fig_xy.add_trace(go.Scatter(x=r["kx_oc"][m3]/sf, y=r["ky_oc"][m3]/sf, mode="markers", marker=dict(size=4, color=pc, symbol="circle", opacity=0.9), name=f"{cn} Final Out", text=ht, hoverinfo="text"))

            if (target != "RGB Overlap View") or (cn == "G"):
                if r["c_ray"]:
                    c = r["c_ray"]
                    pts = [[c["kx_in"], c["ky_in"]], [c["kx_icg"], c["ky_icg"]]]
                    if "Path B" in path_choice: pts.append([c["kx_epe"], c["ky_epe"]])
                    pts.append([c["kx_oc"], c["ky_oc"]])
                    for i in range(len(pts)-1):
                        fig_xy.add_annotation(x=pts[i+1][0]/sf, y=pts[i+1][1]/sf, ax=pts[i][0]/sf, ay=pts[i][1]/sf, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor=pc, opacity=0.8)
        
        lim = max_k * 1.1
        fig_xy.update_layout(xaxis=dict(range=[-lim, lim], scaleanchor="y", scaleratio=1), yaxis=dict(range=[-lim, lim]), width=800, height=800, plot_bgcolor="white")
        st.plotly_chart(fig_xy, use_container_width=True)
        
        # --- Grating Specifications Table Engine ---
        st.subheader("📊 Grating Specifications & Orientation Structural Mapping Summary")
        summary_table = {}
        area_m2 = (oc_width * 1e-3) * (oc_height * 1e-3)
        
        for c, r in results.items():
            mask = r['mask_3']
            icg_line_ang = (angle_icg + 90.0) % 180.0
            epe_line_ang = (angle_epe + 90.0) % 180.0
            oc_line_ang = (r["calculated_a_oc"] + 90.0) % 180.0
            
            if np.any(mask):
                vh, vv, vhop = r['H_mesh'][mask], r['V_mesh'][mask], r['hop_distance'][mask]
                max_h = np.max(vhop)
                h_span_rad = np.radians(np.max(vh) - np.min(vh))
                v_span_rad = np.radians(np.max(vv) - np.min(vv))
                omega = 4.0 * math.asin(math.sin(h_span_rad / 2.0) * math.sin(v_span_rad / 2.0)) if (h_span_rad > 0 and v_span_rad > 0) else 1e-6
                
                c_ray = r["c_ray"]
                k_rho = math.sqrt(c_ray["kx_icg"]**2 + c_ray["ky_icg"]**2)
                k_z = c_ray["kz_icg"]
                
                prop_distance_mm = 40.0 
                num_bounces = prop_distance_mm / (2.0 * thickness_in * (k_rho / max(1e-10, k_z)))
                bulk_path_length_mm = prop_distance_mm / (k_rho / max(1e-10, r["k0"] * n_d_in))
                
                tir_loss_factor = (0.998 ** max(1.0, num_bounces)) * (0.999 ** bulk_path_length_mm)
                lumen_out = 1.0 * r["eff_icg"] * r["eff_epe"] * r["eff_oc"] * tir_loss_factor
                nits_per_lumen = lumen_out / (area_m2 * omega) if omega > 0 else 0.0
                
                summary_table[c] = {
                    "ICG Pitch / Line Angle": f"{r['Lambda_ICG']:.1f}nm / {icg_line_ang:.1f}°",
                    "EPE Pitch / Line Angle": f"{r['Lambda_EPE']:.1f}nm / {epe_line_ang:.1f}°" if epe_line_ang else "-",
                    "OC Pitch / Line Angle": f"{r['Lambda_OC']:.1f}nm / {oc_line_ang:.1f}°",
                    "Effective H-FOV": f"{np.min(vh):.1f}°~{np.max(vh):.1f}°", 
                    "Effective V-FOV": f"{np.min(vv):.1f}°~{np.max(vv):.1f}°", 
                    "FOV Pass Ratio": f"{(np.sum(mask)/mask.size*100):.1f}%",
                    "System Efficiency (nits/lm)": f"{nits_per_lumen:,.0f}"
                }
            else:
                summary_table[c] = {
                    "ICG Pitch / Line Angle": f"{r['Lambda_ICG']:.1f}nm / {icg_line_ang:.1f}°",
                    "EPE Pitch / Line Angle": f"{r['Lambda_EPE']:.1f}nm / {epe_line_ang:.1f}°" if epe_line_ang else "-",
                    "OC Pitch / Line Angle": f"{r['Lambda_OC']:.1f}nm / {oc_line_ang:.1f}°",
                    "Effective H-FOV": "None", "Effective V-FOV": "None", "FOV Pass Ratio": "0%", "System Efficiency (nits/lm)": "0"
                }
        
        if common_mask is not None and np.any(common_mask):
            ref_r = results[list(results.keys())[0]] 
            ch, cv = ref_r["H_mesh"][common_mask], ref_r["V_mesh"][common_mask]
            c_nits = sum([float(summary_table[col]["System Efficiency (nits/lm)"].replace(',', '')) for col in results.keys()]) / len(results)
            
            summary_table["RGB Common"] = {
                "ICG Pitch / Line Angle": "-", "EPE Pitch / Line Angle": "-", "OC Pitch / Line Angle": "-",
                "Effective H-FOV": f"{np.min(ch):.1f}°~{np.max(ch):.1f}°", 
                "Effective V-FOV": f"{np.min(cv):.1f}°~{np.max(cv):.1f}°", 
                "FOV Pass Ratio": f"{(np.sum(common_mask)/common_mask.size*100):.1f}%",
                "System Efficiency (nits/lm)": f"{c_nits:,.0f}"
            }
        st.table(pd.DataFrame(summary_table))

    # --- XZ Section Tab Layout Panel ---
    with tab_xz:
        target_xz = st.selectbox("XZ Visualization Target", viz_options, index=len(viz_options)-1, key="xz_sel")
        fig_xz = go.Figure(); max_kz = 0
        plots_xz = list(results.values()) if target_xz == "RGB Overlap View" else [results[target_xz]]
        arc_ang = np.linspace(0, np.pi, 100)
        for r in plots_xz:
            cn, pc = r["color"], c_map[r["color"]]; sf = r["k0"] if coord_sys == "정규화 파수 (Direction Cosine)" else 1.0
            max_kz = max(max_kz, r["k_wg_max"]/sf)
            fig_xz.add_trace(go.Scatter(x=(r['k0']/sf)*np.cos(arc_ang), y=(r['k0']/sf)*np.sin(arc_ang), mode="lines", line=dict(color=pc, dash="dash"), showlegend=False))
            fig_xz.add_trace(go.Scatter(x=(r['k_wg_max']/sf)*np.cos(arc_ang), y=(r['k_wg_max']/sf)*np.sin(arc_ang), mode="lines", line=dict(color=pc, width=1), fill='tonexty', fillcolor=f"rgba({255 if cn=='R' else 0},{255 if cn=='G' else 0},{255 if cn=='B' else 0},0.04)", showlegend=False))
            if r["c_ray"]:
                c = r["c_ray"]
                fig_xz.add_annotation(x=c["kx_icg"]/sf, y=c["kz_in"]/sf, ax=c["kx_in"]/sf, ay=c["kz_in"]/sf, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor=pc, text="ICG")
                fig_xz.add_trace(go.Scatter(x=[c["kx_icg"]/sf, c["kx_icg"]/sf], y=[c["kz_in"]/sf, c["kz_icg"]/sf], mode="lines", line=dict(color=pc, dash="dot"), showlegend=False))
                curr_kx, curr_kz = c["kx_icg"], c["kz_icg"]
                if "Path B" in path_choice and r["G_EPE_x"] != 0:
                    fig_xz.add_annotation(x=c["kx_epe"]/sf, y=c["kz_icg"]/sf, ax=c["kx_icg"]/sf, ay=c["kz_icg"]/sf, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor=pc, text="EPE")
                    fig_xz.add_trace(go.Scatter(x=[c["kx_epe"]/sf, c["kx_epe"]/sf], y=[c["kz_icg"]/sf, c["kz_epe"]/sf], mode="lines", line=dict(color=pc, dash="dot"), showlegend=False))
                    curr_kx, curr_kz = c["kx_epe"], c["kz_epe"]
                fig_xz.add_annotation(x=c["kx_oc"]/sf, y=curr_kz/sf, ax=curr_kx/sf, ay=curr_kz/sf, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor="purple", text="OC")
                fig_xz.add_trace(go.Scatter(x=[c["kx_oc"]/sf, c["kx_oc"]/sf], y=[curr_kz/sf, c["kz_oc"]/sf], mode="lines", line=dict(color="purple", dash="dot"), showlegend=False))
                fig_xz.add_trace(go.Scatter(x=[c["kx_in"]/sf, c["kx_oc"]/sf], y=[c["kz_in"]/sf, c["kz_oc"]/sf], mode="markers", marker=dict(size=10, color=pc, symbol="star"), name=f"{cn} Central Field Ray"))
        fig_xz.update_layout(title=f"K-Space XZ Cross-Section View - Central Gaze Path ({coord_sys})", xaxis=dict(range=[-max_kz*1.1, max_kz*1.1], scaleanchor="y"), yaxis=dict(range=[0, max_kz*1.1]), height=600, plot_bgcolor="white")
        st.plotly_chart(fig_xz, use_container_width=True)

    # --- Thickness Sweep Tab Layout Panel ---
    with tab_sweep:
        st.markdown("#### Substrate Thickness Sweep Panel (Optimization Loop)")
        c1, c2, c3 = st.columns(3); ts, te, tp = c1.number_input("Start Thickness", 0.1, 3.0, 0.2), c2.number_input("End Thickness", 0.1, 3.0, 1.0), c3.number_input("Step", 0.01, 1.0, 0.05)
        if st.button("Run Sweep Simulation Loop", use_container_width=True):
            t_arr = np.arange(ts, te+1e-9, tp); res_l = []; pb = st.progress(0)
            for i, t in enumerate(t_arr):
                d = {"t": round(t,3)}; cm = None
                for k, wd in [("R",wl_R),("G",wl_G),("B",wl_B)]:
                    if wd["active"]:
                        rt = calculate_k_space(wd, n_d_in, abbe_v_in, m_ord, h_fov[0], h_fov[1], v_fov[0], v_fov[1], t, epd_val_in, path_choice, angle_icg, angle_epe, angle_oc, le_tilt_x, le_tilt_y, auto_oc_angle, oc_find_mode, custom_out_x, custom_out_y); mt = rt["mask_3"]
                        if cm is None: cm = mt.copy()
                        else: cm &= mt
                        vh, vv = rt["H_mesh"][mt], rt["V_mesh"][mt]
                        d[f"{k} H-Span"] = np.max(vh)-np.min(vh) if np.any(mt) else 0
                        d[f"{k} V-Span"] = np.max(vv)-np.min(vv) if np.any(mt) else 0
                
                if cm is not None and all([wl_R['active'], wl_G['active'], wl_B['active']]):
                    ref_r = results[list(results.keys())[0]]
                    ch, cv = ref_r["H_mesh"][cm], ref_r["V_mesh"][cm]
                    d["Common H-Span"] = np.max(ch)-np.min(ch) if np.any(cm) else 0
                    d["Common V-Span"] = np.max(cv)-np.min(cv) if np.any(cm) else 0
                res_l.append(d); pb.progress((i+1)/len(t_arr))
            
            df = pd.DataFrame(res_l); fs = go.Figure()
            for col in df.columns[1:]:
                lc = "black" if "Common" in col else ("red" if "R" in col else ("green" if "G" in col else "blue"))
                ls = "solid" if "H-Span" in col else "dot"
                fs.add_trace(go.Scatter(x=df["t"], y=df[col], mode="lines+markers", name=col, line=dict(color=lc, width=2, dash=ls)))
            fs.update_layout(title="H/V FOV Span Variations across Waveguide Substrate Thickness Bounds", xaxis_title="Thickness (mm)", yaxis_title="Span (°)")
            st.plotly_chart(fs, use_container_width=True); st.dataframe(df.style.format("{:.2f}"), use_container_width=True)
else:
    st.info("💡 사이드바의 설정을 조율하신 후 최하단의 [▶ Run Simulation] 버튼을 클릭하면 정밀 K-space 수치 해석 도면이 활성화됩니다.")
