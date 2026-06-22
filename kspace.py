import streamlit as st
import numpy as np
import math
import pandas as pd
import plotly.graph_objects as go
import json

# --- 1. 기본 설정 및 UI 압축 CSS ---
st.set_page_config(page_title="SRG DOE Waveguide 설계툴", layout="wide")

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

# --- 2. 설정 파일 관리 ---
def export_settings():
    settings = {}
    for key, val in st.session_state.items():
        if any(suffix in key for suffix in ["_num", "_slider", "_active"]):
            settings[key] = val
    settings.update({
        "path_choice": st.session_state.get("path_choice"),
        "coord_sys": st.session_state.get("coord_sys"),
        "single_layer_sync": st.session_state.get("single_layer_sync"),
        "m_order_select": st.session_state.get("m_order_select", 1)
    })
    return json.dumps(settings, indent=4)

def import_settings(json_data):
    data = json.loads(json_data)
    for key, val in data.items():
        st.session_state[key] = val
    st.rerun()

# --- 3. 위젯 동기화 로직 ---
def update_sync(k, val):
    if st.session_state.get("single_layer_sync"):
        parts = k.split('_')
        if len(parts) == 2 and parts[0] in ['R', 'G', 'B'] and parts[1] in ['icg', 'epe', 'oc']:
            for color in ['R', 'G', 'B']:
                st.session_state[f"{color}_{parts[1]}_slider"] = val
                st.session_state[f"{color}_{parts[1]}_num"] = val

def update_from_slider(k):
    val = st.session_state[f"{k}_slider"]
    st.session_state[f"{k}_num"] = val
    update_sync(k, val)

def update_from_num(k):
    val = st.session_state[f"{k}_num"]
    st.session_state[f"{k}_slider"] = val
    update_sync(k, val)

def dual_input(label, min_val, max_val, default_val, step, k, fmt=None):
    if f"{k}_slider" not in st.session_state: st.session_state[f"{k}_slider"] = default_val
    if f"{k}_num" not in st.session_state: st.session_state[f"{k}_num"] = default_val
    st.markdown(f"<div style='font-size:11px; margin-top:5px;'>{label}</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([7, 3])
    with col1: st.slider(label, min_val, max_val, key=f"{k}_slider", step=step, on_change=update_from_slider, args=(k,), label_visibility="collapsed", format=fmt)
    with col2: st.number_input(label, min_val, max_val, key=f"{k}_num", step=step, on_change=update_from_num, args=(k,), label_visibility="collapsed", format=fmt)
    return st.session_state[f"{k}_slider"]

def dual_range_input(label, min_val, max_val, default_val, step, k):
    if f"{k}_slider" not in st.session_state:
        st.session_state[f"{k}_slider"] = default_val
        st.session_state[f"{k}_min_num"], st.session_state[f"{k}_max_num"] = default_val[0], default_val[1]
    st.markdown(f"<div style='font-size:11px; margin-top:5px;'>{label}</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([5.4, 2.3, 2.3])
    with c1: st.slider(label, min_val, max_val, key=f"{k}_slider", step=step, on_change=lambda: [st.session_state.update({f"{k}_min_num":st.session_state[f"{k}_slider"][0], f"{k}_max_num":st.session_state[f"{k}_slider"][1]})], label_visibility="collapsed")
    with c2: st.number_input("min", min_val, max_val, key=f"{k}_min_num", step=step, on_change=lambda: [st.session_state.update({f"{k}_slider":(st.session_state[f"{k}_min_num"], st.session_state[f"{k}_max_num"])})], label_visibility="collapsed")
    with c3: st.number_input("max", min_val, max_val, key=f"{k}_max_num", step=step, on_change=lambda: [st.session_state.update({f"{k}_slider":(st.session_state[f"{k}_min_num"], st.session_state[f"{k}_max_num"])})], label_visibility="collapsed")
    return st.session_state[f"{k}_slider"]

# --- 4. 물리 계산 엔진 ---
def get_refractive_index(lam, n_d, V_d):
    B = (n_d - 1) / V_d * (486.1**2 * 656.3**2) / (486.1**2 - 656.3**2) * 1e-6
    return (n_d - B / (589.3/1000)**2) + B / (lam/1000)**2

def calculate_k_space(wl_dict, n_d, V_d, m_order, h_min, h_max, v_min, v_max, t_mm, epd_limit, path_type, a_icg, a_epe, a_oc):
    if not wl_dict["active"]: return None
    lam = wl_dict["lambda"]; n_lam = get_refractive_index(lam, n_d, V_d); k0 = 2 * math.pi / lam; k_wg_max = n_lam * k0
    
    G_ICG_x, G_ICG_y = m_order * (2*math.pi/wl_dict["Lambda_ICG"]) * np.array([math.cos(math.radians(a_icg)), math.sin(math.radians(a_icg))])
    G_EPE_x, G_EPE_y = (0, 0)
    if "Path B" in path_type and wl_dict["Lambda_EPE"]:
        G_EPE_mag = 2 * math.pi / wl_dict["Lambda_EPE"]
        G_EPE_x, G_EPE_y = G_EPE_mag * np.array([math.cos(math.radians(a_epe)), math.sin(math.radians(a_epe))])
    G_OC_x, G_OC_y = (2*math.pi/wl_dict["Lambda_OC"]) * np.array([math.cos(math.radians(a_oc)), math.sin(math.radians(a_oc))])
    
    H_mesh, V_mesh = np.meshgrid(np.radians(np.arange(h_min, h_max + 1, 1)), np.radians(np.arange(v_min, v_max + 1, 1)))
    kx_in, ky_in = k0 * np.sin(H_mesh), k0 * np.sin(V_mesh); kz_in = np.sqrt(np.maximum(k0**2 - kx_in**2 - ky_in**2, 0))
    
    kx_icg, ky_icg = kx_in + G_ICG_x, ky_in + G_ICG_y; kz_icg = np.sqrt(np.maximum(k_wg_max**2 - kx_icg**2 - ky_icg**2, 0))
    tir_mask_icg = ((kx_icg**2 + ky_icg**2) > k0**2) & ((kx_icg**2 + ky_icg**2) < k_wg_max**2)
    
    kx_epe, ky_epe = kx_icg + G_EPE_x, ky_icg + G_EPE_y; kz_epe = np.sqrt(np.maximum(k_wg_max**2 - kx_epe**2 - ky_epe**2, 0))
    tir_mask_epe = ((kx_epe**2 + ky_epe**2) > k0**2) & ((kx_epe**2 + ky_epe**2) < k_wg_max**2) if "Path B" in path_type else np.ones_like(tir_mask_icg, dtype=bool)
    
    kx_oc, ky_oc = kx_epe + G_OC_x, ky_epe + G_OC_y; kz_oc = np.sqrt(np.maximum(k0**2 - kx_oc**2 - ky_oc**2, 0))
    tir_mask_oc = ((kx_oc**2 + ky_oc**2) <= k0**2)
    
    hop_dist = 2 * t_mm * (np.sqrt(kx_epe**2 + ky_epe**2) / np.maximum(kz_epe, 1e-10))
    mask_0, mask_1, mask_2, mask_3 = np.ones_like(kx_in, dtype=bool), tir_mask_icg, tir_mask_icg & tir_mask_epe, tir_mask_icg & tir_mask_epe & tir_mask_oc & (hop_dist <= epd_limit)
    
    center_idx = (np.abs(H_mesh) < 1e-5) & (np.abs(V_mesh) < 1e-5); c_ray = None
    if np.any(center_idx):
        c_ray = {
            "kx_in": kx_in[center_idx][0], "ky_in": ky_in[center_idx][0], "kz_in": kz_in[center_idx][0],
            "kx_icg": kx_icg[center_idx][0], "ky_icg": ky_icg[center_idx][0], "kz_icg": kz_icg[center_idx][0],
            "kx_epe": kx_epe[center_idx][0], "ky_epe": ky_epe[center_idx][0], "kz_epe": kz_epe[center_idx][0],
            "kx_oc": kx_oc[center_idx][0], "ky_oc": ky_oc[center_idx][0], "kz_oc": kz_oc[center_idx][0]
        }
        
    return {"color": wl_dict["color"], "k0": k0, "k_wg_max": k_wg_max, "kx_in": kx_in, "ky_in": ky_in, "kz_in": kz_in, "kx_icg": kx_icg, "ky_icg": ky_icg, "kz_icg": kz_icg, "kx_epe": kx_epe, "ky_epe": ky_epe, "kz_epe": kz_epe, "kx_oc": kx_oc, "ky_oc": ky_oc, "kz_oc": kz_oc, "mask_0": mask_0, "mask_1": mask_1, "mask_2": mask_2, "mask_3": mask_3, "hop_distance": hop_dist, "H_mesh": np.degrees(H_mesh), "V_mesh": np.degrees(V_mesh), "c_ray": c_ray, "G_ICG_x": G_ICG_x, "G_EPE_x": G_EPE_x, "G_OC_x": G_OC_x}

# --- 5. 사이드바 UI ---
st.sidebar.markdown("### 💾 설정 관리")
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1: st.download_button("설정 저장", data=export_settings(), file_name="waveguide_config.json", use_container_width=True)
with col_s2: 
    uploaded = st.sidebar.file_uploader("불러오기", type="json", label_visibility="collapsed")
    if uploaded: st.sidebar.button("적용", on_click=import_settings, args=(uploaded.getvalue().decode("utf-8"),), use_container_width=True)

st.sidebar.markdown("---")
path_choice = st.sidebar.radio("그레이팅 경로", ["Path A (ICG→OC)", "Path B (ICG→EPE→OC)"], index=1, horizontal=True, key="path_choice")
single_layer_sync = st.sidebar.checkbox("Single Layer 모드 (RGB 동기화)", value=True, key="single_layer_sync")
coord_sys = st.sidebar.radio("K-Space 좌표계", ["절대 파수 (nm⁻¹)", "정규화 파수 (Direction Cosine)"], index=1, horizontal=True, key="coord_sys")

st.sidebar.markdown("---")
# [수정] 굴절률 입력부에 실수형 포맷 '%.2f' 명시적 적용
n_d_in = dual_input("기본 굴절률 (n at 589nm)", 1.0, 3.0, 1.75, 0.01, "n_d", "%.2f")
abbe_v_in = dual_input("아베수 (Abbe Vd)", 10.0, 100.0, 35.0, 1.0, "abbe_v", "%.1f")
thickness_in = dual_input("현재 두께 t (mm)", 0.1, 3.0, 0.40, 0.01, "thickness", "%.2f") 
epd_val_in = dual_input("라이트엔진 EPD (mm)", 1.0, 50.0, 3.5, 0.1, "epd_val", "%.1f") 
h_fov = dual_range_input("H FOV 범위 (°)", -60, 60, (-30, 30), 1, "h_fov")
v_fov = dual_range_input("V FOV 범위 (°)", -60, 60, (-20, 20), 1, "v_fov")
m_ord = st.sidebar.selectbox("주 회절 차수 (m)", [1, -1, 2, -2], index=st.session_state.get("m_order_idx", 0), key="m_order_select")

st.sidebar.markdown("---")
# [수정] 각도 파라미터들의 스텝을 0.01로 세분화하고 소수점 2자리 포맷('%.2f') 지정
angle_icg = dual_input("ICG 벡터 방향 (°)", 0.0, 360.0, 0.0, 0.01, "angle_icg", "%.2f")
angle_epe = dual_input("EPE 벡터 방향 (°)", 0.0, 360.0, 240.0, 0.01, "angle_epe", "%.2f") if "Path B" in path_choice else 0.0
angle_oc = dual_input("OC 벡터 방향 (°)", 0.0, 360.0, 120.0, 0.01, "angle_oc", "%.2f")

st.sidebar.markdown("---")
def get_wl_inputs(name, def_l, def_p, n_d, V_d):
    act = st.sidebar.checkbox(f"{name} 활성화", value=True, key=f"{name}_active")
    if act:
        with st.sidebar.container():
            st.markdown(f"**[{name}] 세부 설정**")
            wl = dual_input("λ (nm)", 400, 750, def_l, 1, f"{name}_wl")
            limit_p = wl / get_refractive_index(wl, n_d, V_d)
            icg_p = dual_input("Λ_ICG (nm)", 100, 1000, def_p, 1, f"{name}_icg")
            if icg_p < limit_p: st.error(f"⚠️ 회절한계: 최소 {limit_p:.1f}nm 필요")
            else: st.caption(f"물리적 한계 주기: {limit_p:.1f}nm")
            epe_p = dual_input("Λ_EPE (nm)", 100, 1000, icg_p if single_layer_sync else def_p, 1, f"{name}_epe") if "Path B" in path_choice else None
            oc_p = dual_input("Λ_OC (nm)", 100, 1000, icg_p if single_layer_sync else def_p, 1, f"{name}_oc")
            return {"active": True, "lambda": wl, "Lambda_ICG": icg_p, "Lambda_EPE": epe_p, "Lambda_OC": oc_p, "color": name}
    return {"active": False}

wl_R = get_wl_inputs("R", 638, 300, n_d_in, abbe_v_in)
wl_G = get_wl_inputs("G", 520, 300, n_d_in, abbe_v_in)
wl_B = get_wl_inputs("B", 450, 300, n_d_in, abbe_v_in)

# --- 6. 분석 로직 실행 ---
results = {}
for data in [wl_R, wl_G, wl_B]:
    res = calculate_k_space(data, n_d_in, abbe_v_in, m_ord, h_fov[0], h_fov[1], v_fov[0], v_fov[1], thickness_in, epd_val_in, path_choice, angle_icg, angle_epe, angle_oc)
    if res: results[data["color"]] = res

# --- 7. 메인 뷰 시각화 ---
st.title("SRG DOE Waveguide 프리검토 분석")

if not results:
    st.warning("활성화된 파장이 없습니다.")
else:
    viz_options = list(results.keys())
    if len(results) > 1: viz_options.append("RGB 통합 뷰 (Overlap)")
    tab_xy, tab_xz, tab_sweep = st.tabs(["K-Space (XY 평면)", "K-Space (XZ 단면)", "두께(t) 스윕 분석"])
    c_map = {"R": "red", "G": "green", "B": "blue"}

    # --- XY 평면 탭 ---
    with tab_xy:
        target = st.selectbox("XY 시각화 대상", viz_options, index=len(viz_options)-1, key="xy_sel")
        fig_xy = go.Figure(); max_k = 0; common_mask = None
        plots = list(results.values()) if target == "RGB 통합 뷰 (Overlap)" else [results[target]]
        
        for r in plots:
            cn, pc = r["color"], c_map[r["color"]]; sf = r["k0"] if coord_sys == "정규화 파수 (Direction Cosine)" else 1.0
            max_k = max(max_k, r["k_wg_max"]/sf)
            fig_xy.add_shape(type="circle", x0=-r["k0"]/sf, y0=-r["k0"]/sf, x1=r["k0"]/sf, y1=r["k0"]/sf, line_color=pc, line_dash="dash", opacity=0.4)
            fig_xy.add_shape(type="circle", x0=-r["k_wg_max"]/sf, y0=-r["k_wg_max"]/sf, x1=r["k_wg_max"]/sf, y1=r["k_wg_max"]/sf, line_color=pc, fillcolor=pc, opacity=0.03)
            m0, m1, m2, m3 = r["mask_0"], r["mask_1"], r["mask_2"], r["mask_3"]
            
            if common_mask is None: common_mask = m3.copy()
            else: common_mask &= m3

            fig_xy.add_trace(go.Scatter(x=r["kx_in"][m0]/sf, y=r["ky_in"][m0]/sf, mode="markers", marker=dict(size=2, color=pc, symbol="circle-open", opacity=0.1), name=f"{cn} Input", hoverinfo="skip"))
            fig_xy.add_trace(go.Scatter(x=r["kx_icg"][m1]/sf, y=r["ky_icg"][m1]/sf, mode="markers", marker=dict(size=3, color=pc, symbol="square", opacity=0.3), name=f"{cn} Coupled", hoverinfo="skip"))
            if "Path B" in path_choice:
                fig_xy.add_trace(go.Scatter(x=r["kx_epe"][m2]/sf, y=r["ky_epe"][m2]/sf, mode="markers", marker=dict(size=3, color=pc, symbol="diamond", opacity=0.5), name=f"{cn} EPE", hoverinfo="skip"))
            ht = [f"H:{h:.0f} V:{v:.0f}<br>Hop:{hp:.2f}mm<br>Overlap:{(epd_val_in-hp):.2f}mm" for h,v,hp in zip(r["H_mesh"][m3], r["V_mesh"][m3], r["hop_distance"][m3])]
            fig_xy.add_trace(go.Scatter(x=r["kx_oc"][m3]/sf, y=r["ky_oc"][m3]/sf, mode="markers", marker=dict(size=4, color=pc, symbol="circle", opacity=0.9), name=f"{cn} Output", text=ht, hoverinfo="text"))

            if (target != "RGB 통합 뷰 (Overlap)") or (cn == "G"):
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
        
        st.subheader("📊 유효 FOV 및 마진 요약")
        summary_table = {}
        for c, r in results.items():
            mask = r['mask_3']
            if np.any(mask):
                vh, vv, vhop = r['H_mesh'][mask], r['V_mesh'][mask], r['hop_distance'][mask]
                max_h = np.max(vhop)
                summary_table[c] = {"H-FOV": f"{np.min(vh):.0f}°~{np.max(vh):.0f}°", "V-FOV": f"{np.min(vv):.0f}°~{np.max(vv):.0f}°", "Max Hop": f"{max_h:.2f}mm", "Min Overlap": f"{(epd_val_in - max_h):.2f}mm", "Pass": f"{(np.sum(mask)/mask.size*100):.1f}%"}
            else:
                summary_table[c] = {"H-FOV": "None", "V-FOV": "None", "Max Hop": "-", "Min Overlap": "-", "Pass": "0%"}
        
        if common_mask is not None and np.any(common_mask):
            ref_r = results[list(results.keys())[0]] 
            ch, cv = ref_r["H_mesh"][common_mask], ref_r["V_mesh"][common_mask]
            summary_table["Common"] = {"H-FOV": f"{np.min(ch):.0f}°~{np.max(ch):.0f}°", "V-FOV": f"{np.min(cv):.0f}°~{np.max(cv):.0f}°", "Max Hop": "-", "Min Overlap": "-", "Pass": f"{(np.sum(common_mask)/common_mask.size*100):.1f}%"}
            
        st.table(pd.DataFrame(summary_table))

    # --- XZ 단면 탭 ---
    with tab_xz:
        target_xz = st.selectbox("XZ 시각화 대상", viz_options, index=len(viz_options)-1, key="xz_sel")
        fig_xz = go.Figure(); max_kz = 0
        plots_xz = list(results.values()) if target_xz == "RGB 통합 뷰 (Overlap)" else [results[target_xz]]
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
                fig_xz.add_trace(go.Scatter(x=[c["kx_in"]/sf, c["kx_oc"]/sf], y=[c["kz_in"]/sf, c["kz_oc"]/sf], mode="markers", marker=dict(size=10, color=pc, symbol="star"), name=f"{cn} Ray Path"))
        fig_xz.update_layout(title=f"K-Space XZ - 중심 시야각(0°) 궤적 ({coord_sys})", xaxis=dict(range=[-max_kz*1.1, max_kz*1.1], scaleanchor="y"), yaxis=dict(range=[0, max_kz*1.1]), height=600, plot_bgcolor="white")
        st.plotly_chart(fig_xz, use_container_width=True)

    # --- 스윕 분석 탭 ---
    with tab_sweep:
        st.markdown("#### 두께 스윕 분석 (EPD 기반 탈락량 확인)")
        c1, c2, c3 = st.columns(3); ts, te, tp = c1.number_input("시작", 0.1, 3.0, 0.2), c2.number_input("종료", 0.1, 3.0, 1.0), c3.number_input("스텝", 0.01, 1.0, 0.05)
        if st.button("스윕 실행", use_container_width=True):
            t_arr = np.arange(ts, te+1e-9, tp); res_l = []; pb = st.progress(0)
            for i, t in enumerate(t_arr):
                d = {"t": round(t,3)}; cm = None
                for k, wd in [("R",wl_R),("G",wl_G),("B",wl_B)]:
                    if wd["active"]:
                        rt = calculate_k_space(wd, n_d_in, abbe_v_in, m_ord, h_fov[0], h_fov[1], v_fov[0], v_fov[1], t, epd_val_in, path_choice, angle_icg, angle_epe, angle_oc); mt = rt["mask_3"]
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
            fs.update_layout(title="두께 변화에 따른 H/V FOV Span 추이", xaxis_title="Thickness (mm)", yaxis_title="Span (°)")
            st.plotly_chart(fs, use_container_width=True); st.dataframe(df.style.format("{:.2f}"), use_container_width=True)
