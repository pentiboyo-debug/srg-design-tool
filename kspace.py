import streamlit as st
import numpy as np
import math
import pandas as pd
import plotly.graph_objects as go
import json
import os
import urllib.request
import urllib.error

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

# --- 2. AI API 호출 함수 (urllib 내장 라이브러리 사용 — 별도 패키지 불필요) ---
def call_claude_api(api_key: str, prompt: str,
                    model: str = "claude-sonnet-4-6",
                    max_tokens: int = 2000) -> str:
    """
    anthropic 패키지 없이 urllib 만으로 Anthropic API 호출.
    표준 라이브러리만 사용하므로 추가 설치 불필요.
    """
    url     = "https://api.anthropic.com/v1/messages"
    payload = json.dumps({
        "model":      model,
        "max_tokens": max_tokens,
        "messages":   [{"role": "user", "content": prompt}]
    }).encode("utf-8")
    headers = {
        "Content-Type":      "application/json",
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
    }
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["content"][0]["text"]
    except urllib.error.HTTPError as e:
        body = json.loads(e.read().decode("utf-8"))
        msg  = body.get("error", {}).get("message", str(e))
        raise RuntimeError(f"HTTP {e.code}: {msg}")


def call_gemini_api(api_key: str, prompt: str,
                    model: str = "gemini-2.0-flash",
                    max_output_tokens: int = 2000) -> str:
    """
    urllib만으로 Google Gemini API 호출.
    모델명이 지원되지 않으면 여러 후보 모델/엔드포인트로 자동 재시도한다.
    """
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_output_tokens}
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    model_candidates = []
    if model:
        model_candidates.append(model)
    model_candidates.extend([
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ])

    seen = set()
    for candidate in model_candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        for version in ["v1", "v1beta"]:
            url = f"https://generativelanguage.googleapis.com/{version}/models/{candidate}:generateContent?key={api_key}"
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    return result["candidates"][0]["content"]["parts"][0]["text"]
            except urllib.error.HTTPError as e:
                body = json.loads(e.read().decode("utf-8"))
                msg = body.get("error", {}).get("message", str(e))
                if e.code == 404:
                    continue
                raise RuntimeError(f"HTTP {e.code}: {msg}")
            except Exception as e:
                raise RuntimeError(str(e))

    raise RuntimeError("사용 가능한 Gemini 모델을 찾지 못했습니다. Google AI Studio에서 API 키와 모델 접근 권한을 확인해주세요.")


def get_secret_value(key_name: str, fallback: str = "") -> str:
    """
    환경변수 -> Streamlit Secrets -> 기본값 순으로 API 키 조회.
    """
    if os.environ.get(key_name):
        return os.environ[key_name]
    try:
        return st.secrets[key_name]
    except Exception:
        return fallback

# --- 3. Configuration Management ---
def export_settings():
    settings = {}
    for key, val in st.session_state.items():
        if any(suffix in key for suffix in ["_num", "_slider", "_active"]):
            settings[key] = val
    settings.update({
        "path_choice":       st.session_state.get("path_choice"),
        "grating_face_mode": st.session_state.get("grating_face_mode", "Transmission (Front Face)"),
        "coord_sys":         st.session_state.get("coord_sys"),
        "single_layer_sync": st.session_state.get("single_layer_sync"),
        "m_order_select":    st.session_state.get("m_order_select", 1),
        "auto_oc_angle":     st.session_state.get("auto_oc_angle", False),
        "oc_find_mode":      st.session_state.get("oc_find_mode", "Specular Target"),
    })
    return json.dumps(settings, indent=4)

def import_settings(json_data):
    data = json.loads(json_data)
    for key, val in data.items():
        st.session_state[key] = val
    st.rerun()

# --- 3. Widget Synchronization Logic ---
# [FIX] update_sync: k는 "_slider"/"_num" 접미사가 없는 기본 키로 전달됨.
#        이전 코드는 k.endswith("_icg_slider") 등을 비교했으나 실제 k = "R_icg" 형태여서
#        조건이 절대 True가 되지 않아 동기화가 완전히 무력화되어 있었음. 수정 완료.
def update_sync(k, val):
    if st.session_state.get("single_layer_sync"):
        # [FIX] angle_icg / angle_epe / angle_oc 키가 각각 _icg / _epe / _oc 로 끝나서
        #        파장별 그레이팅 주기 동기화 조건에 오매칭되는 버그 수정.
        #        동기화는 반드시 R_ / G_ / B_ 로 시작하는 파장채널 키에만 적용.
        if not (k.startswith("R_") or k.startswith("G_") or k.startswith("B_")):
            return
        for suffix in ['icg', 'epe', 'oc', 'efficg', 'effepe', 'effoc']:
            if k.endswith(f"_{suffix}"):
                for color in ['R', 'G', 'B']:
                    st.session_state[f"{color}_{suffix}_slider"] = float(val)
                    st.session_state[f"{color}_{suffix}_num"]    = float(val)
                break

def update_from_slider(k):
    if f"{k}_slider" not in st.session_state:
        st.session_state[f"{k}_slider"] = 0.0
    if f"{k}_num" not in st.session_state:
        st.session_state[f"{k}_num"] = 0.0
    val = st.session_state[f"{k}_slider"]
    st.session_state[f"{k}_num"] = float(val)
    update_sync(k, val)


def update_from_num(k):
    if f"{k}_num" not in st.session_state:
        st.session_state[f"{k}_num"] = 0.0
    if f"{k}_slider" not in st.session_state:
        st.session_state[f"{k}_slider"] = 0.0
    val = st.session_state[f"{k}_num"]
    st.session_state[f"{k}_slider"] = float(val)
    update_sync(k, val)


def _clamp_float(value, min_val, max_val, default_val):
    try:
        val = float(value)
    except Exception:
        return float(default_val)
    return min(max(val, float(min_val)), float(max_val))


def _sanitize_dual_input_state(k, min_val, max_val, default_val):
    slider_key = f"{k}_slider"
    num_key = f"{k}_num"

    if slider_key not in st.session_state or st.session_state[slider_key] is None:
        st.session_state[slider_key] = float(default_val)
    else:
        st.session_state[slider_key] = _clamp_float(st.session_state[slider_key], min_val, max_val, default_val)

    if num_key not in st.session_state or st.session_state[num_key] is None:
        st.session_state[num_key] = float(default_val)
    else:
        st.session_state[num_key] = _clamp_float(st.session_state[num_key], min_val, max_val, default_val)


def _sanitize_dual_range_state(k, min_val, max_val, default_val):
    slider_key = f"{k}_slider"
    min_key = f"{k}_min_num"
    max_key = f"{k}_max_num"

    if slider_key not in st.session_state or st.session_state[slider_key] is None:
        st.session_state[slider_key] = (float(default_val[0]), float(default_val[1]))
    else:
        value = st.session_state[slider_key]
        if not (isinstance(value, (list, tuple)) and len(value) == 2):
            st.session_state[slider_key] = (float(default_val[0]), float(default_val[1]))
        else:
            st.session_state[slider_key] = (
                _clamp_float(value[0], min_val, max_val, default_val[0]),
                _clamp_float(value[1], min_val, max_val, default_val[1])
            )

    if min_key not in st.session_state or st.session_state[min_key] is None:
        st.session_state[min_key] = float(default_val[0])
    else:
        st.session_state[min_key] = _clamp_float(st.session_state[min_key], min_val, max_val, default_val[0])

    if max_key not in st.session_state or st.session_state[max_key] is None:
        st.session_state[max_key] = float(default_val[1])
    else:
        st.session_state[max_key] = _clamp_float(st.session_state[max_key], min_val, max_val, default_val[1])

    if st.session_state[min_key] > st.session_state[max_key]:
        st.session_state[min_key], st.session_state[max_key] = st.session_state[max_key], st.session_state[min_key]
        st.session_state[slider_key] = (
            st.session_state[min_key], st.session_state[max_key]
        )


# [FIX] dual_input: col1.slider / col2.number_input 으로 명시 호출하여 버전 독립성 확보
def dual_input(label, min_val, max_val, default_val, step, k, fmt=None, sidebar=False):
    _sanitize_dual_input_state(k, min_val, max_val, default_val)
    target = st.sidebar if sidebar else st
    target.markdown(f"<div style='font-size:11px; margin-top:5px;'>{label}</div>", unsafe_allow_html=True)
    col1, col2 = target.columns([7, 3])
    col1.slider(
        label, float(min_val), float(max_val),
        key=f"{k}_slider", step=float(step),
        on_change=update_from_slider, args=(k,),
        label_visibility="collapsed", format=fmt
    )
    col2.number_input(
        label, float(min_val), float(max_val),
        key=f"{k}_num", step=float(step),
        on_change=update_from_num, args=(k,),
        label_visibility="collapsed", format=fmt
    )
    return st.session_state[f"{k}_slider"]

def _update_range_from_slider(k):
    if f"{k}_slider" not in st.session_state:
        return
    value = st.session_state[f"{k}_slider"]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        st.session_state[f"{k}_min_num"] = float(value[0])
        st.session_state[f"{k}_max_num"] = float(value[1])


def _update_slider_from_range(k):
    min_key = f"{k}_min_num"
    max_key = f"{k}_max_num"
    if min_key not in st.session_state or max_key not in st.session_state:
        return
    st.session_state[f"{k}_slider"] = (
        float(st.session_state[min_key]),
        float(st.session_state[max_key])
    )


def dual_range_input(label, min_val, max_val, default_val, step, k):
    _sanitize_dual_range_state(k, min_val, max_val, default_val)
    st.markdown(f"<div style='font-size:11px; margin-top:5px;'>{label}</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([5.4, 2.3, 2.3])
    c1.slider(
        label, float(min_val), float(max_val),
        key=f"{k}_slider", step=float(step),
        on_change=_update_range_from_slider, args=(k,),
        label_visibility="collapsed"
    )
    c2.number_input(
        "min", float(min_val), float(max_val),
        key=f"{k}_min_num", step=float(step),
        on_change=_update_slider_from_range, args=(k,),
        label_visibility="collapsed", format="%.2f"
    )
    c3.number_input(
        "max", float(min_val), float(max_val),
        key=f"{k}_max_num", step=float(step),
        on_change=_update_slider_from_range, args=(k,),
        label_visibility="collapsed", format="%.2f"
    )
    return st.session_state[f"{k}_slider"]

# --- 4. Computational Physics Engine ---
# [FIX] Cauchy 2항 모델 + Abbe수 표준 μm 단위계로 완전 정형화
# V_d = (n_d-1)/(n_F-n_C),  n(λ)=A+B/λ² (λ:μm)
# → B = (n_d-1)/V_d / (1/λ_F²-1/λ_C²),  A = n_d - B/λ_d²
def get_refractive_index(lam, n_d, V_d):
    lam_F, lam_d, lam_C = 0.4861, 0.5893, 0.6563   # Fraunhofer lines (μm)
    B = ((n_d - 1.0) / V_d) / (1.0 / lam_F**2 - 1.0 / lam_C**2)
    A = n_d - B / lam_d**2
    return A + B / (lam / 1000.0)**2                 # lam: nm → μm

def calculate_k_space(
    wl_dict, n_d, V_d, m_order,
    h_min, h_max, v_min, v_max,
    t_mm, epd_limit, path_type, grating_face_mode, oc_width_val,
    a_icg, a_epe, a_oc,
    le_tilt_x, le_tilt_y,
    auto_oc_flag, oc_find_mode, custom_out_x, custom_out_y
):
    if not wl_dict["active"]:
        return None

    lam      = wl_dict["lambda"]
    n_lam    = get_refractive_index(lam, n_d, V_d)
    k0       = 2 * math.pi / lam
    k_wg_max = n_lam * k0

    # Grating vectors
    G_ICG_x = m_order * (2 * math.pi / wl_dict["Lambda_ICG"]) * math.cos(math.radians(a_icg))
    G_ICG_y = m_order * (2 * math.pi / wl_dict["Lambda_ICG"]) * math.sin(math.radians(a_icg))
    G_EPE_x, G_EPE_y = 0.0, 0.0
    if "Path B" in path_type and wl_dict["Lambda_EPE"]:
        G_EPE_x = (2 * math.pi / wl_dict["Lambda_EPE"]) * math.cos(math.radians(a_epe))
        G_EPE_y = (2 * math.pi / wl_dict["Lambda_EPE"]) * math.sin(math.radians(a_epe))

    # Center ray k-vector (for auto-OC)
    if grating_face_mode == "Reflection (Back Face)":
        k_x_center_in = k0 * math.sin(math.radians(le_tilt_x)) / n_lam
        k_y_center_in = k0 * math.sin(math.radians(le_tilt_y)) / n_lam
    else:
        k_x_center_in = k0 * math.sin(math.radians(le_tilt_x))
        k_y_center_in = k0 * math.sin(math.radians(le_tilt_y))

    k_x_center_epe = k_x_center_in + G_ICG_x + G_EPE_x
    k_y_center_epe = k_y_center_in + G_ICG_y + G_EPE_y

    # OC grating vector (manual or auto)
    if auto_oc_flag:
        if oc_find_mode == "Specular Target":
            k_x_target = k_x_center_in
            k_y_target = k_y_center_in
        else:
            if grating_face_mode == "Reflection (Back Face)":
                k_x_target = k0 * math.sin(math.radians(custom_out_x)) / n_lam
                k_y_target = k0 * math.sin(math.radians(custom_out_y)) / n_lam
            else:
                k_x_target = k0 * math.sin(math.radians(custom_out_x))
                k_y_target = k0 * math.sin(math.radians(custom_out_y))
        G_OC_x   = k_x_target - k_x_center_epe
        G_OC_y   = k_y_target - k_y_center_epe
        G_OC_mag = math.sqrt(G_OC_x**2 + G_OC_y**2)
        if G_OC_mag > 0:
            wl_dict["Lambda_OC"] = (2 * math.pi) / G_OC_mag
            a_oc = float(np.degrees(math.atan2(G_OC_y, G_OC_x)) % 360.0)
        else:
            G_OC_x = (2 * math.pi / wl_dict["Lambda_OC"]) * math.cos(math.radians(a_oc))
            G_OC_y = (2 * math.pi / wl_dict["Lambda_OC"]) * math.sin(math.radians(a_oc))
    else:
        G_OC_x = (2 * math.pi / wl_dict["Lambda_OC"]) * math.cos(math.radians(a_oc))
        G_OC_y = (2 * math.pi / wl_dict["Lambda_OC"]) * math.sin(math.radians(a_oc))

    # FOV mesh
    H_deg, V_deg = np.meshgrid(
        np.arange(h_min, h_max + 0.1, 0.5),
        np.arange(v_min, v_max + 0.1, 0.5)
    )

    # [FIX] Back Face: kz_in을 k_wg_max 기준으로 계산 (매질 내부 파수 스케일)
    if grating_face_mode == "Reflection (Back Face)":
        kx_in = k0 * np.sin(np.radians(H_deg + le_tilt_x)) / n_lam
        ky_in = k0 * np.sin(np.radians(V_deg + le_tilt_y)) / n_lam
        kz_in = np.sqrt(np.maximum(k_wg_max**2 - kx_in**2 - ky_in**2, 0))
    else:
        kx_in = k0 * np.sin(np.radians(H_deg + le_tilt_x))
        ky_in = k0 * np.sin(np.radians(V_deg + le_tilt_y))
        kz_in = np.sqrt(np.maximum(k0**2 - kx_in**2 - ky_in**2, 0))

    # ICG
    kx_icg = kx_in + G_ICG_x
    ky_icg = ky_in + G_ICG_y
    kz_icg = np.sqrt(np.maximum(k_wg_max**2 - kx_icg**2 - ky_icg**2, 0))
    tir_mask_icg = (
        (kx_icg**2 + ky_icg**2 > k0**2) &
        (kx_icg**2 + ky_icg**2 < k_wg_max**2)
    )

    # EPE
    kx_epe = kx_icg + G_EPE_x
    ky_epe = ky_icg + G_EPE_y
    kz_epe = np.sqrt(np.maximum(k_wg_max**2 - kx_epe**2 - ky_epe**2, 0))
    if "Path B" in path_type:
        tir_mask_epe = (
            (kx_epe**2 + ky_epe**2 > k0**2) &
            (kx_epe**2 + ky_epe**2 < k_wg_max**2)
        )
    else:
        tir_mask_epe = np.ones_like(tir_mask_icg, dtype=bool)

    # OC
    kx_oc = kx_epe + G_OC_x
    ky_oc = ky_epe + G_OC_y
    kz_oc = np.sqrt(np.maximum(k0**2 - kx_oc**2 - ky_oc**2, 0))
    # [FIX] tir_mask_oc: 상한 조건만으로 evanescent 완전 차단. 중복 조건(>=0) 제거.
    tir_mask_oc = (kx_oc**2 + ky_oc**2) <= k0**2

    # [FIX] hop_distance: Path A/B 명시 분기
    if "Path B" in path_type:
        k_prop_x, k_prop_y, k_prop_z = kx_epe, ky_epe, kz_epe
    else:
        k_prop_x, k_prop_y, k_prop_z = kx_icg, ky_icg, kz_icg
    hop_dist = 2 * t_mm * (
        np.sqrt(k_prop_x**2 + k_prop_y**2) / np.maximum(k_prop_z, 1e-10)
    )

    mask_0 = np.ones_like(kx_in, dtype=bool)
    mask_1 = tir_mask_icg
    mask_2 = tir_mask_icg & tir_mask_epe
    mask_3 = tir_mask_icg & tir_mask_epe & tir_mask_oc & (hop_dist <= epd_limit)

    c_idx_v = np.argmin(np.abs(V_deg[:, 0]))
    c_idx_h = np.argmin(np.abs(H_deg[0, :]))
    c_ray = {
        "kx_in":  kx_in [c_idx_v, c_idx_h], "ky_in":  ky_in [c_idx_v, c_idx_h], "kz_in":  kz_in [c_idx_v, c_idx_h],
        "kx_icg": kx_icg[c_idx_v, c_idx_h], "ky_icg": ky_icg[c_idx_v, c_idx_h], "kz_icg": kz_icg[c_idx_v, c_idx_h],
        "kx_epe": kx_epe[c_idx_v, c_idx_h], "ky_epe": ky_epe[c_idx_v, c_idx_h], "kz_epe": kz_epe[c_idx_v, c_idx_h],
        "kx_oc":  kx_oc [c_idx_v, c_idx_h], "ky_oc":  ky_oc [c_idx_v, c_idx_h], "kz_oc":  kz_oc [c_idx_v, c_idx_h],
    }

    return {
        "color": wl_dict["color"],
        "k0": k0, "k_wg_max": k_wg_max,
        "kx_in": kx_in, "ky_in": ky_in, "kz_in": kz_in,
        "kx_icg": kx_icg, "ky_icg": ky_icg, "kz_icg": kz_icg,
        "kx_epe": kx_epe, "ky_epe": ky_epe, "kz_epe": kz_epe,
        "kx_oc":  kx_oc,  "ky_oc":  ky_oc,  "kz_oc":  kz_oc,
        "mask_0": mask_0, "mask_1": mask_1, "mask_2": mask_2, "mask_3": mask_3,
        "hop_distance": hop_dist,
        "H_mesh": H_deg, "V_mesh": V_deg,
        "c_ray": c_ray,
        "G_ICG_x": G_ICG_x, "G_EPE_x": G_EPE_x, "G_OC_x": G_OC_x,
        "Lambda_ICG": wl_dict["Lambda_ICG"],
        "Lambda_EPE": wl_dict["Lambda_EPE"],
        "Lambda_OC":  wl_dict["Lambda_OC"],
        "eff_icg": wl_dict["eff_icg"],
        "eff_epe": wl_dict["eff_epe"],
        "eff_oc":  wl_dict["eff_oc"],
        "calculated_a_oc": a_oc,
    }

# --- 5. Session State Initialization ---
if "srg_cached_results"  not in st.session_state: st.session_state["srg_cached_results"]  = None
if "auto_oc_pitch_val"   not in st.session_state: st.session_state["auto_oc_pitch_val"]   = 300.0
if "auto_oc_vector_ang"  not in st.session_state: st.session_state["auto_oc_vector_ang"]  = 120.0
if "ai_analysis_result"  not in st.session_state: st.session_state["ai_analysis_result"]  = None
if "ai_analysis_params"  not in st.session_state: st.session_state["ai_analysis_params"]  = None

# --- 6. Sidebar UI ---
st.sidebar.markdown("### 💾 Configuration Management")
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    st.sidebar.download_button("Save Setup", data=export_settings(), file_name="waveguide_config.json", use_container_width=True)
with col_s2:
    uploaded = st.sidebar.file_uploader("Load Setup", type="json", label_visibility="collapsed")
    if uploaded:
        st.sidebar.button("Apply Settings", on_click=import_settings, args=(uploaded.getvalue().decode("utf-8"),), use_container_width=True)

st.sidebar.markdown("---")
path_choice       = st.sidebar.radio("Grating Path Type", ["Path A (ICG→OC)", "Path B (ICG→EPE→OC)"], index=1, horizontal=True, key="path_choice")
grating_face_mode = st.sidebar.radio("Grating Spatial Position Mode", ["Transmission (Front Face)", "Reflection (Back Face)"], index=0, key="grating_face_mode")
single_layer_sync = st.sidebar.checkbox("Single Layer Mode (RGB Sync)", value=True, key="single_layer_sync")
coord_sys         = st.sidebar.radio("K-Space Coordinates System", ["Absolute Wavevector (nm⁻¹)", "Normalized Wavevector (Direction Cosine)"], index=1, horizontal=True, key="coord_sys")

st.sidebar.markdown("---")
st.sidebar.markdown("**⚙️ Light Engine Alignment Tilt Panel**")
le_tilt_x = dual_input("LE Horizontal Incident Angle θ_x (°)",           -30.0, 30.0, 0.0, 0.1, "le_tilt_x", "%.1f", sidebar=True)
le_tilt_y = dual_input("LE Vertical Incident Angle θ_y (° -:Gnd +:Sky)",  -30.0, 30.0, 0.0, 0.1, "le_tilt_y", "%.1f", sidebar=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**📐 Hardware Glass Properties**")
n_d_in    = dual_input("Substrate Index (n at 589nm)", 1.0,   3.0,  1.75, 0.01, "n_d",    "%.2f")
abbe_v_in = dual_input("Abbe Number (Vd)",            10.0, 100.0, 35.0,  0.1,  "abbe_v", "%.1f")

if grating_face_mode == "Reflection (Back Face)":
    n_green = get_refractive_index(520.0, n_d_in, abbe_v_in)
    def _safe_arcsin(x):
        return float(np.degrees(np.arcsin(x))) if abs(x) <= 1.0 else 0.0
    glass_tilt_x = _safe_arcsin(math.sin(math.radians(le_tilt_x)) / n_green)
    glass_tilt_y = _safe_arcsin(math.sin(math.radians(le_tilt_y)) / n_green)
    st.sidebar.markdown(f"""
    <div style="background-color:rgba(16,185,129,0.08);border:1px solid #10b981;padding:8px;border-radius:4px;margin-top:5px;margin-bottom:10px;">
        <div style="font-size:11px;color:#10b981;font-weight:bold;">🔍 Snell's Law Back Face Refraction View</div>
        <div style="font-size:11.5px;margin-top:3px;font-weight:500;color:#333;">• Green Refractive Index (n_G): <b>{n_green:.3f}</b></div>
        <div style="font-size:11.5px;font-weight:500;color:#444;">• Effective Glass Angle X: <span style="color:#10b981;font-weight:700;">{glass_tilt_x:.2f}°</span> (Air: {le_tilt_x:.1f}°)</div>
        <div style="font-size:11.5px;font-weight:500;color:#444;">• Effective Glass Angle Y: <span style="color:#10b981;font-weight:700;">{glass_tilt_y:.2f}°</span> (Air: {le_tilt_y:.1f}°)</div>
    </div>
    """, unsafe_allow_html=True)

thickness_in = dual_input("Substrate Thickness t (mm)", 0.1,  3.0, 0.40, 0.01, "thickness", "%.2f")
epd_val_in   = dual_input("Light Engine EPD Size (mm)", 1.0, 50.0,  3.5,  0.1, "epd_val",   "%.1f")
h_fov = dual_range_input("Horizontal FOV Bounds (°)", -60, 60, (-30, 30), 0.01, "h_fov")
v_fov = dual_range_input("Vertical FOV Bounds (°)",   -60, 60, (-20, 20), 0.01, "v_fov")
m_ord = st.sidebar.selectbox("Diffraction Order (m)", [1, -1, 2, -2], index=1, key="m_order_select")

st.sidebar.markdown("---")
st.sidebar.markdown("**📐 Out-Coupler Aperture Form Factor**")
oc_width  = dual_input("OC Aperture Width (mm)",  5.0, 100.0, 30.0, 0.1, "oc_width",  "%.1f", sidebar=True)
oc_height = dual_input("OC Aperture Height (mm)", 5.0, 100.0, 20.0, 0.1, "oc_height", "%.1f", sidebar=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**🔮 Grating Vector Rotation Alignment**")
angle_icg = dual_input("ICG Vector Angle (°)", 0.0, 360.0,   0.0, 0.01, "angle_icg", "%.2f", sidebar=True)
angle_epe = dual_input("EPE Vector Angle (°)", 0.0, 360.0, 240.0, 0.01, "angle_epe", "%.2f", sidebar=True) if "Path B" in path_choice else 0.0

auto_oc_angle = st.sidebar.checkbox("Auto-find OC Angle Mode", value=False, key="auto_oc_angle")
oc_find_mode  = "Specular Target"
custom_out_x  = 0.0
custom_out_y  = 0.0

if auto_oc_angle:
    oc_find_mode = st.sidebar.radio("OC Auto-Find Target Condition", ["Specular Target", "Custom Output Angle"], index=0, key="oc_find_mode")
    if oc_find_mode == "Custom Output Angle":
        st.sidebar.markdown("*🎯 Target Out-Coupling Angle Setup*")
        custom_out_x = dual_input("Target Horizontal Out Angle (°)", -30.0, 30.0, 0.0, 0.1, "custom_out_x", "%.1f", sidebar=True)
        custom_out_y = dual_input("Target Vertical Out Angle (°)",   -30.0, 30.0, 0.0, 0.1, "custom_out_y", "%.1f", sidebar=True)
    angle_oc     = st.session_state["auto_oc_vector_ang"]
    auto_oc_line = (angle_oc + 90.0) % 180.0
    theme_color  = "#4b96ff" if oc_find_mode == "Specular Target" else "#ff964b"
    st.sidebar.markdown(f"""
    <div style="background-color:rgba(75,150,255,0.08);border:1px solid {theme_color};padding:8px;border-radius:4px;margin-top:5px;margin-bottom:5px;">
        <div style="font-size:11px;color:{theme_color};font-weight:bold;">🔒 OC Auto-Tracking Active ({oc_find_mode})</div>
        <div style="font-size:12px;font-weight:600;margin-top:3px;">• Calculated Λ_OC: {st.session_state["auto_oc_pitch_val"]:.2f} nm</div>
        <div style="font-size:12px;font-weight:600;">• Vector Angle: {angle_oc:.2f}°</div>
        <div style="font-size:12px;font-weight:600;">• Line Angle: {auto_oc_line:.2f}°</div>
    </div>
    """, unsafe_allow_html=True)
else:
    angle_oc = dual_input("OC Vector Angle (°)", 0.0, 360.0, 120.0, 0.01, "angle_oc", "%.2f", sidebar=True)

st.sidebar.markdown("---")

def get_wl_inputs(name, def_l, def_p, n_d, V_d):
    act = st.sidebar.checkbox(f"Enable Wavelength {name}", value=True, key=f"{name}_active")
    if not act:
        return {"active": False}
    with st.sidebar.container():
        st.markdown(f"**[{name} Channel] Parameters**")
        wl      = dual_input("λ Wavelength (nm)", 400.0, 750.0, def_l, 0.01, f"{name}_wl",  "%.2f", sidebar=True)
        limit_p = wl / get_refractive_index(wl, n_d, V_d)
        icg_p   = dual_input("Λ_ICG Period (nm)", 100.0, 1000.0, def_p, 0.01, f"{name}_icg", "%.2f", sidebar=True)
        if icg_p < limit_p:
            st.error(f"⚠️ Limit Error: Min {limit_p:.2f}nm required")
        else:
            st.caption(f"Physical Limit Grating Pitch: {limit_p:.2f}nm")

        epe_p = dual_input("Λ_EPE Period (nm)", 100.0, 1000.0,
                           icg_p if single_layer_sync else def_p,
                           0.01, f"{name}_epe", "%.2f", sidebar=True) if "Path B" in path_choice else None

        if auto_oc_angle:
            oc_p = st.session_state["auto_oc_pitch_val"]
        else:
            oc_p = dual_input("Λ_OC Period (nm)", 100.0, 1000.0,
                              icg_p if single_layer_sync else def_p,
                              0.01, f"{name}_oc", "%.2f", sidebar=True)

        st.markdown("*Diffraction Efficiencies (0.00 ~ 1.00)*")
        eff_icg = dual_input("ICG Efficiency", 0.00, 1.00, 0.30, 0.01, f"{name}_efficg", "%.2f", sidebar=True)
        # [FIX] Path A일 때 float(1.0) 명시하여 타입 안전성 보장
        eff_epe = float(dual_input("EPE Efficiency", 0.00, 1.00, 0.20, 0.01, f"{name}_effepe", "%.2f", sidebar=True)
                        if "Path B" in path_choice else 1.0)
        eff_oc  = dual_input("OC Efficiency",  0.00, 1.00, 0.40, 0.01, f"{name}_effoc",  "%.2f", sidebar=True)

        return {
            "active": True, "lambda": wl,
            "Lambda_ICG": icg_p, "Lambda_EPE": epe_p, "Lambda_OC": oc_p,
            "eff_icg": eff_icg, "eff_epe": eff_epe, "eff_oc": eff_oc,
            "color": name,
        }

wl_R = get_wl_inputs("R", 638.0, 300.0, n_d_in, abbe_v_in)
wl_G = get_wl_inputs("G", 520.0, 300.0, n_d_in, abbe_v_in)
wl_B = get_wl_inputs("B", 450.0, 300.0, n_d_in, abbe_v_in)

st.sidebar.markdown("---")
st.sidebar.markdown("**🤖 AI Design Briefing**")
ai_mode = st.sidebar.radio(
    "AI 사용 방식",
    ["🔑 API Key 직접 호출", "📋 프롬프트 복사 (무료)"],
    index=1,
    key="ai_mode_sel",
    help="API Key 없이도 프롬프트 복사 모드로 AI 웹 앱에서 무료로 사용할 수 있습니다."
)
ai_provider = st.sidebar.radio(
    "AI Provider",
    ["Claude", "Gemini"],
    index=0,
    key="ai_provider_sel",
    help="Claude 또는 Gemini 중 하나를 선택해 직접 API 호출할 수 있습니다."
)
ai_api_key = ""
if ai_mode == "🔑 API Key 직접 호출":
    if ai_provider == "Claude":
        _env_key = get_secret_value("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
        ai_api_key = st.sidebar.text_input(
            "Anthropic API Key",
            value=_env_key,
            type="password",
            placeholder="sk-ant-...",
            help="환경변수 ANTHROPIC_API_KEY 또는 Streamlit Secrets에 저장하면 자동 입력됩니다.",
            label_visibility="collapsed"
        )
    else:
        _env_key = get_secret_value("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
        ai_api_key = st.sidebar.text_input(
            "Google Gemini API Key",
            value=_env_key,
            type="password",
            placeholder="AIza...",
            help="환경변수 GEMINI_API_KEY 또는 Streamlit Secrets에 저장하면 자동 입력됩니다.",
            label_visibility="collapsed"
        )

st.sidebar.markdown("---")
run_simulation_trigger = st.sidebar.button(label="▶ Run Simulation", use_container_width=True)

# --- 7. Simulation Trigger ---
if run_simulation_trigger:
    results = {}
    for data in [wl_R, wl_G, wl_B]:
        res = calculate_k_space(
            data, n_d_in, abbe_v_in, m_ord,
            h_fov[0], h_fov[1], v_fov[0], v_fov[1],
            thickness_in, epd_val_in, path_choice, grating_face_mode, oc_width,
            angle_icg, angle_epe, angle_oc,
            le_tilt_x, le_tilt_y,
            auto_oc_angle, oc_find_mode, custom_out_x, custom_out_y
        )
        if res:
            results[data["color"]] = res
            if auto_oc_angle:
                st.session_state["auto_oc_pitch_val"]  = res["Lambda_OC"]
                st.session_state["auto_oc_vector_ang"] = res["calculated_a_oc"]
    st.session_state["srg_cached_results"] = results
    # 새 시뮬레이션 실행 시 이전 AI 분석 초기화
    st.session_state["ai_analysis_result"] = None
    st.session_state["ai_analysis_params"] = {
        "path_choice": path_choice,
        "grating_face_mode": grating_face_mode,
        "n_d": n_d_in, "abbe_v": abbe_v_in,
        "thickness_mm": thickness_in, "epd_mm": epd_val_in,
        "h_fov": h_fov, "v_fov": v_fov,
        "oc_width": oc_width, "oc_height": oc_height,
        "m_ord": m_ord,
        "le_tilt_x": le_tilt_x, "le_tilt_y": le_tilt_y,
        "wl_R": wl_R, "wl_G": wl_G, "wl_B": wl_B,
    }
    st.rerun()

# --- 8. Main Dashboard ---
results = st.session_state["srg_cached_results"]

if results is not None:
    st.title("SRG DOE Waveguide Simulation Dashboard")
    viz_options = list(results.keys())
    if len(results) > 1:
        viz_options.append("RGB Overlap View")

    tab_xy, tab_xz, tab_sweep, tab_ai = st.tabs([
        "K-Space (XY Layout)",
        "K-Space (XZ Profile Cross-Section)",
        "Thickness (t) Margin Sweep Analysis",
        "🤖 AI Design Briefing"
    ])
    c_map = {"R": "red", "G": "green", "B": "blue"}

    # ── XY Tab ──────────────────────────────────────────────────────────────
    with tab_xy:
        target      = st.selectbox("XY Visualization Target", viz_options, index=len(viz_options)-1, key="xy_sel")
        fig_xy      = go.Figure()
        common_mask = None
        plots       = list(results.values()) if target == "RGB Overlap View" else [results[target]]

        for r in plots:
            cn, pc = r["color"], c_map[r["color"]]
            sf     = r["k0"] if coord_sys == "Normalized Wavevector (Direction Cosine)" else 1.0

            fig_xy.add_shape(type="circle",
                x0=-r["k0"]/sf,      y0=-r["k0"]/sf,
                x1= r["k0"]/sf,      y1= r["k0"]/sf,
                line_color=pc, line_dash="dash", opacity=0.4)
            fig_xy.add_shape(type="circle",
                x0=-r["k_wg_max"]/sf, y0=-r["k_wg_max"]/sf,
                x1= r["k_wg_max"]/sf, y1= r["k_wg_max"]/sf,
                line_color=pc, fillcolor=pc, opacity=0.03)

            m0, m1, m2, m3 = r["mask_0"], r["mask_1"], r["mask_2"], r["mask_3"]
            common_mask = m3.copy() if common_mask is None else (common_mask & m3)

            fig_xy.add_trace(go.Scatter(
                x=r["kx_in"][m0]/sf, y=r["ky_in"][m0]/sf, mode="markers",
                marker=dict(size=2, color=pc, symbol="circle-open", opacity=0.1),
                name=f"{cn} Input FOV", hoverinfo="skip"))
            fig_xy.add_trace(go.Scatter(
                x=r["kx_icg"][m1]/sf, y=r["ky_icg"][m1]/sf, mode="markers",
                marker=dict(size=3, color=pc, symbol="square", opacity=0.3),
                name=f"{cn} Coupled TIR", hoverinfo="skip"))
            if "Path B" in path_choice:
                fig_xy.add_trace(go.Scatter(
                    x=r["kx_epe"][m2]/sf, y=r["ky_epe"][m2]/sf, mode="markers",
                    marker=dict(size=3, color=pc, symbol="diamond", opacity=0.5),
                    name=f"{cn} EPE Extracted", hoverinfo="skip"))

            hover_txt = [
                f"H:{h:.2f}° V:{v:.2f}°<br>Hop:{hp:.2f}mm  Overlap:{epd_val_in-hp:.2f}mm"
                for h, v, hp in zip(r["H_mesh"][m3], r["V_mesh"][m3], r["hop_distance"][m3])
            ]
            fig_xy.add_trace(go.Scatter(
                x=r["kx_oc"][m3]/sf, y=r["ky_oc"][m3]/sf, mode="markers",
                marker=dict(size=4, color=pc, symbol="circle", opacity=0.9),
                name=f"{cn} Final Out", text=hover_txt, hoverinfo="text"))

            if target != "RGB Overlap View" or cn == "G":
                c = r["c_ray"]
                pts = [[c["kx_in"], c["ky_in"]], [c["kx_icg"], c["ky_icg"]]]
                if "Path B" in path_choice:
                    pts.append([c["kx_epe"], c["ky_epe"]])
                pts.append([c["kx_oc"], c["ky_oc"]])
                for i in range(len(pts) - 1):
                    fig_xy.add_annotation(
                        x=pts[i+1][0]/sf, y=pts[i+1][1]/sf,
                        ax=pts[i][0]/sf,  ay=pts[i][1]/sf,
                        xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True, arrowhead=2, arrowcolor=pc, opacity=0.8)

        lim = 2.0 if coord_sys == "Normalized Wavevector (Direction Cosine)" else 0.025
        fig_xy.update_layout(
            xaxis=dict(range=[-lim, lim], scaleanchor="y", scaleratio=1, title=f"Kx ({coord_sys})"),
            yaxis=dict(range=[-lim, lim], title=f"Ky ({coord_sys})"),
            width=800, height=800, plot_bgcolor="white")
        st.plotly_chart(fig_xy, use_container_width=True)

        # Summary table
        st.subheader("📊 Grating Specifications & Orientation Structural Mapping Summary")
        summary_table = {}
        area_m2 = (oc_width * 1e-3) * (oc_height * 1e-3)

        for c, r in results.items():
            mask         = r["mask_3"]
            icg_line_ang = (angle_icg + 90.0) % 180.0
            epe_line_ang = (angle_epe + 90.0) % 180.0 if angle_epe else None
            oc_line_ang  = (r["calculated_a_oc"] + 90.0) % 180.0

            if np.any(mask):
                vh   = r["H_mesh"][mask]
                vv   = r["V_mesh"][mask]
                h_span_rad = np.radians(np.max(vh) - np.min(vh))
                v_span_rad = np.radians(np.max(vv) - np.min(vv))
                omega = (4.0 * math.asin(math.sin(h_span_rad/2.0) * math.sin(v_span_rad/2.0))
                         if (h_span_rad > 0 and v_span_rad > 0) else 1e-6)

                c_ray  = r["c_ray"]
                k_rho  = math.sqrt(c_ray["kx_icg"]**2 + c_ray["ky_icg"]**2)
                k_z    = max(c_ray["kz_icg"], 1e-10)
                # [FIX] prop_distance_mm → oc_width (실제 OC 가로 크기 연동)
                prop_distance_mm = oc_width
                num_bounces      = prop_distance_mm / (2.0 * thickness_in * (k_rho / k_z))
                bulk_path_len    = prop_distance_mm / (k_rho / max(1e-10, r["k0"] * n_d_in))
                tir_loss_factor  = (0.998 ** max(1.0, num_bounces)) * (0.999 ** bulk_path_len)
                lumen_out        = r["eff_icg"] * r["eff_epe"] * r["eff_oc"] * tir_loss_factor
                nits_per_lumen   = lumen_out / (area_m2 * omega) if omega > 0 else 0.0

                summary_table[c] = {
                    "ICG Pitch / Line Angle": f"{r['Lambda_ICG']:.1f}nm / {icg_line_ang:.1f}°",
                    "EPE Pitch / Line Angle": f"{r['Lambda_EPE']:.1f}nm / {epe_line_ang:.1f}°" if epe_line_ang is not None else "-",
                    "OC Pitch / Line Angle":  f"{r['Lambda_OC']:.1f}nm / {oc_line_ang:.1f}°",
                    "Effective H-FOV": f"{np.min(vh):.1f}°~{np.max(vh):.1f}°",
                    "Effective V-FOV": f"{np.min(vv):.1f}°~{np.max(vv):.1f}°",
                    "FOV Pass Ratio":  f"{np.sum(mask)/mask.size*100:.1f}%",
                    "System Efficiency (nits/lm)": f"{nits_per_lumen:,.0f}",
                }
            else:
                summary_table[c] = {
                    "ICG Pitch / Line Angle": f"{r['Lambda_ICG']:.1f}nm / {icg_line_ang:.1f}°",
                    "EPE Pitch / Line Angle": f"{r['Lambda_EPE']:.1f}nm / {epe_line_ang:.1f}°" if epe_line_ang is not None else "-",
                    "OC Pitch / Line Angle":  f"{r['Lambda_OC']:.1f}nm / {oc_line_ang:.1f}°",
                    "Effective H-FOV": "None", "Effective V-FOV": "None",
                    "FOV Pass Ratio": "0%", "System Efficiency (nits/lm)": "0",
                }

        if common_mask is not None and np.any(common_mask):
            ref_r  = results[list(results.keys())[0]]
            ch     = ref_r["H_mesh"][common_mask]
            cv     = ref_r["V_mesh"][common_mask]
            c_nits = (sum(float(summary_table[col]["System Efficiency (nits/lm)"].replace(",", ""))
                          for col in results) / len(results))
            summary_table["RGB Common"] = {
                "ICG Pitch / Line Angle": "-", "EPE Pitch / Line Angle": "-", "OC Pitch / Line Angle": "-",
                "Effective H-FOV": f"{np.min(ch):.1f}°~{np.max(ch):.1f}°",
                "Effective V-FOV": f"{np.min(cv):.1f}°~{np.max(cv):.1f}°",
                "FOV Pass Ratio":  f"{np.sum(common_mask)/common_mask.size*100:.1f}%",
                "System Efficiency (nits/lm)": f"{c_nits:,.0f}",
            }
        st.table(pd.DataFrame(summary_table))

    # ── XZ Tab ──────────────────────────────────────────────────────────────
    with tab_xz:
        target_xz = st.selectbox("XZ Visualization Target", viz_options, index=len(viz_options)-1, key="xz_sel")
        fig_xz    = go.Figure()
        max_kz    = 0
        plots_xz  = list(results.values()) if target_xz == "RGB Overlap View" else [results[target_xz]]
        arc_ang   = np.linspace(0, np.pi, 100)

        for r in plots_xz:
            cn, pc = r["color"], c_map[r["color"]]
            sf     = r["k0"] if coord_sys == "Normalized Wavevector (Direction Cosine)" else 1.0
            max_kz = max(max_kz, r["k_wg_max"] / sf)

            fig_xz.add_trace(go.Scatter(
                x=(r["k0"]/sf)*np.cos(arc_ang), y=(r["k0"]/sf)*np.sin(arc_ang),
                mode="lines", line=dict(color=pc, dash="dash"), showlegend=False))
            fig_xz.add_trace(go.Scatter(
                x=(r["k_wg_max"]/sf)*np.cos(arc_ang), y=(r["k_wg_max"]/sf)*np.sin(arc_ang),
                mode="lines", line=dict(color=pc, width=1),
                fill="tonexty",
                fillcolor=f"rgba({255 if cn=='R' else 0},{255 if cn=='G' else 0},{255 if cn=='B' else 0},0.04)",
                showlegend=False))

            c = r["c_ray"]
            fig_xz.add_annotation(
                x=c["kx_icg"]/sf, y=c["kz_in"]/sf,
                ax=c["kx_in"]/sf, ay=c["kz_in"]/sf,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowcolor=pc, text="ICG")
            fig_xz.add_trace(go.Scatter(
                x=[c["kx_icg"]/sf, c["kx_icg"]/sf],
                y=[c["kz_in"]/sf,  c["kz_icg"]/sf],
                mode="lines", line=dict(color=pc, dash="dot"), showlegend=False))

            curr_kx, curr_kz = c["kx_icg"], c["kz_icg"]
            if "Path B" in path_choice and r["G_EPE_x"] != 0:
                fig_xz.add_annotation(
                    x=c["kx_epe"]/sf, y=c["kz_icg"]/sf,
                    ax=c["kx_icg"]/sf, ay=c["kz_icg"]/sf,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowcolor=pc, text="EPE")
                fig_xz.add_trace(go.Scatter(
                    x=[c["kx_epe"]/sf, c["kx_epe"]/sf],
                    y=[c["kz_icg"]/sf, c["kz_epe"]/sf],
                    mode="lines", line=dict(color=pc, dash="dot"), showlegend=False))
                curr_kx, curr_kz = c["kx_epe"], c["kz_epe"]

            fig_xz.add_annotation(
                x=c["kx_oc"]/sf, y=curr_kz/sf,
                ax=curr_kx/sf,   ay=curr_kz/sf,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowcolor="purple", text="OC")
            fig_xz.add_trace(go.Scatter(
                x=[c["kx_oc"]/sf, c["kx_oc"]/sf],
                y=[curr_kz/sf,    c["kz_oc"]/sf],
                mode="lines", line=dict(color="purple", dash="dot"), showlegend=False))
            fig_xz.add_trace(go.Scatter(
                x=[c["kx_in"]/sf, c["kx_oc"]/sf],
                y=[c["kz_in"]/sf, c["kz_oc"]/sf],
                mode="markers", marker=dict(size=10, color=pc, symbol="star"),
                name=f"{cn} Central Field Ray"))

        lim_z = 2.0 if coord_sys == "Normalized Wavevector (Direction Cosine)" else max_kz * 1.1
        fig_xz.update_layout(
            title=f"K-Space XZ Cross-Section — Central Gaze Path ({coord_sys})",
            xaxis=dict(range=[-lim_z, lim_z], scaleanchor="y"),
            yaxis=dict(range=[0, lim_z]),
            height=600, plot_bgcolor="white")
        st.plotly_chart(fig_xz, use_container_width=True)

    # ── Thickness Sweep Tab ──────────────────────────────────────────────────
    with tab_sweep:
        st.markdown("#### Substrate Thickness Sweep Panel (Optimization Loop)")
        c1, c2, c3 = st.columns(3)
        ts = c1.number_input("Start Thickness (mm)", 0.1, 3.0, 0.2)
        te = c2.number_input("End Thickness (mm)",   0.1, 3.0, 1.0)
        tp = c3.number_input("Step (mm)",            0.01, 1.0, 0.05)

        if st.button("Run Sweep Simulation Loop", use_container_width=True):
            t_arr  = np.arange(ts, te + 1e-9, tp)
            res_l  = []
            pb     = st.progress(0)

            for i, t in enumerate(t_arr):
                d       = {"t": round(float(t), 3)}
                cm      = None
                last_rt = None

                for ch_key, wd in [("R", wl_R), ("G", wl_G), ("B", wl_B)]:
                    if not wd["active"]:
                        continue
                    rt = calculate_k_space(
                        wd, n_d_in, abbe_v_in, m_ord,
                        h_fov[0], h_fov[1], v_fov[0], v_fov[1],
                        t, epd_val_in, path_choice, grating_face_mode, oc_width,
                        angle_icg, angle_epe, angle_oc,
                        le_tilt_x, le_tilt_y,
                        auto_oc_angle, oc_find_mode, custom_out_x, custom_out_y
                    )
                    mt      = rt["mask_3"]
                    last_rt = rt
                    cm      = mt.copy() if cm is None else (cm & mt)
                    vh, vv  = rt["H_mesh"][mt], rt["V_mesh"][mt]
                    d[f"{ch_key} H-Span"] = float(np.max(vh) - np.min(vh)) if np.any(mt) else 0.0
                    d[f"{ch_key} V-Span"] = float(np.max(vv) - np.min(vv)) if np.any(mt) else 0.0

                # [FIX] 캐시된 results 대신 루프 내 last_rt 메시 사용
                if (cm is not None and last_rt is not None
                        and all(wd["active"] for wd in [wl_R, wl_G, wl_B])):
                    ch_arr = last_rt["H_mesh"][cm]
                    cv_arr = last_rt["V_mesh"][cm]
                    d["Common H-Span"] = float(np.max(ch_arr) - np.min(ch_arr)) if np.any(cm) else 0.0
                    d["Common V-Span"] = float(np.max(cv_arr) - np.min(cv_arr)) if np.any(cm) else 0.0

                res_l.append(d)
                pb.progress((i + 1) / len(t_arr))

            df = pd.DataFrame(res_l)
            fs = go.Figure()
            for col in df.columns[1:]:
                lc = "black" if "Common" in col else ("red" if "R" in col else ("green" if "G" in col else "blue"))
                ls = "solid" if "H-Span" in col else "dot"
                fs.add_trace(go.Scatter(
                    x=df["t"], y=df[col], mode="lines+markers", name=col,
                    line=dict(color=lc, width=2, dash=ls)))
            fs.update_layout(
                title="H/V FOV Span vs Waveguide Substrate Thickness",
                xaxis_title="Thickness (mm)", yaxis_title="Span (°)")
            st.plotly_chart(fs, use_container_width=True)
            st.dataframe(df.style.format("{:.2f}"), use_container_width=True)

    # ── AI Design Briefing Tab ───────────────────────────────────────────────
    with tab_ai:
        st.markdown("### 🤖 AI Design Briefing")
        st.caption(
            "시뮬레이션 결과를 AI가 분석하고 개선 방향을 제안합니다.  \n"
            "**API Key 없이도** 프롬프트 복사 모드로 Claude.ai에서 무료로 사용할 수 있습니다."
        )

        # ── 프롬프트 생성 함수 ─────────────────────────────────────────────
        def build_ai_prompt(params, sim_results):
            p = params

            # 파장별 파라미터
            wl_lines = []
            for ch_key, wd in [("R", p["wl_R"]), ("G", p["wl_G"]), ("B", p["wl_B"])]:
                if not wd.get("active"):
                    continue
                line = (f"  [{ch_key}] λ={wd['lambda']:.1f}nm | "
                        f"Λ_ICG={wd['Lambda_ICG']:.1f}nm")
                if wd.get("Lambda_EPE"):
                    line += f" | Λ_EPE={wd['Lambda_EPE']:.1f}nm"
                line += (f" | Λ_OC={wd['Lambda_OC']:.1f}nm | "
                         f"η_ICG={wd['eff_icg']:.2f} / η_EPE={wd['eff_epe']:.2f} / η_OC={wd['eff_oc']:.2f}")
                wl_lines.append(line)
            wl_text = "\n".join(wl_lines) if wl_lines else "  (활성 채널 없음)"

            # 시뮬레이션 결과
            res_lines = []
            common_mask = None
            for color, r in sim_results.items():
                mask = r["mask_3"]
                common_mask = mask.copy() if common_mask is None else (common_mask & mask)
                if np.any(mask):
                    vh = r["H_mesh"][mask]; vv = r["V_mesh"][mask]
                    res_lines.append(
                        f"  [{color}] H-FOV: {np.min(vh):.1f}°~{np.max(vh):.1f}° | "
                        f"V-FOV: {np.min(vv):.1f}°~{np.max(vv):.1f}° | "
                        f"Pass: {np.sum(mask)/mask.size*100:.1f}%"
                    )
                else:
                    res_lines.append(f"  [{color}] TIR 조건 불충족 — 유효 FOV 없음")

            if common_mask is not None and np.any(common_mask):
                ref = list(sim_results.values())[0]
                ch  = ref["H_mesh"][common_mask]; cv = ref["V_mesh"][common_mask]
                res_lines.append(
                    f"\n  [RGB 공통] H-FOV: {np.min(ch):.1f}°~{np.max(ch):.1f}° | "
                    f"V-FOV: {np.min(cv):.1f}°~{np.max(cv):.1f}° | "
                    f"Pass: {np.sum(common_mask)/common_mask.size*100:.1f}%"
                )
            else:
                res_lines.append("\n  [RGB 공통] 공통 유효 FOV 없음")

            return f"""당신은 AR 광학 설계 전문가입니다.
아래 SRG DOE Waveguide 프리검토 설계 파라미터와 K-space 시뮬레이션 결과를 분석하여
개발자에게 한국어로 브리핑과 개선 제안을 해주세요.

────────────────────────────────
## 현재 설계 파라미터
────────────────────────────────
그레이팅 경로         : {p['path_choice']}
가공 면               : {p['grating_face_mode']}
웨이브가이드 굴절률   : n_d = {p['n_d']:.3f}  (Abbe Vd = {p['abbe_v']:.1f})
기판 두께             : {p['thickness_mm']:.2f} mm
LE EPD 크기           : {p['epd_mm']:.1f} mm
입사 H-FOV            : {p['h_fov'][0]:.1f}° ~ {p['h_fov'][1]:.1f}°
입사 V-FOV            : {p['v_fov'][0]:.1f}° ~ {p['v_fov'][1]:.1f}°
OC 개구 크기          : {p['oc_width']:.1f} mm × {p['oc_height']:.1f} mm
회절 차수 m           : {p['m_ord']}
LE 틸트               : θ_x={p['le_tilt_x']:.1f}°, θ_y={p['le_tilt_y']:.1f}°

파장별 그레이팅 설계값:
{wl_text}

────────────────────────────────
## K-Space 시뮬레이션 결과
────────────────────────────────
{chr(10).join(res_lines)}

────────────────────────────────
위 내용을 바탕으로 아래 4가지 섹션으로 나누어 응답해주세요.
마크다운 형식으로 작성하고, 기술적 근거를 포함해주세요.

### 1. 📋 설계 현황 브리핑
현재 설계의 핵심 구성과 상태를 3~5문장으로 간결하게 요약해주세요.

### 2. 🔍 시뮬레이션 결과 분석
- 각 파장(R/G/B)의 TIR 조건 충족 여부 및 유효 FOV 평가
- RGB 공통 FOV의 균형성 (색수차 관점)
- 현재 설계에서 두드러지는 강점 또는 한계

### 3. 🚀 개선 방향 제안 (3~5가지)
각 제안마다: **제안 항목** / 물리적 근거 / 예상 효과를 포함해주세요.
수치 예시(파라미터 변경 범위)를 반드시 포함해주세요.

### 4. ⚠️ 주의사항
제조 또는 광학적으로 주의해야 할 잠재적 리스크를 2~3가지 언급해주세요.
"""

        # ── 현재 파라미터 스냅샷 ───────────────────────────────────────────
        params_snap = st.session_state.get("ai_analysis_params") or {
            "path_choice": path_choice, "grating_face_mode": grating_face_mode,
            "n_d": n_d_in, "abbe_v": abbe_v_in,
            "thickness_mm": thickness_in, "epd_mm": epd_val_in,
            "h_fov": h_fov, "v_fov": v_fov,
            "oc_width": oc_width, "oc_height": oc_height,
            "m_ord": m_ord, "le_tilt_x": le_tilt_x, "le_tilt_y": le_tilt_y,
            "wl_R": wl_R, "wl_G": wl_G, "wl_B": wl_B,
        }
        prompt_text = build_ai_prompt(params_snap, results)

        # ═══════════════════════════════════════════════════════════════════
        # 모드 A: API Key 직접 호출
        # ═══════════════════════════════════════════════════════════════════
        if ai_mode == "🔑 API Key 직접 호출":
            st.markdown("#### 🔑 API Key 직접 호출 모드")
            st.caption(f"{ai_provider} API Key로 자동 분석합니다. (사이드바에서 Key 입력)")

            col_btn1, col_btn2 = st.columns([3, 1])
            with col_btn1:
                run_ai = st.button(
                    "🤖 AI 설계 분석 실행",
                    use_container_width=True,
                    disabled=not ai_api_key.strip(),
                    help="사이드바에 API Key를 먼저 입력하세요." if not ai_api_key.strip() else ""
                )
            with col_btn2:
                if st.button("🗑️ 초기화", use_container_width=True, key="ai_reset_api"):
                    st.session_state["ai_analysis_result"] = None
                    st.rerun()

            if not ai_api_key.strip():
                if ai_provider == "Claude":
                    st.warning("사이드바에 Anthropic API Key를 입력해주세요.")
                else:
                    st.warning("사이드바에 Google Gemini API Key를 입력해주세요.")

            if run_ai and ai_api_key.strip():
                try:
                    with st.spinner("🤖 AI가 설계를 분석 중입니다..."):
                        if ai_provider == "Claude":
                            answer = call_claude_api(ai_api_key.strip(), prompt_text)
                        else:
                            answer = call_gemini_api(ai_api_key.strip(), prompt_text)
                    st.session_state["ai_analysis_result"] = answer
                    st.rerun()
                except RuntimeError as e:
                    err = str(e)
                    if "401" in err:
                        st.error("❌ API Key 인증 실패 — Key를 다시 확인해주세요.")
                    elif "429" in err:
                        st.error("❌ API 요청 한도 초과 — 잠시 후 다시 시도해주세요.")
                    else:
                        st.error(f"❌ 오류: {err}")

            if st.session_state.get("ai_analysis_result"):
                st.info("💡 아래는 최근 시뮬레이션에 대한 AI 분석입니다. 새 시뮬레이션 실행 시 초기화됩니다.")
                st.markdown(st.session_state["ai_analysis_result"])

        # ═══════════════════════════════════════════════════════════════════
        # 모드 B: 프롬프트 복사 (무료 — API Key 불필요)
        # ═══════════════════════════════════════════════════════════════════
        else:
            st.markdown("#### 📋 프롬프트 복사 모드 (무료)")
            st.markdown("""
<div style='background:#e8f4e8;border:1px solid #4caf50;border-radius:8px;padding:14px;margin-bottom:12px;'>
<b>사용 방법</b><br>
① 아래 <b>[📋 프롬프트 복사]</b> 버튼 클릭<br>
② <a href="https://claude.ai" target="_blank"><b>Claude.ai</b></a> 또는 <a href="https://gemini.google.com" target="_blank"><b>Gemini</b></a> 웹 앱을 열어 붙여넣기 (Ctrl+V)<br>
③ AI 응답을 복사한 뒤 아래 <b>응답 입력창</b>에 붙여넣기<br>
④ <b>[💾 저장]</b> 버튼으로 결과 보관
</div>
""", unsafe_allow_html=True)

            # 프롬프트 표시 + 복사 버튼
            st.markdown("**① 생성된 프롬프트 (클릭 → 전체 선택 → 복사)**")
            st.text_area(
                "prompt_area", value=prompt_text, height=280,
                label_visibility="collapsed", key="prompt_display"
            )
            # Streamlit 클립보드 복사 (st.code 이용한 one-click copy)
            st.markdown("**또는 아래 코드블록을 클릭해 한 번에 복사하세요:**")
            st.code(prompt_text, language=None)

            st.markdown("---")
            st.markdown("**③ Claude 응답을 여기에 붙여넣기**")
            pasted = st.text_area(
                "AI 응답 붙여넣기", height=300,
                placeholder="Claude.ai에서 받은 응답을 여기에 붙여넣으세요...",
                key="ai_paste_area"
            )

            col_save, col_clear = st.columns([3, 1])
            with col_save:
                if st.button("💾 저장", use_container_width=True, key="ai_save_btn"):
                    if pasted.strip():
                        st.session_state["ai_analysis_result"] = pasted.strip()
                        st.success("✅ 저장 완료!")
                        st.rerun()
                    else:
                        st.warning("응답을 먼저 붙여넣어 주세요.")
            with col_clear:
                if st.button("🗑️ 초기화", use_container_width=True, key="ai_reset_manual"):
                    st.session_state["ai_analysis_result"] = None
                    st.rerun()

            # 저장된 결과 표시
            if st.session_state.get("ai_analysis_result"):
                st.markdown("---")
                st.markdown("#### 📄 저장된 AI 분석 결과")
                st.markdown(st.session_state["ai_analysis_result"])

else:
    st.info("💡 사이드바의 설정을 조율하신 후 최하단의 [▶ Run Simulation] 버튼을 클릭하면 정밀 K-space 수치 해석 도면이 활성화됩니다.")
