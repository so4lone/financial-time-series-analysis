import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

@st.cache_data
def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


@st.cache_data
def calculate_macd(series, fast_period, slow_period, signal_period=9):
    ema_fast = calculate_ema(series, fast_period)
    ema_slow = calculate_ema(series, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    return macd_line, signal_line


@st.cache_data
def optimize_ema(series, target_min_segments):
    best_score = float('inf')
    best_params = (12, 26)
    target = target_min_segments

    for fast in range(10, 42, 2):
        for slow in range(20, 82, 2):
            if slow <= fast + 10:
                continue

            macd, signal = calculate_macd(series, fast, slow)
            hist = macd - signal
            trend_direction = np.where(hist > 0, 1, -1)
            trend_changes = np.diff(trend_direction)
            change_indices = np.where(trend_changes != 0)[0] + 1
            segment_lengths = np.diff(np.concatenate(([0], change_indices, [len(series)])))

            total_segments = len(segment_lengths)
            false_segments = np.sum(segment_lengths < 7)

            if total_segments == 0:
                continue

            ratio = false_segments / total_segments
            penalty = (target - total_segments) * 0.1 if total_segments < target else 0
            score = ratio + penalty

            if score < best_score and total_segments >= target:
                best_score = score
                best_params = (fast, slow)

    return best_params


def calculate_trend_for_segment(segment_df, y_name):
    if len(segment_df) < 5:
        return None, None

    start_date = segment_df['Date'].iloc[0]
    x = (segment_df['Date'] - start_date).dt.days.values
    y = segment_df['Value'].values

    slope, intercept, _, _, _ = stats.linregress(x, y)
    y_pred = slope * x + intercept
    residuals = y - y_pred

    shift = np.min(residuals) if slope > 0 else np.max(residuals)
    shifted_intercept = intercept + shift

    return slope, shifted_intercept

if 'data_confirmed' not in st.session_state:
    st.session_state.data_confirmed = False


def reset_confirmation():
    st.session_state.data_confirmed = False


def confirm_data():
    st.session_state.data_confirmed = True

st.set_page_config(layout="wide", page_title="Анализ Временных Рядов")
st.title("📈 Анализ трендов и прогнозирование пробоя")

uploaded_file = st.file_uploader("Загрузите таблицу данных (Excel или CSV)", type=['xlsx', 'csv'],
                                 on_change=reset_confirmation)

if uploaded_file is not None:
    is_csv = uploaded_file.name.endswith('.csv')

    if is_csv:
        delimiter = st.radio(
            "🔠 Рразделитель столбцов в CSV:",
            options=[";", ",", "\\t", "|"],
            index=0,
            horizontal=True,
            on_change=reset_confirmation
        )

    try:
        if is_csv:
            raw_df = pd.read_csv(uploaded_file, sep=delimiter)
        else:
            raw_df = pd.read_excel(uploaded_file, sheet_name=0)
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")
        st.stop()

    st.markdown("### 🛠 Настройка формата данных")

    col1, col2, col3 = st.columns(3)
    columns = raw_df.columns.tolist()

    if len(columns) <= 1:
        st.warning("⚠️ Внимание: Найден только 1 столбец. Скорее всего, выбран неправильный разделитель для CSV.")

    with col1:
        date_col = st.selectbox("Столбец с датами (ось X):", columns, index=0, on_change=reset_confirmation)

    with col2:
        default_val_idx = len(columns) - 1 if len(columns) > 1 else 0
        val_col = st.selectbox("Столбец со значениями (ось Y):", columns, index=default_val_idx,
                               on_change=reset_confirmation)

    with col3:
        data_format = st.radio(
            "Формат значений:",
            ["Абсолютные (готовые значения)", "Кумулятивные (накопительный итог)"],
            on_change=reset_confirmation
        )

    st.button("Продолжить", type="primary", on_click=confirm_data)
    st.divider()

    if st.session_state.data_confirmed:

        df = pd.DataFrame()
        df['Date'] = pd.to_datetime(raw_df[date_col].astype(str).str.strip(), errors='coerce')
        df['Raw_Value'] = pd.to_numeric(raw_df[val_col], errors='coerce').fillna(0)
        df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

        if data_format == "Кумулятивные (накопительный итог)":
            df['Value'] = df['Raw_Value'].diff().fillna(0)
        else:
            df['Value'] = df['Raw_Value']

        series = df['Value']

        st.sidebar.header("⚙️ Параметры алгоритма")
        mode = st.sidebar.radio("Режим подбора EMA", ["Автоматический", "Вручную"])

        if mode == "Автоматический":
            target_min_segments = st.sidebar.number_input(
                "Минимальное число макро-трендов:",
                min_value=1,
                max_value=30,
                value=9,
                step=1,
                help="Настройка для определения периодов EMA. Подбирать значение рекомендуется исходя из временного промежутка данных и шага времени"
            )

            with st.spinner('Анализ структуры ряда (Grid Search)...'):
                fast_period, slow_period = optimize_ema(series, target_min_segments)
            st.sidebar.success(
                f"Оптимальные параметры:\n\nБыстрая EMA: **{fast_period}**\n\nМедленная EMA: **{slow_period}**")
        else:
            fast_period = st.sidebar.number_input("Быстрая EMA", min_value=2, max_value=100, value=18)
            slow_period = st.sidebar.number_input("Медленная EMA", min_value=5, max_value=200, value=76)

        st.sidebar.header("🛡 Защита от рыночного шума")
        min_confirm_days = st.sidebar.slider("Дней для подтверждения разворота", min_value=1, max_value=40, value=14)

        df['MACD'], df['Signal'] = calculate_macd(series, fast_period, slow_period)
        df['Hist'] = df['MACD'] - df['Signal']
        df['Trend_Dir'] = np.where(df['Hist'] > 0, 1, -1)

        current_state = df['Trend_Dir'].iloc[0]
        states = [current_state]
        days_in_new_state = 0

        for i in range(1, len(df)):
            raw_state = df['Trend_Dir'].iloc[i]
            if raw_state != current_state:
                days_in_new_state += 1
                if days_in_new_state >= min_confirm_days:
                    current_state = raw_state
                    days_in_new_state = 0
                    for j in range(1, min_confirm_days + 1):
                        states[-j] = current_state
            else:
                days_in_new_state = 0
            states.append(current_state)

        df['Smooth_Trend'] = states
        df['Crossing'] = df['Smooth_Trend'] != df['Smooth_Trend'].shift(1)
        df.loc[0, 'Crossing'] = False

        df['Segment_ID'] = df['Crossing'].cumsum()

        segments_stats = []
        for seg_id in df['Segment_ID'].unique():
            segment = df[df['Segment_ID'] == seg_id]
            slope, intercept = calculate_trend_for_segment(segment, 'Value')

            if slope is not None:
                segments_stats.append({
                    'Segment_ID': seg_id,
                    'Start_Date': segment['Date'].iloc[0],
                    'End_Date': segment['Date'].iloc[-1],
                    'Days': len(segment),
                    'Slope': slope,
                    'Intercept': intercept,
                    'Trend': 'Рост' if slope > 0 else 'Спад'
                })

        segments_df = pd.DataFrame(segments_stats)
        median_segment_length = segments_df['Days'].median() if not segments_df.empty else 60

        last_segment_id = df['Segment_ID'].max()
        last_segment = df[df['Segment_ID'] == last_segment_id]
        last_segment_length = len(last_segment)

        st.subheader("Прогнозирование")

        last_stats = segments_df[segments_df['Segment_ID'] == last_segment_id].iloc[0]
        slope = last_stats['Slope']
        intercept = last_stats['Intercept']

        has_active_forecast = False
        forecast_df = pd.DataFrame()

        current_len = last_segment_length
        local_window = max(7, int(current_len * 0.3))  # 30% от тренда, но не меньше 7 дней
        last_n_days = df[df['Segment_ID'] == last_segment_id].tail(local_window)

        if len(last_n_days) >= 5:
            df_last_10 = last_n_days.copy()
            start_date_last_seg = df[df['Segment_ID'] == last_segment_id]['Date'].iloc[0]

            df_last_10['Days_From_Start'] = (df_last_10['Date'] - start_date_last_seg).dt.days
            x_fact = df_last_10['Days_From_Start'].values
            y_fact = df_last_10['Value'].values
            k_fact, b_fact = np.polyfit(x_fact, y_fact, 1)
            k_res, b_res = slope, intercept

            convergence_speed = abs(k_fact - k_res)
            is_attack_aggressive = convergence_speed > (abs(k_res) * 0.3)

            if slope < 0:
                trend_type = "Спад"
                line_name = "сопротивления"
                is_breakout_possible = (k_fact > k_res) and is_attack_aggressive
            else:
                trend_type = "Рост"
                line_name = "поддержки"
                is_breakout_possible = (k_fact < k_res) and is_attack_aggressive

            if is_breakout_possible:
                x_break = (b_fact - b_res) / (k_res - k_fact)
                current_duration = (last_segment['Date'].iloc[-1] - start_date_last_seg).days
                days_to_break = int(x_break - current_duration)
                validation_window = median_segment_length

                if 0 < days_to_break <= validation_window:
                    break_date = last_segment['Date'].iloc[-1] + pd.Timedelta(days=days_to_break)
                    st.success(
                        f"🔥 Ожидается пересечение линии {line_name} через **{days_to_break} дней**.\n\nРасчетная дата смены тренда: **{break_date.strftime('%d.%m.%Y')}**")

                    forecast_data = []
                    last_date = last_segment['Date'].iloc[-1]

                    for i in range(1, days_to_break + 1):
                        next_date = last_date + pd.Timedelta(days=i)
                        forecast_data.append({
                            'Date': next_date,
                            'Value': np.nan,
                            'MACD': np.nan,
                            'Signal': np.nan,
                            'Segment_ID': last_segment_id,
                            'is_forecast': True,
                            'Crossing': False
                        })
                    forecast_df = pd.DataFrame(forecast_data)
                    has_active_forecast = True

                elif days_to_break > validation_window:
                    st.info(
                        f"ℹ️ Локальный тренд сходится с линией {line_name}, но пробой ожидается через **{days_to_break} дней**. Это превышает медианное окно валидации ({validation_window:.1f} дней).")
                else:
                    st.info(
                        "ℹ️ Математическое пересечение находится в прошлом. Текущая динамика не предполагает пробоя в ближайшее время.")
            else:
                st.info(f"ℹ️ Текущая динамика сильнее линии {line_name}. Пробой пока невозможен (тренд продолжается).")
        else:
            st.warning("⚠️ Недостаточно данных для оценки локального напора (нужно минимум 5 дней в текущем сегменте).")

        df['is_forecast'] = False

        if has_active_forecast and not forecast_df.empty:
            df = pd.concat([df, forecast_df], ignore_index=True)

        df['Days_From_Start'] = (df['Date'] - df.groupby('Segment_ID')['Date'].transform('first')).dt.days

        df = df.merge(segments_df[['Segment_ID', 'Slope', 'Intercept']], on='Segment_ID', how='left')

        df['Trend_Value'] = df['Slope'] * df['Days_From_Start'] + df['Intercept']

        if has_active_forecast:
            all_segments = df['Segment_ID'].unique()
            for seg_id in all_segments:
                if seg_id != last_segment_id:
                    seg_indices = df[df['Segment_ID'] == seg_id].index
                    if len(seg_indices) > 0:
                        df.loc[seg_indices[-1], 'Trend_Value'] = np.nan
        else:
            last_in_segment = df.groupby('Segment_ID').tail(1).index
            df.loc[last_in_segment, 'Trend_Value'] = np.nan

        historical = df[df['is_forecast'] == False]
        forecast = df[df['is_forecast'] == True]

        if has_active_forecast and not forecast.empty:
            forecast_for_plot = pd.concat([historical.iloc[[-1]], forecast], ignore_index=True)
        else:
            forecast_for_plot = forecast

        st.subheader("📊 Анализ временного ряда и индикатор MACD")

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=('Исходные данные с линиями локальных трендов (МНК)', 'MACD и Signal Line')
        )

        fig.add_trace(go.Scatter(x=historical['Date'], y=historical['Value'], mode='lines', name='Значения (Y)',
                                 line=dict(color='blue', width=1)), row=1, col=1)

        fig.add_trace(
            go.Scatter(x=historical['Date'], y=historical['Trend_Value'], mode='lines', name='Линия тренда (МНК)',
                       line=dict(color='black', width=2), connectgaps=False), row=1, col=1)

        if has_active_forecast and not forecast_for_plot.empty:
            fig.add_trace(go.Scatter(
                x=forecast_for_plot['Date'],
                y=forecast_for_plot['Trend_Value'],
                mode='lines',
                name='Прогноз',
                line=dict(color='red', width=3, dash='solid')
            ), row=1, col=1)

        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD', line=dict(color='purple', width=2.5)),
            row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Signal'], mode='lines', name='Signal Line',
                                 line=dict(color='orange', width=1.5)), row=2, col=1)

        crossings = historical[historical['Crossing'] == True]
        fig.add_trace(go.Scatter(x=crossings['Date'], y=crossings['MACD'], mode='markers', name='Точки разворота',
                                 marker=dict(color='red', size=8, symbol='x')), row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

        fig.update_layout(height=850, hovermode='x unified',
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_yaxes(title_text='Значения', row=1, col=1)
        fig.update_yaxes(title_text='MACD', row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Таблица полученных сегментов"):
            st.write(f"Медианная длина тренда: **{median_segment_length:.1f} дней**")
            st.dataframe(segments_df)

else:
    st.info("Пожалуйста, загрузите файл с данными для начала работы. Поддерживаются форматы Excel (.xlsx) и CSV.")