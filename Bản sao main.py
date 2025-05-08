import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Thiết lập trang
st.set_page_config(page_title="Phân Tích Cổ Phiếu FPT & CMG", layout="wide")


# Tạo dữ liệu demo cho cổ phiếu
def generate_stock_data():
    np.random.seed(42)

    # Tạo dữ liệu ngẫu nhiên nhưng có xu hướng cho FPT và CMG
    date_range = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")

    # Tạo dữ liệu FPT với xu hướng tăng
    fpt_base = np.linspace(50, 80, len(date_range))
    fpt_noise = np.random.normal(0, 2, len(date_range))
    fpt_prices = fpt_base + fpt_noise

    # Tạo dữ liệu CMG với biến động mạnh
    cmg_base = np.sin(np.linspace(0, 10, len(date_range))) * 15 + 60
    cmg_noise = np.random.normal(0, 3, len(date_range))
    cmg_prices = cmg_base + cmg_noise

    # Tạo volume ngẫu nhiên
    volume = np.random.randint(100000, 500000, len(date_range))

    # Tạo dataframe
    fpt_data = pd.DataFrame({
        'Ngày': date_range,
        'Mã': 'FPT',
        'Giá đóng cửa': fpt_prices,
        'Giá mở cửa': fpt_prices - np.random.uniform(0.5, 2, len(date_range)),
        'Cao nhất': fpt_prices + np.random.uniform(0.5, 1.5, len(date_range)),
        'Thấp nhất': fpt_prices - np.random.uniform(0.5, 1.5, len(date_range)),
        'Khối lượng': volume,
        'Thay đổi (%)': np.random.normal(0.2, 0.8, len(date_range))
    })

    cmg_data = pd.DataFrame({
        'Ngày': date_range,
        'Mã': 'CMG',
        'Giá đóng cửa': cmg_prices,
        'Giá mở cửa': cmg_prices - np.random.uniform(0.5, 3, len(date_range)),
        'Cao nhất': cmg_prices + np.random.uniform(0.5, 2.5, len(date_range)),
        'Thấp nhất': cmg_prices - np.random.uniform(0.5, 2.5, len(date_range)),
        'Khối lượng': volume * 1.5,
        'Thay đổi (%)': np.random.normal(0, 1.2, len(date_range))
    })

    return pd.concat([fpt_data, cmg_data])


# Tạo dữ liệu chỉ số tài chính
def generate_financials():
    metrics = ['P/E', 'P/B', 'ROE', 'ROA', 'EPS', 'Beta']
    fpt_values = [15.2, 3.8, 22.5, 12.1, 4500, 1.2]
    cmg_values = [18.6, 4.5, 18.3, 9.8, 3200, 1.8]

    return pd.DataFrame({
        'Chỉ số': metrics * 2,
        'Giá trị': fpt_values + cmg_values,
        'Mã': ['FPT'] * len(metrics) + ['CMG'] * len(metrics)
    })


# Sidebar menu
def sidebar_controls():
    st.sidebar.title("Menu Phân Tích")

    selected_stock = st.sidebar.selectbox(
        "Chọn mã cổ phiếu",
        ['FPT', 'CMG', 'So sánh cả hai']
    )

    analysis_type = st.sidebar.radio(
        "Loại phân tích",
        ['Biểu đồ giá', 'Chỉ số tài chính', 'Khối lượng giao dịch', 'Phân tích kỹ thuật']
    )

    date_range = st.sidebar.date_input(
        "Chọn khoảng thời gian",
        value=[datetime(2023, 1, 1), datetime(2023, 12, 31)]
    )

    show_volume = st.sidebar.checkbox("Hiển thị khối lượng giao dịch", True)

    return {
        'selected_stock': selected_stock,
        'analysis_type': analysis_type,
        'date_range': date_range,
        'show_volume': show_volume
    }


# Tạo biểu đồ giá
def render_price_chart(data, params):
    filtered_data = data[
        (data['Ngày'] >= pd.to_datetime(params['date_range'][0])) &
        (data['Ngày'] <= pd.to_datetime(params['date_range'][1]))
        ]

    if params['selected_stock'] != 'So sánh cả hai':
        filtered_data = filtered_data[filtered_data['Mã'] == params['selected_stock']]

    fig = go.Figure()

    if params['selected_stock'] == 'So sánh cả hai':
        for stock in ['FPT', 'CMG']:
            stock_data = filtered_data[filtered_data['Mã'] == stock]
            fig.add_trace(go.Scatter(
                x=stock_data['Ngày'],
                y=stock_data['Giá đóng cửa'],
                name=stock,
                mode='lines'
            ))
    else:
        # Biểu đồ nến nếu chọn 1 mã
        fig = go.Figure(go.Candlestick(
            x=filtered_data['Ngày'],
            open=filtered_data['Giá mở cửa'],
            high=filtered_data['Cao nhất'],
            low=filtered_data['Thấp nhất'],
            close=filtered_data['Giá đóng cửa'],
            name=params['selected_stock']
        ))

        if params['show_volume']:
            fig.add_trace(go.Bar(
                x=filtered_data['Ngày'],
                y=filtered_data['Khối lượng'],
                name='Khối lượng',
                yaxis='y2',
                marker_color='rgba(100, 150, 200, 0.6)'
            ))

            fig.update_layout(
                yaxis2=dict(
                    title='Khối lượng',
                    overlaying='y',
                    side='right',
                    showgrid=False
                )
            )

    fig.update_layout(
        title=f"Biểu đồ giá {' - '.join([params['selected_stock']] if params['selected_stock'] != 'So sánh cả hai' else ['FPT', 'CMG'])}",
        xaxis_title='Ngày',
        yaxis_title='Giá (nghìn VND)',
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)


# Tạo bảng chỉ số tài chính
def render_financials(financials_data, params):
    if params['selected_stock'] == 'So sánh cả hai':
        display_data = financials_data
    else:
        display_data = financials_data[financials_data['Mã'] == params['selected_stock']]

    st.subheader(f"Chỉ số tài chính {params['selected_stock']}")

    # Hiển thị dưới dạng bảng
    st.dataframe(
        display_data.pivot(index='Chỉ số', columns='Mã', values='Giá trị'),
        use_container_width=True
    )

    # Hiển thị dưới dạng biểu đồ cột
    fig = px.bar(
        display_data,
        x='Chỉ số',
        y='Giá trị',
        color='Mã',
        barmode='group',
        text='Giá trị'
    )

    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        title='So sánh chỉ số tài chính',
        yaxis_title='Giá trị'
    )

    st.plotly_chart(fig, use_container_width=True)


# Tạo biểu đồ khối lượng
def render_volume_chart(data, params):
    filtered_data = data[
        (data['Ngày'] >= pd.to_datetime(params['date_range'][0])) &
        (data['Ngày'] <= pd.to_datetime(params['date_range'][1]))
        ]

    if params['selected_stock'] != 'So sánh cả hai':
        filtered_data = filtered_data[filtered_data['Mã'] == params['selected_stock']]

    fig = px.bar(
        filtered_data,
        x='Ngày',
        y='Khối lượng',
        color='Mã' if params['selected_stock'] == 'So sánh cả hai' else None,
        title=f"Khối lượng giao dịch {params['selected_stock']}"
    )

    st.plotly_chart(fig, use_container_width=True)


# Tạo phân tích kỹ thuật
def render_technical_analysis(data, params):
    if params['selected_stock'] == 'So sánh cả hai':
        st.warning("Vui lòng chọn 1 mã cổ phiếu để xem phân tích kỹ thuật")
        return

    filtered_data = data[
        (data['Mã'] == params['selected_stock']) &
        (data['Ngày'] >= pd.to_datetime(params['date_range'][0])) &
        (data['Ngày'] <= pd.to_datetime(params['date_range'][1]))
        ].set_index('Ngày')

    # Tính toán các chỉ báo kỹ thuật đơn giản
    filtered_data['MA20'] = filtered_data['Giá đóng cửa'].rolling(window=20).mean()
    filtered_data['MA50'] = filtered_data['Giá đóng cửa'].rolling(window=50).mean()
    filtered_data['RSI'] = 70 - (filtered_data['Thay đổi (%)'].rolling(window=14).mean() * 10)

    fig = go.Figure()

    # Giá và đường MA
    fig.add_trace(go.Scatter(
        x=filtered_data.index,
        y=filtered_data['Giá đóng cửa'],
        name='Giá đóng cửa',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=filtered_data.index,
        y=filtered_data['MA20'],
        name='MA 20 ngày',
        line=dict(color='orange', dash='dot')
    ))

    fig.add_trace(go.Scatter(
        x=filtered_data.index,
        y=filtered_data['MA50'],
        name='MA 50 ngày',
        line=dict(color='red', dash='dot')
    ))

    # Tạo layout với subplot cho RSI
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=('Giá và đường MA', 'RSI'))

    # Giá và đường MA (subplot 1)
    fig.add_trace(go.Scatter(
        x=filtered_data.index,
        y=filtered_data['Giá đóng cửa'],
        name='Giá đóng cửa',
        line=dict(color='blue')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=filtered_data.index,
        y=filtered_data['MA20'],
        name='MA 20 ngày',
        line=dict(color='orange', dash='dot')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=filtered_data.index,
        y=filtered_data['MA50'],
        name='MA 50 ngày',
        line=dict(color='red', dash='dot')
    ), row=1, col=1)

    # RSI (subplot 2)
    fig.add_trace(go.Scatter(
        x=filtered_data.index,
        y=filtered_data['RSI'],
        name='RSI',
        line=dict(color='purple')
    ), row=2, col=1)

    # Đường 70 và 30 cho RSI
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    fig.update_layout(
        title=f"Phân tích kỹ thuật {params['selected_stock']}",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)


# Main app
def main():
    st.title("📈 Phân Tích Cổ Phiếu FPT & CMG")

    # Tạo dữ liệu
    stock_data = generate_stock_data()
    financials_data = generate_financials()

    # Hiển thị sidebar và lấy tham số
    params = sidebar_controls()

    # Hiển thị thông tin cơ bản
    st.subheader(f"Thông tin phân tích: {params['selected_stock']}")

    # Hiển thị nội dung theo lựa chọn
    if params['analysis_type'] == 'Biểu đồ giá':
        render_price_chart(stock_data, params)
    elif params['analysis_type'] == 'Chỉ số tài chính':
        render_financials(financials_data, params)
    elif params['analysis_type'] == 'Khối lượng giao dịch':
        render_volume_chart(stock_data, params)
    elif params['analysis_type'] == 'Phân tích kỹ thuật':
        render_technical_analysis(stock_data, params)

    # Hiển thị dữ liệu thô (tùy chọn)
    with st.expander("Xem dữ liệu thô"):
        st.dataframe(stock_data[stock_data['Mã'] == params['selected_stock']] if params[
                                                                                     'selected_stock'] != 'So sánh cả hai' else stock_data)


if __name__ == "__main__":
    from plotly.subplots import make_subplots

    main()