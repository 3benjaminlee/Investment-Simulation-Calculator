import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Investment Simulation Calculator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Set theme to light mode
st.markdown("""
    <style>
        [data-testid="stSidebar"][aria-expanded="true"]{
            background-color: #f8f9fa;
        }
        [data-testid="stSidebar"][aria-expanded="false"]{
            background-color: #f8f9fa;
        }
        div[data-testid="stToolbar"] {
            visibility: hidden;
        }
        div[data-testid="stDecoration"] {
            visibility: hidden;
        }
        div[data-testid="stStatusWidget"] {
            visibility: hidden;
        }
        #MainMenu {
            visibility: hidden;
        }
        header {
            visibility: hidden;
        }
        footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# Custom CSS for premium look and mobile responsiveness
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 0 0 0 0; /* Removed top padding to reduce whitespace */
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Dark mode specific styles */
    [data-theme="dark"] {
        background-color: #262730;
    }

    [data-theme="dark"] .stRadio > label {
        color: #ffffff !important;
    }

    [data-theme="dark"] .stNumberInput > label {
        color: #ffffff !important;
    }

    [data-theme="dark"] .stSelectSlider > label {
        color: #ffffff !important;
    }

    [data-theme="dark"] .stMetric {
        background-color: #1e1e1e;
        border: 1px solid #404040;
    }

    [data-theme="dark"] .stMetric [data-testid="stMetricValue"] {
        color: #ffffff;
    }

    [data-theme="dark"] .stMetric [data-testid="stMetricLabel"] {
        color: #cccccc;
    }

    [data-theme="dark"] .stButton>button {
        background-color: #0066cc;
        color: white;
    }

    [data-theme="dark"] .stButton>button:hover {
        background-color: #0055aa;
    }

    /* Headings - both light and dark mode */
    h1 {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    
    @media (max-width: 768px) {
        h1 {
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }
    }
    
    h2 {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.2rem;
    }
    
    @media (max-width: 768px) {
        h2 {
            font-size: 1.2rem;
            margin-bottom: 0.8rem;
        }

        /* Improve mobile dark mode readability */
        [data-theme="dark"] .stRadio > div {
            background-color: #1e1e1e;
            border-radius: 4px;
            padding: 0.5rem;
        }

        [data-theme="dark"] .stNumberInput > div > div > input {
            background-color: #1e1e1e;
            color: #ffffff;
            border: 1px solid #404040;
        }

        [data-theme="dark"] .stSelectSlider > div {
            background-color: #1e1e1e;
            border-radius: 4px;
            padding: 0.5rem;
        }
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        border: none;
        padding: 0.8rem;
        font-weight: 500;
        border-radius: 4px;
        transition: all 0.2s ease;
        font-size: 1rem;
    }
    
    .stButton>button:hover {
        background-color: #0055aa;
    }
    
    /* Calculate button specific styling */
    .stButton>button[kind="primary"] {
        background-color: #0066cc;
        color: white !important;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 102, 204, 0.2);
    }
    
    .stButton>button[kind="primary"]:hover {
        background-color: #0055aa;
        box-shadow: 0 4px 8px rgba(0, 102, 204, 0.3);
        color: white !important;
    }
    
    .stButton>button[kind="primary"]:active {
        background-color: #004488;
        box-shadow: 0 2px 4px rgba(0, 102, 204, 0.2);
        color: white !important;
    }
    
    /* Input fields - Original styling */
    .stSelectbox, .stNumberInput>div>div>input, div[data-baseweb="select"] {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 0.3rem;
        font-size: 1rem;
        box-shadow: none;
    }
    
    /* Select box specific styling */
    div[data-baseweb="select"] {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        background-color: #f8f9fa;
    }
    
    /* Number input specific styling */
    .stNumberInput>div>div>input {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        background-color: #f8f9fa;
    }
    
    /* Hover states for all inputs */
    .stSelectbox:hover, .stNumberInput>div>div>input:hover, div[data-baseweb="select"]:hover {
        background-color: #f0f0f0;
    }
    
    /* Focus states for all inputs */
    .stSelectbox:focus, .stNumberInput>div>div>input:focus, div[data-baseweb="select"]:focus {
        background-color: #f0f0f0;
        box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.2);
    }
    
    /* Make all input containers look the same */
    .stSelectbox, .stNumberInput, div[data-baseweb="select"] {
        margin-bottom: 0rem; /* Removed margin for compact spacing */
    }
    
    /* Metrics */
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 4px;
        border: 1px solid #e0e0e0;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333333;
    }
    
    .stMetric [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #666666;
    }
    
    @media (max-width: 768px) {
        .stMetric {
            padding: 0.8rem;
        }
        
        .stMetric [data-testid="stMetricValue"] {
            font-size: 1.2rem;
        }
        
        .stMetric [data-testid="stMetricLabel"] {
            font-size: 0.8rem;
        }
    }
    
    /* Text */
    .stMarkdown {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #333333;
        line-height: 1.5;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background-color: #e0e0e0;
        margin: 1.5rem 0;
    }
    
    /* Success and warning messages */
    .stAlert {
        border-radius: 4px;
        padding: 0.8rem;
        margin-bottom: 1rem;
    }
    
    /* Chart container */
    .element-container {
        margin-bottom: 0.5rem; /* Reduced margin for compact spacing */
    }
    
    /* Columns */
    .row-widget.stHorizontal {
        gap: 1.5rem;
    }
    
    /* Mobile-specific adjustments */
    @media (max-width: 768px) {
        .row-widget.stHorizontal {
            gap: 1rem;
        }
        
        .element-container {
            margin-bottom: 0.5rem; /* Reduced margin for compact spacing */
        }
        
        /* Make inputs more touch-friendly */
        .stSelectbox, .stNumberInput>div>div>input, div[data-baseweb="select"] {
            padding: 0.5rem;
            min-height: 44px; /* Minimum touch target size */
        }
        
        /* Adjust button size for touch */
        .stButton>button {
            padding: 0.9rem;
            min-height: 44px; /* Minimum touch target size */
        }
        
        /* Adjust expander for mobile */
        .streamlit-expanderHeader {
            font-size: 0.9rem;
            padding: 0.7rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Top stocks and S&P 500 data with default return rates
TOP_STOCKS = {
    "Custom Return Rate": None,
    "S&P 500 (^GSPC)": {"return": 10.0, "volatility": 15.0},
    "Apple (AAPL)": {"return": 15.0, "volatility": 25.0},
    "Microsoft (MSFT)": {"return": 18.0, "volatility": 22.0},
    "Amazon (AMZN)": {"return": 20.0, "volatility": 30.0},
    "Alphabet (GOOGL)": {"return": 16.0, "volatility": 24.0},
    "Meta (META)": {"return": 22.0, "volatility": 35.0},
    "Tesla (TSLA)": {"return": 25.0, "volatility": 45.0},
    "NVIDIA (NVDA)": {"return": 30.0, "volatility": 40.0},
    "Johnson & Johnson (JNJ)": {"return": 8.0, "volatility": 15.0},
    "Procter & Gamble (PG)": {"return": 9.0, "volatility": 14.0}
}

def calculate_investment(
    initial_investment,
    regular_contribution,
    contribution_frequency,
    investment_term,
    expected_return,
    compounding_frequency
):
    """Calculate investment growth over time."""
    # Convert annual rate to periodic rate
    periods_per_year = {
        'Annually': 1,
        'Semi-Annually': 2,
        'Quarterly': 4,
        'Monthly': 12,
        'Daily': 252
    }
    
    n = periods_per_year[compounding_frequency]
    r = expected_return / 100 / n  # Convert percentage to decimal and adjust for compounding frequency
    
    # Calculate number of periods
    t = investment_term * n
    
    # Calculate contribution periods
    contribution_periods = {
        'Monthly': 12,
        'Bi-Weekly': 26,
        'Quarterly': 4,
        'Annually': 1
    }
    m = contribution_periods[contribution_frequency]
    
    # Calculate future value
    future_value = initial_investment * (1 + r)**t
    future_value += regular_contribution * ((1 + r)**t - 1) / r * (m/n)
    
    total_contributions = initial_investment + (regular_contribution * m * investment_term)
    total_interest = future_value - total_contributions
    
    return {
        'future_value': future_value,
        'total_contributions': total_contributions,
        'total_interest': total_interest
    }

def generate_historical_data(ticker, period, return_rate, volatility):
    """Generate synthetic historical data based on default values."""
    # Calculate number of days
    period_days = {
        '1y': 365,
        '3y': 365*3,
        '5y': 365*5,
        '10y': 365*10,
        'max': 365*30
    }
    days = period_days.get(period, 365*10)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate price data using random walk with drift
    np.random.seed(42)  # For reproducibility
    daily_return = return_rate / 100 / 252  # Convert annual return to daily
    daily_volatility = volatility / 100 / np.sqrt(252)  # Convert annual volatility to daily
    
    # Generate random returns
    random_returns = np.random.normal(daily_return, daily_volatility, size=len(dates))
    
    # Calculate price series
    initial_price = 100  # Arbitrary starting price
    price_series = initial_price * (1 + random_returns).cumprod()
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': price_series,
        'High': price_series * (1 + np.random.uniform(0, 0.02, size=len(dates))),
        'Low': price_series * (1 - np.random.uniform(0, 0.02, size=len(dates))),
        'Close': price_series,
        'Volume': np.random.randint(1000000, 10000000, size=len(dates))
    }, index=dates)
    
    # Adjust high and low to ensure they're above/below close
    df['High'] = df[['High', 'Close']].max(axis=1)
    df['Low'] = df[['Low', 'Close']].min(axis=1)
    
    return df

def calculate_stock_metrics(data):
    """Calculate stock performance metrics."""
    if data is None or len(data) < 2:
        return None
    
    # Calculate daily returns
    data['Returns'] = data['Close'].pct_change()
    
    # Calculate annualized metrics
    years = len(data) / 252  # Assuming 252 trading days per year
    total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
    annual_return = (1 + total_return) ** (1/years) - 1
    volatility = data['Returns'].std() * np.sqrt(252)  # Annualized volatility
    
    return {
        'annual_return': annual_return * 100,  # Convert to percentage
        'volatility': volatility * 100,  # Convert to percentage
        'total_return': total_return * 100  # Convert to percentage
    }

# Initialize session state for input values
if 'initial_investment' not in st.session_state:
    st.session_state.initial_investment = 0
if 'regular_contribution' not in st.session_state:
    st.session_state.regular_contribution = 0

# Main app
st.title("Investment Simulation Calculator - AI Side Project")

# Create main layout - responsive columns
left_col, right_col = st.columns([1, 1])

with left_col:
    # Investment Calculator section
    # st.header("Investment Parameters")
    
    # Input fields
    initial_investment = st.number_input(
        "Initial Investment ($)",
        min_value=0,
        value=st.session_state.initial_investment,
        step=1000,
        format="%d"
    )
    
    regular_contribution = st.number_input(
        "Contribution Amount ($)",
        min_value=0,
        value=st.session_state.regular_contribution,
        step=100,
        format="%d"
    )
    
    # Replace selectbox with radio buttons for contribution frequency
    contribution_frequency = st.radio(
        "Contribution Frequency",
        ['Bi-Weekly', 'Monthly', 'Quarterly', 'Annually'],
        horizontal=True
    )
    
    # Use select slider for investment term
    investment_term = st.select_slider(
        "Investment Term (Years)",
        options=list(range(1, 51)),
        value=10
    )
    
    # Stock selection with radio buttons for better mobile experience
    selected_stock = st.radio(
        "Select Investment Strategy",
        list(TOP_STOCKS.keys())
    )
    
    # Analysis period selector with radio buttons
    period = st.radio(
        "Historical Analysis Period To Use Future Simulation",
        ['1y', '3y', '5y', '10y', 'max'],
        horizontal=True
    )
    
    # Get stock data and calculate historical return rate
    if selected_stock != "Custom Return Rate":
        stock_data = TOP_STOCKS[selected_stock]
        return_rate = stock_data["return"]
        volatility = stock_data["volatility"]
        
        # Generate synthetic historical data
        data = generate_historical_data(selected_stock, period, return_rate, volatility)
        
        if data is not None:
            metrics = calculate_stock_metrics(data)
            
            if metrics:
                # Use the calculated historical return rate for projections
                expected_return = metrics['annual_return']
                st.success(f"Using historical return rate of {expected_return:.2f}% from {period} analysis")
            else:
                # Fallback to predefined return rate if metrics calculation fails
                expected_return = stock_data["return"]
                st.warning(f"Could not calculate historical return rate. Using default of {expected_return:.2f}%")
        else:
            # Fallback to predefined return rate if data generation fails
            expected_return = stock_data["return"]
            st.warning(f"Could not generate historical data. Using default return rate of {expected_return:.2f}%")
    else:
        expected_return = st.number_input(
            "Expected Annual Return (%)",
            min_value=0.0,
            value=7.0,
            step=0.5
        )
    
    compounding_frequency = st.radio(
        "Compounding Frequency",
        ['Daily', 'Monthly', 'Quarterly', 'Semi-Annually', 'Annually'],
        horizontal=True
    )
    
    if st.button("Calculate", key="calc_button"):
        results = calculate_investment(
            initial_investment,
            regular_contribution,
            contribution_frequency,
            investment_term,
            expected_return,
            compounding_frequency
        )
        
        # Display results
        st.subheader("Investment Results")
        
        # Display metrics directly without nesting
        st.metric("Total Contributions", f"${results['total_contributions']:,.2f}")
        st.metric("Total Interest Earned", f"${results['total_interest']:,.2f}")
        st.metric("Final Value", f"${results['future_value']:,.2f}")

with right_col:
    # Historical Analysis section
    if selected_stock != "Custom Return Rate":
        st.header("Historical Performance")
        
        # Get stock data
        stock_data = TOP_STOCKS[selected_stock]
        return_rate = stock_data["return"]
        volatility = stock_data["volatility"]
        
        # Generate synthetic historical data
        data = generate_historical_data(selected_stock, period, return_rate, volatility)
        
        if data is not None:
            metrics = calculate_stock_metrics(data)
            
            if metrics:
                st.subheader(f"{selected_stock} Performance Metrics")
                
                # Display metrics directly without nesting
                st.metric("Average Annual Return", f"{metrics['annual_return']:.2f}%")
                st.metric("Volatility", f"{metrics['volatility']:.2f}%")
                st.metric("Total Return", f"{metrics['total_return']:.2f}%")
                
                # Create price chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#000000', width=2)
                ))
                fig.update_layout(
                    title=f"{selected_stock} Historical Price",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=400,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Arial", size=12),
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
                st.plotly_chart(fig, use_container_width=True)

# Disclaimer at the bottom of the page
st.markdown("---")
with st.expander("Terms of Use & Disclaimer", expanded=False):
    st.markdown("""
    <p style='color: #666666; font-size: 0.9rem; margin: 0;'>
        <strong>Disclaimer:</strong> This investment calculator is provided for educational and entertainment purposes only. 
        The calculations and projections shown are based on simplified assumptions and historical data, and should not be 
        considered as financial advice. Past performance does not guarantee future results. Always consult with qualified 
        financial professionals before making investment decisions. The creator of this calculator assumes no responsibility 
        for any financial decisions made based on the information provided here.
    </p>
    """, unsafe_allow_html=True) 