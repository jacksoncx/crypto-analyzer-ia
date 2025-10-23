"""
Backend Flask para Análise de Criptomoedas com Análise Técnica, Machine Learning e Sentimento de Mercado
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import warnings
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuração da API Alpha Vantage
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
ALPHA_VANTAGE_BASE_URL = 'https://www.alphavantage.co/query'

# Configuração da API Alternative.me (Fear and Greed Index)
FEAR_GREED_API_URL = 'https://api.alternative.me/fng/'

# Cache para armazenar dados históricos (evita múltiplas chamadas à API)
data_cache = {}
fear_greed_cache = {'value': None, 'timestamp': None}

def fetch_fear_greed_index():
    """
    Busca o Índice de Medo e Ganância da API Alternative.me.
    Retorna um valor de 0-100 onde 0 = Extreme Fear e 100 = Extreme Greed.
    """
    global fear_greed_cache
    
    try:
        # Verificar se o cache ainda é válido (atualizado há menos de 1 hora)
        if fear_greed_cache['value'] is not None and fear_greed_cache['timestamp'] is not None:
            if (datetime.now() - fear_greed_cache['timestamp']).seconds < 3600:
                return fear_greed_cache['value']
        
        response = requests.get(FEAR_GREED_API_URL, params={'limit': 1}, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'data' in data and len(data['data']) > 0:
            value = int(data['data'][0]['value'])
            fear_greed_cache['value'] = value
            fear_greed_cache['timestamp'] = datetime.now()
            return value
        
        return 50  # Valor neutro padrão
    
    except Exception as e:
        print(f"Erro ao buscar Fear and Greed Index: {e}")
        return 50  # Valor neutro padrão em caso de erro

def fetch_crypto_data(symbol, market='USD'):
    """
    Busca dados históricos de criptomoedas da API Alpha Vantage.
    """
    try:
        params = {
            'function': 'DIGITAL_CURRENCY_DAILY',
            'symbol': symbol.upper(),
            'market': market.upper(),
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'Time Series (Digital Currency Daily)' not in data:
            return None
        
        return data['Time Series (Digital Currency Daily)']
    
    except Exception as e:
        print(f"Erro ao buscar dados: {e}")
        return None

def calculate_rsi(prices, period=14):
    """
    Calcula o Índice de Força Relativa (RSI).
    RSI = 100 - (100 / (1 + RS))
    onde RS = Ganho Médio / Perda Média
    """
    if len(prices) < period + 1:
        return None
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100 if avg_gain > 0 else 0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calcula o MACD (Convergência/Divergência de Médias Móveis).
    """
    if len(prices) < slow:
        return None, None, None
    
    prices_array = np.array(prices)
    
    # Calcular EMAs (Médias Móveis Exponenciais)
    ema_fast = pd.Series(prices_array).ewm(span=fast).mean().values[-1]
    ema_slow = pd.Series(prices_array).ewm(span=slow).mean().values[-1]
    
    # MACD = EMA Rápida - EMA Lenta
    macd = ema_fast - ema_slow
    
    # Signal Line = EMA de 9 períodos do MACD
    signal_line = macd * 0.8  # Aproximação simplificada
    
    # Histogram = MACD - Signal Line
    histogram = macd - signal_line
    
    return macd, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """
    Calcula as Bandas de Bollinger.
    """
    if len(prices) < period:
        return None, None, None
    
    prices_array = np.array(prices[-period:])
    sma = np.mean(prices_array)
    std = np.std(prices_array)
    
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    
    return upper_band, sma, lower_band

def calculate_moving_averages(prices, short=20, long=50):
    """
    Calcula as Médias Móveis Simples (SMA).
    """
    if len(prices) < long:
        return None, None
    
    prices_array = np.array(prices)
    sma_short = np.mean(prices_array[-short:])
    sma_long = np.mean(prices_array[-long:])
    
    return sma_short, sma_long

def prepare_features_for_ml(prices, volumes):
    """
    Prepara os features para o modelo de Machine Learning.
    """
    features = []
    
    if len(prices) < 50:
        return None
    
    # Calcular indicadores técnicos
    rsi = calculate_rsi(prices)
    macd, signal, histogram = calculate_macd(prices)
    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(prices)
    sma_20, sma_50 = calculate_moving_averages(prices)
    
    # Variação percentual de 24h
    price_change_24h = ((prices[-1] - prices[-2]) / prices[-2]) * 100 if len(prices) > 1 else 0
    
    # Variação percentual de 7 dias
    price_change_7d = ((prices[-1] - prices[-7]) / prices[-7]) * 100 if len(prices) > 7 else 0
    
    # Volume médio
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
    
    # Volatilidade (desvio padrão dos últimos 20 dias)
    volatility = np.std(prices[-20:]) if len(prices) >= 20 else np.std(prices)
    
    # Normalizar features
    features_dict = {
        'rsi': rsi if rsi is not None else 50,
        'macd': macd if macd is not None else 0,
        'price_change_24h': price_change_24h,
        'price_change_7d': price_change_7d,
        'volatility': volatility,
        'sma_ratio': (sma_20 / sma_50 - 1) * 100 if sma_50 and sma_20 else 0
    }
    
    return features_dict

def train_prediction_model(prices, volumes, fear_greed_value=50):
    """
    Treina um modelo de Machine Learning para prever a direção do preço.
    Inclui o Fear and Greed Index como feature adicional.
    """
    if len(prices) < 50:
        return None
    
    try:
        # Preparar dados de treinamento
        X = []
        y = []
        
        for i in range(30, len(prices) - 1):
            # Features baseados nos últimos 30 dias
            window_prices = prices[i-30:i]
            window_volumes = volumes[i-30:i] if i < len(volumes) else [0] * 30
            
            rsi = calculate_rsi(window_prices)
            macd, _, _ = calculate_macd(window_prices)
            price_change = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
            volatility = np.std(window_prices)
            
            # Normalizar o Fear and Greed Index para a escala 0-1
            fear_greed_normalized = fear_greed_value / 100.0
            
            feature_set = [
                rsi if rsi is not None else 50,
                macd if macd is not None else 0,
                price_change,
                volatility,
                fear_greed_normalized
            ]
            
            X.append(feature_set)
            
            # Label: 1 se o preço subiu no próximo dia, 0 caso contrário
            next_price_change = ((prices[i+1] - prices[i]) / prices[i]) * 100
            y.append(1 if next_price_change > 0 else 0)
        
        if len(X) < 10:
            return None
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalizar features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Treinar modelo (Regressão Logística é mais leve e rápido)
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_scaled, y)
        
        return model, scaler
    
    except Exception as e:
        print(f"Erro ao treinar modelo: {e}")
        return None

def predict_direction(prices, volumes, fear_greed_value=50):
    """
    Usa o modelo treinado para prever a direção do preço.
    Inclui o Fear and Greed Index como feature.
    """
    if len(prices) < 50:
        # Usar análise técnica simples se não houver dados suficientes
        return {
            'prediction_percent': 50,
            'direction': 'Neutro',
            'confidence': 'Baixa'
        }
    
    try:
        model_result = train_prediction_model(prices, volumes, fear_greed_value)
        
        if model_result is None:
            return {
                'prediction_percent': 50,
                'direction': 'Neutro',
                'confidence': 'Baixa'
            }
        
        model, scaler = model_result
        
        # Preparar features para o último dia
        window_prices = prices[-30:]
        window_volumes = volumes[-30:] if len(volumes) >= 30 else volumes
        
        rsi = calculate_rsi(window_prices)
        macd, _, _ = calculate_macd(window_prices)
        price_change = ((prices[-1] - prices[-2]) / prices[-2]) * 100 if len(prices) > 1 else 0
        volatility = np.std(window_prices)
        
        # Normalizar o Fear and Greed Index
        fear_greed_normalized = fear_greed_value / 100.0
        
        feature_set = np.array([[
            rsi if rsi is not None else 50,
            macd if macd is not None else 0,
            price_change,
            volatility,
            fear_greed_normalized
        ]])
        
        feature_scaled = scaler.transform(feature_set)
        
        # Prever probabilidade
        probability = model.predict_proba(feature_scaled)[0]
        
        # Probabilidade de subir (classe 1)
        prob_up = probability[1] * 100
        
        # Determinar confiança
        max_prob = max(probability)
        confidence = 'Alta' if max_prob > 0.65 else 'Média' if max_prob > 0.55 else 'Baixa'
        
        return {
            'prediction_percent': int(prob_up),
            'direction': 'SUBIR' if prob_up > 50 else 'DESCER',
            'confidence': confidence
        }
    
    except Exception as e:
        print(f"Erro na previsão: {e}")
        return {
            'prediction_percent': 50,
            'direction': 'Neutro',
            'confidence': 'Baixa'
        }

@app.route('/api/analyze/<symbol>', methods=['GET'])
def analyze_crypto(symbol):
    """
    Endpoint principal para análise de criptomoedas.
    """
    market = request.args.get('market', 'USD')
    
    try:
        # Buscar dados históricos
        time_series = fetch_crypto_data(symbol, market)
        
        if not time_series:
            return jsonify({
                'error': 'Criptomoeda não encontrada ou limite de requisições atingido'
            }), 404
        
        # Buscar o Índice de Medo e Ganância
        fear_greed_value = fetch_fear_greed_index()
        
        # Processar dados
        dates = sorted(time_series.keys())
        prices = []
        volumes = []
        
        for date in dates:
            data = time_series[date]
            prices.append(float(data[f'4a. close ({market})']))
            volumes.append(float(data[f'5. volume']))
        
        # Inverter para ter os dados mais recentes no final
        prices = prices[::-1]
        volumes = volumes[::-1]
        
        current_price = prices[-1]
        previous_price = prices[-2] if len(prices) > 1 else current_price
        
        # Calcular variações
        change_24h = ((current_price - previous_price) / previous_price) * 100 if previous_price > 0 else 0
        change_7d = ((current_price - prices[-7]) / prices[-7]) * 100 if len(prices) > 7 and prices[-7] > 0 else 0
        
        # Calcular indicadores técnicos
        rsi = calculate_rsi(prices)
        macd, signal, histogram = calculate_macd(prices)
        upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(prices)
        sma_20, sma_50 = calculate_moving_averages(prices)
        
        # Interpretação de indicadores
        rsi_status = 'Sobrecomprado' if rsi and rsi > 70 else 'Sobrevendido' if rsi and rsi < 30 else 'Neutro'
        macd_status = 'Positivo' if macd and macd > 0 else 'Negativo'
        
        # Previsão com IA (incluindo Fear and Greed Index)
        prediction = predict_direction(prices, volumes, fear_greed_value)
        
        # Calcular volume médio e razão
        avg_volume_20d = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        volume_ratio = (volumes[-1] / avg_volume_20d - 1) * 100 if avg_volume_20d > 0 else 0
        
        # Interpretação do Fear and Greed Index
        if fear_greed_value < 25:
            fear_greed_status = 'Extreme Fear'
        elif fear_greed_value < 45:
            fear_greed_status = 'Fear'
        elif fear_greed_value < 55:
            fear_greed_status = 'Neutral'
        elif fear_greed_value < 75:
            fear_greed_status = 'Greed'
        else:
            fear_greed_status = 'Extreme Greed'
        
        # Preparar resposta
        response = {
            'symbol': symbol.upper(),
            'market': market,
            'current_price': round(current_price, 8),
            'change_24h': round(change_24h, 2),
            'change_7d': round(change_7d, 2),
            'technical_indicators': {
                'rsi': round(rsi, 2) if rsi else None,
                'rsi_status': rsi_status,
                'macd': round(macd, 8) if macd else None,
                'macd_status': macd_status,
                'bollinger_upper': round(upper_bb, 8) if upper_bb else None,
                'bollinger_middle': round(middle_bb, 8) if middle_bb else None,
                'bollinger_lower': round(lower_bb, 8) if lower_bb else None,
                'sma_20': round(sma_20, 8) if sma_20 else None,
                'sma_50': round(sma_50, 8) if sma_50 else None,
            },
            'volume': {
                'current': round(volumes[-1], 2),
                'average_20d': round(avg_volume_20d, 2),
                'ratio_percent': round(volume_ratio, 2)
            },
            'market_sentiment': {
                'fear_greed_index': fear_greed_value,
                'fear_greed_status': fear_greed_status,
                'description': 'Índice de Medo e Ganância do mercado (0=Extreme Fear, 100=Extreme Greed)'
            },
            'prediction': {
                'percent': prediction['prediction_percent'],
                'direction': prediction['direction'],
                'confidence': prediction['confidence']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """
    Endpoint de saúde da API.
    """
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/', methods=['GET'])
def index():
    """
    Rota raiz retorna informações da API.
    """
    return jsonify({
        'name': 'Crypto Analyzer API',
        'version': '2.0.0',
        'features': [
            'Análise Técnica Avançada (RSI, MACD, Bandas de Bollinger)',
            'Machine Learning com Fear and Greed Index',
            'Previsão de Direção de Preço com IA'
        ],
        'endpoints': {
            '/api/analyze/<symbol>': 'Analisar uma criptomoeda (ex: /api/analyze/BTC?market=USD)',
            '/api/health': 'Verificar saúde da API'
        }
    }), 200

if __name__ == '__main__':
    # Para desenvolvimento
    app.run(debug=True, host='0.0.0.0', port=5000)

