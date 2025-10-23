# 📊 Analisador de Criptomoedas - IA & Análise Técnica (v2.0)

Um site completo de análise de criptomoedas com **Análise Técnica Avançada**, **Sentimento de Mercado** e **Machine Learning** para previsão de direção de preços.

## 🎯 Características

- **Análise Técnica Avançada**: RSI, MACD, Bandas de Bollinger, Médias Móveis (Dados da Alpha Vantage API)
- **Sentimento de Mercado**: Integração com o **Fear & Greed Index** (Alternative.me)
- **Machine Learning**: Modelo de Regressão Logística treinado com **Indicadores Técnicos e Sentimento**
- **Dados Reais**: Integração com Alpha Vantage API para dados históricos de criptomoedas
- **Interface Moderna**: Design responsivo e intuitivo
- **Sem Login**: Acesso imediato, sem necessidade de autenticação
- **Gratuito**: Totalmente gratuito para usar e hospedar

## 🚀 Deploy Gratuito

### Opção 1: Render (Recomendado)

1. **Crie uma conta** em [render.com](https://render.com)
2. **Conecte seu repositório GitHub** (faça um fork ou crie um novo repositório)
3. **Crie um novo Web Service**:
   - Selecione o repositório
   - Environment: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
4. **Adicione a variável de ambiente**:
   - `ALPHA_VANTAGE_API_KEY`: Sua chave da Alpha Vantage (obtenha em https://www.alphavantage.co/support/#api-key)
5. **Deploy automático** ao fazer push no GitHub

### Opção 2: Vercel (Para Frontend)

Se quiser hospedar apenas o frontend em Vercel:

1. **Crie uma conta** em [vercel.com](https://vercel.com)
2. **Importe o repositório**
3. **Configure a variável de ambiente** para apontar para o backend (ex: `https://seu-render-app.onrender.com`)

## 🛠️ Instalação Local

### Pré-requisitos

- Python 3.8+
- pip

### Passos

1. **Clone o repositório**:
```bash
git clone https://github.com/seu-usuario/crypto-analyzer.git
cd crypto-analyzer
```

2. **Crie um ambiente virtual**:
```bash
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

4. **Configure a variável de ambiente**:
```bash
cp .env.example .env
# Edite .env e adicione sua chave da Alpha Vantage
```

5. **Execute o servidor**:
```bash
python3 app.py
```

6. **Acesse** em `http://localhost:5000`

## 📚 API Endpoints

### GET `/api/analyze/<symbol>`

Analisa uma criptomoeda e retorna indicadores técnicos, sentimento e previsão de IA.

**Parâmetros**:
- `symbol` (obrigatório): Símbolo da criptomoeda (ex: BTC, ETH, ADA)
- `market` (opcional): Moeda de cotação (padrão: USD)

**Exemplo**:
```bash
curl http://localhost:5000/api/analyze/BTC?market=USD
```

**Resposta (Estrutura Adicional de Sentimento)**:
```json
{
  // ... outros campos
  "market_sentiment": {
    "fear_greed_index": 72,
    "fear_greed_status": "Greed",
    "description": "Índice de Medo e Ganância do mercado (0=Extreme Fear, 100=Extreme Greed)"
  },
  "prediction": {
    "percent": 75,
    "direction": "SUBIR",
    "confidence": "Alta"
  },
  // ... outros campos
}
```

### GET `/api/health`

Verifica o status da API.

**Exemplo**:
```bash
curl http://localhost:5000/api/health
```

## 🔑 Obtenha uma Chave da Alpha Vantage

1. Acesse [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Insira seu email
3. Você receberá uma chave gratuita por email
4. Use a chave no arquivo `.env` ou na variável de ambiente do seu servidor

## 📊 Indicadores Técnicos e Sentimento

### RSI (Índice de Força Relativa)
- **Intervalo**: 0-100
- **Sobrecomprado**: > 70 (possível queda)
- **Sobrevendido**: < 30 (possível subida)

### MACD (Convergência/Divergência de Médias Móveis)
- **Positivo**: Tendência de alta
- **Negativo**: Tendência de baixa

### Fear & Greed Index (Índice de Medo e Ganância)
- **Fonte**: Alternative.me
- **0 (Extreme Fear)**: Possível oportunidade de compra
- **100 (Extreme Greed)**: Possível correção de mercado
- **Uso**: Integrado ao modelo de IA para melhorar a precisão da previsão.

## 🤖 Modelo de Machine Learning

O modelo de previsão utiliza:
- **Algoritmo**: Regressão Logística
- **Features**: RSI, MACD, Variação de Preço, Volatilidade, **Fear & Greed Index**
- **Treinamento**: Dados históricos dos últimos 30+ dias
- **Saída**: Probabilidade de subida/descida (0-100%)

## ⚠️ Limitações

- **Alpha Vantage**: 5 chamadas por minuto, 500 por dia (plano gratuito)
- **Previsões**: Baseadas em análise técnica e sentimento, não são garantias de lucro.

## 🔒 Segurança

- Não armazena dados pessoais
- Não requer login
- Chave da API armazenada apenas no servidor
- CORS habilitado para acesso do frontend

## 📝 Licença

Este projeto é de código aberto e pode ser usado livremente.

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para fazer um fork e enviar pull requests.

## 📧 Suporte

Para dúvidas ou problemas, abra uma issue no repositório.

---

**Desenvolvido com ❤️ para análise de criptomoedas**

