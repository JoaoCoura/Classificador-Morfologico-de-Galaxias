# Classificador Morfologico de Galáxias
Este projeto é uma aplicação Streamlit que realiza a classificação morfológica de galáxias utilizando duas arquiteturas distintas:

ResNet-50 Fine-Tuned

Modelo Autoral (CNN criada do zero)

Após gerar as previsões, o sistema exibe os resultados separadamente e também realiza um ensemble por média, combinando as distribuições de probabilidade para produzir uma predição mais robusta.

## Como Executar
1. Instale as dependências
```bash
pip install streamlit tensorflow pillow numpy opencv-python
```

2. Coloque os modelos .keras na mesma pasta do script
```bash
arquiteturaResNet50.keras
arquiteturaPropria.keras
```

3. Execute o aplicativo
```bash
streamlit run galaxy_classifier_app.py
```
