# 1. Use uma imagem base oficial do Python
FROM python:3.10-slim

# 2. Defina o diretório de trabalho dentro do container
WORKDIR /app

# 3. Copie o arquivo de requisitos para o diretório de trabalho
COPY requirements.txt ./

# 4. Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copie todos os arquivos do projeto para o diretório de trabalho
COPY . .

# 6. Exponha a porta que o Streamlit usa
EXPOSE 8501

# 7. Defina o comando para rodar a aplicação quando o container iniciar
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]