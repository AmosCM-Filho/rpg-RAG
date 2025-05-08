import os
from PyPDF2 import PdfReader

# Caminho da pasta onde está o PDF
pasta_pdf = 'data'
nome_arquivo = 'LDJ.pdf'
pagina_desejada = 146  # Número da página que você quer (começa do 1)

# Monta o caminho completo
caminho_pdf = os.path.join(pasta_pdf, nome_arquivo)

# Verifica se o arquivo existe
if os.path.exists(caminho_pdf):
    # Abre o PDF
    leitor = PdfReader(caminho_pdf)
    
    # Corrige o índice da página (começa em 0 no Python)
    indice_pagina = pagina_desejada - 1

    if 0 <= indice_pagina < len(leitor.pages):
        pagina = leitor.pages[indice_pagina]
        texto = pagina.extract_text()
        print(f"Conteúdo da página {pagina_desejada}:\n")
        print(texto)
    else:
        print(f"Erro: O PDF tem apenas {len(leitor.pages)} páginas.")
else:
    print("Arquivo PDF não encontrado.")
