import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

if __name__ == "__main__":
    # Criação do conversor PDF
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )  

    # Processa o PDF
    rendered = converter("data/LDJ.pdf")

    # Extrai texto e imagens
    text, _, images = text_from_rendered(rendered)

    # Caminho de saída
    pasta_saida = "output_rag"
    os.makedirs(pasta_saida, exist_ok=True)  # cria a pasta se não existir

    caminho_arquivo_md = os.path.join(pasta_saida, "LDJ_output.md")

    # Salva o texto em Markdown
    with open(caminho_arquivo_md, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Arquivo Markdown gerado com sucesso: {caminho_arquivo_md}")

