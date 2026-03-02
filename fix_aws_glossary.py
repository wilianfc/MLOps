import re

STYLE = (
    "text;html=1;strokeColor=#3b82f6;fillColor=#eff6ff;"
    "align=left;verticalAlign=top;whiteSpace=wrap;"
    "rounded=1;arcSize=5;fontSize=11;fontFamily=Arial;shadow=1;"
)

# Extrair value diretamente via posição no arquivo
src = open("churn-mlops/docs/clustering_rag_guide.drawio", encoding="utf-8").read()

# encontrar a tag pelo id
start_tag = 'id="glossary_acronyms"'
idx = src.find(start_tag)
if idx == -1:
    print("ERRO: id não encontrado")
    exit(1)

# achar value=" após o id
val_start = src.find('value="', idx) + len('value="')
# encontrar o fechamento da tag mxCell (">")
# O value termina no " que precede o vertex="1"
# Buscar '" vertex=' após val_start
val_end = src.find('" vertex="1">', val_start)
if val_end == -1:
    val_end = src.find('" vertex="1" >', val_start)
gval = src[val_start:val_end]
print(f"Value extraído com {len(gval)} chars OK")

content = open("churn-mlops/docs/aws_deploy_pipeline.drawio", encoding="utf-8").read()

if "glossary_acronyms" in content:
    print("Já presente, pulando.")
else:
    node = (
        f'        <mxCell id="glossary_acronyms" parent="1" style="{STYLE}" '
        f'value="{gval}" vertex="1">\n'
        f'          <mxGeometry height="390" width="2600" x="40" y="1330" as="geometry" />\n'
        f'        </mxCell>\n'
    )
    content = content.replace("\n    </root>", "\n" + node + "    </root>")
    open("churn-mlops/docs/aws_deploy_pipeline.drawio", "w", encoding="utf-8").write(content)
    print("OK: aws_deploy_pipeline.drawio")

import xml.etree.ElementTree as ET
for f in [
    "churn-mlops/docs/clustering_rag_guide.drawio",
    "churn-mlops/docs/crisp_dm_bpmn_pipeline.drawio",
    "churn-mlops/docs/aws_deploy_pipeline.drawio",
]:
    try:
        ET.parse(f)
        has = "glossary_acronyms" in open(f, encoding="utf-8").read()
        print(f"XML OK | glossario={'sim' if has else 'NAO'}: {f.split('/')[-1]}")
    except Exception as e:
        print(f"ERRO XML: {f.split('/')[-1]} -> {e}")
