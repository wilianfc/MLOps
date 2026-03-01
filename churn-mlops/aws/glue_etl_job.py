"""
=============================================================================
AWS GLUE — ETL JOB: Preparação de Dados para Churn Prediction
=============================================================================

O AWS Glue é um serviço serverless de ETL (Extract, Transform, Load) baseado
em Apache Spark. Neste pipeline de MLOps, ele substitui a geração sintética
de dados do train.py, conectando-se a fontes reais (S3, RDS, Redshift) e
entregando dados limpos e particionados para o SageMaker treinar.

POSIÇÃO NO PIPELINE MLOps:
  ┌─────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────────┐
  │  S3 Raw │───▶│ Glue ETL │───▶│ S3 Processed │───▶│  SageMaker   │
  │  (CSV)  │    │ (este    │    │  (Parquet +  │    │  Training    │
  │         │    │ script)  │    │  particionado│    │  Job         │
  └─────────┘    └──────────┘    └──────────────┘    └──────────────┘

COMO FAZER O DEPLOY DESTE JOB:
  1. Acesse o console AWS → Glue → Jobs → "Create job"
  2. Tipo: "Spark script editor"
  3. Cole este script
  4. Configure os parâmetros de job abaixo (--INPUT_S3_PATH, etc.)
  5. IAM Role: precisa de permissões s3:GetObject e s3:PutObject

  Via AWS CLI:
    aws glue create-job \
      --name "churn-etl-job" \
      --role "arn:aws:iam::ACCOUNT_ID:role/GlueServiceRole" \
      --command '{"Name":"glueetl","ScriptLocation":"s3://seu-bucket/scripts/glue_etl_job.py","PythonVersion":"3"}' \
      --glue-version "4.0" \
      --default-arguments '{
        "--INPUT_S3_PATH": "s3://seu-bucket/raw/customers/",
        "--OUTPUT_S3_PATH": "s3://seu-bucket/processed/churn/",
        "--enable-metrics": "true",
        "--enable-continuous-cloudwatch-log": "true"
      }'

REFERÊNCIAS:
  - AWS Samples: automated-machine-learning-pipeline-options-on-aws
  - Red Hat Feature Store: camada de dados unificada para pipelines de ML
  - AWS Well-Architected ML Lens: design reprodutível no ciclo de vida de ML
=============================================================================
"""

import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Imports do Glue e Spark — disponíveis no runtime do AWS Glue
# pyspark e awsglue são injetados automaticamente pelo ambiente Glue
# ---------------------------------------------------------------------------
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame

from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, FloatType, StringType, TimestampType,
)
from pyspark.sql.window import Window

# ---------------------------------------------------------------------------
# 1. Inicialização do contexto Glue/Spark
# ---------------------------------------------------------------------------
args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "INPUT_S3_PATH",      # ex: s3://seu-bucket/raw/customers/
        "OUTPUT_S3_PATH",     # ex: s3://seu-bucket/processed/churn/
        "TRAIN_RATIO",        # ex: 0.8 (proporção para treino)
    ],
)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# Logger do Glue (aparece no CloudWatch Logs automaticamente)
logger = glueContext.get_logger()
logger.info(f"[CHURN ETL] Job iniciado: {args['JOB_NAME']}")
logger.info(f"[CHURN ETL] Input:  {args['INPUT_S3_PATH']}")
logger.info(f"[CHURN ETL] Output: {args['OUTPUT_S3_PATH']}")

TRAIN_RATIO = float(args.get("TRAIN_RATIO", "0.8"))

# ---------------------------------------------------------------------------
# 2. Schema explícito — evita inferência incorreta de tipos
#    Em produção, o schema vem do Glue Data Catalog (mais robusto)
# ---------------------------------------------------------------------------
CUSTOMER_SCHEMA = StructType([
    StructField("customer_id",        StringType(),  nullable=False),
    StructField("tenure_months",      IntegerType(), nullable=True),
    StructField("monthly_charges",    FloatType(),   nullable=True),
    StructField("total_charges",      FloatType(),   nullable=True),
    StructField("num_products",       IntegerType(), nullable=True),
    StructField("support_calls",      IntegerType(), nullable=True),
    StructField("payment_delay_days", IntegerType(), nullable=True),
    StructField("age",                IntegerType(), nullable=True),
    StructField("satisfaction_score", FloatType(),   nullable=True),
    StructField("churn",              IntegerType(), nullable=False),
    StructField("event_date",         TimestampType(), nullable=True),
])

# ---------------------------------------------------------------------------
# 3. Extração — lê os dados brutos do S3 via Glue DynamicFrame
#    DynamicFrame é a abstração do Glue sobre o Spark DataFrame.
#    Suporta esquemas inconsistentes entre registros (dados sujos do mundo real).
# ---------------------------------------------------------------------------
logger.info("[CHURN ETL] Lendo dados brutos do S3...")

raw_dynamic_frame = glueContext.create_dynamic_frame.from_options(
    format_options={
        "withHeader": True,
        "separator": ",",
        "quoteChar": '"',
    },
    connection_type="s3",
    format="csv",
    connection_options={
        "paths": [args["INPUT_S3_PATH"]],
        "recurse": True,  # Lê subdiretórios (dados particionados por data)
    },
    transformation_ctx="raw_dynamic_frame",
)

logger.info(f"[CHURN ETL] Registros brutos lidos: {raw_dynamic_frame.count()}")

# Converte para Spark DataFrame para transformações mais expressivas
df_raw = raw_dynamic_frame.toDF()

# ---------------------------------------------------------------------------
# 4. Transformação — limpeza e feature engineering
# ---------------------------------------------------------------------------

def transform_customer_data(df):
    """
    Pipeline de transformação dos dados de clientes.

    Etapas:
      1. Remover duplicatas por customer_id (último registro wins)
      2. Tratar valores nulos com estratégias por feature
      3. Validar ranges das features (data quality gate)
      4. Engenharia de features derivadas
      5. Adicionar metadados de rastreabilidade

    Args:
        df: Spark DataFrame com dados brutos.

    Returns:
        Spark DataFrame limpo e enriquecido.
    """
    initial_count = df.count()
    logger.info(f"[CHURN ETL] Iniciando transformações. Registros: {initial_count}")

    # ── 4.1 Remove duplicatas ─────────────────────────────────────────────
    # Mantém o registro mais recente por customer_id
    window_latest = Window.partitionBy("customer_id").orderBy(F.col("event_date").desc())
    df = (
        df
        .withColumn("_row_num", F.row_number().over(window_latest))
        .filter(F.col("_row_num") == 1)
        .drop("_row_num")
    )
    dedup_count = df.count()
    logger.info(f"[CHURN ETL] Após deduplicação: {dedup_count} (removidos: {initial_count - dedup_count})")

    # ── 4.2 Tratamento de nulos ────────────────────────────────────────────
    # Estratégia: mediana para numéricas, valor neutro para categóricas
    # ATENÇÃO: em produção, calcule a mediana SOMENTE no conjunto de treino
    # e aplique no de teste (equivalente ao StandardScaler no scikit-learn)
    fill_values = {
        "tenure_months":      24,    # Mediana estimada do negócio
        "monthly_charges":    150.0,
        "total_charges":      3600.0,
        "num_products":       2,
        "support_calls":      2,
        "payment_delay_days": 0,
        "age":                40,
        "satisfaction_score": 5.0,
    }
    df = df.na.fill(fill_values)

    # Remove registros sem rótulo (churn) — não podem ser usados no treino
    df = df.filter(F.col("churn").isNotNull())
    logger.info(f"[CHURN ETL] Após remoção de rótulos nulos: {df.count()}")

    # ── 4.3 Validação de ranges (Data Quality Gate) ────────────────────────
    # Registros fora dos ranges esperados são sinalizados, não removidos
    # (permite auditoria posterior no S3)
    df = df.withColumn(
        "data_quality_flag",
        F.when(
            (F.col("tenure_months") < 1) | (F.col("tenure_months") > 72) |
            (F.col("monthly_charges") < 50) | (F.col("monthly_charges") > 500) |
            (F.col("satisfaction_score") < 0) | (F.col("satisfaction_score") > 10) |
            (F.col("age") < 18) | (F.col("age") > 80),
            F.lit("INVALID_RANGE")
        ).otherwise(F.lit("OK"))
    )

    invalid_count = df.filter(F.col("data_quality_flag") == "INVALID_RANGE").count()
    if invalid_count > 0:
        logger.warn(f"[CHURN ETL] Registros com range inválido: {invalid_count}")

    # ── 4.4 Feature Engineering ────────────────────────────────────────────
    # Cria features derivadas que podem aumentar o poder preditivo
    df = df.withColumn(
        # Receita média mensal por produto (normaliza pelo portfólio)
        "avg_revenue_per_product",
        F.round(F.col("monthly_charges") / F.col("num_products"), 2)
    ).withColumn(
        # Taxa de chamados de suporte por mês de contrato
        "support_call_rate",
        F.round(F.col("support_calls") / F.greatest(F.col("tenure_months"), F.lit(1)), 4)
    ).withColumn(
        # Flag de cliente novo (< 6 meses) — comportamento diferente
        "is_new_customer",
        F.when(F.col("tenure_months") < 6, F.lit(1)).otherwise(F.lit(0))
    )

    # ── 4.5 Metadados de rastreabilidade (MLOps) ──────────────────────────
    df = df.withColumn(
        "etl_processed_at",
        F.lit(datetime.utcnow().isoformat() + "Z")
    ).withColumn(
        "etl_job_name",
        F.lit(args["JOB_NAME"])
    )

    logger.info(f"[CHURN ETL] Transformações concluídas. Features geradas: {len(df.columns)}")
    return df


df_transformed = transform_customer_data(df_raw)

# ---------------------------------------------------------------------------
# 5. Divisão Treino/Teste — ESTRATIFICADA por churn
#    Mantém a mesma proporção de classes nos dois splits
# ---------------------------------------------------------------------------
logger.info(f"[CHURN ETL] Dividindo dados (treino={TRAIN_RATIO}, teste={1-TRAIN_RATIO})...")

# Separa por classe e aplica o split em cada uma
df_churn_1 = df_transformed.filter(F.col("churn") == 1)
df_churn_0 = df_transformed.filter(F.col("churn") == 0)

train_1, test_1 = df_churn_1.randomSplit([TRAIN_RATIO, 1 - TRAIN_RATIO], seed=42)
train_0, test_0 = df_churn_0.randomSplit([TRAIN_RATIO, 1 - TRAIN_RATIO], seed=42)

df_train = train_1.union(train_0)
df_test  = test_1.union(test_0)

logger.info(f"[CHURN ETL] Treino: {df_train.count()} | Teste: {df_test.count()}")
logger.info(
    f"[CHURN ETL] Taxa de churn — Treino: {df_train.filter(F.col('churn')==1).count() / df_train.count() * 100:.1f}%"
)

# ---------------------------------------------------------------------------
# 6. Carga — Salva no S3 em formato Parquet, particionado por data
#    Parquet é mais eficiente que CSV: compressão, tipagem, leitura colunar
# ---------------------------------------------------------------------------

def write_to_s3(df, output_path: str, label: str):
    """
    Escreve o DataFrame no S3 como Parquet, particionado por data de processamento.

    A partição por data permite que o SageMaker leia apenas os dados
    relevantes para um treinamento específico, sem reprocessar tudo.

    Args:
        df:          Spark DataFrame a ser salvo.
        output_path: Path S3 de destino.
        label:       'train' ou 'test'.
    """
    full_path = f"{output_path}/{label}/"
    logger.info(f"[CHURN ETL] Salvando {label} em {full_path}...")

    # Converte de volta para DynamicFrame para usar o sink do Glue
    dynamic_frame = DynamicFrame.fromDF(df, glueContext, label)

    glueContext.write_dynamic_frame.from_options(
        frame=dynamic_frame,
        connection_type="s3",
        format="glueparquet",
        connection_options={
            "path": full_path,
            "partitionKeys": [],  # Em produção: ["year", "month"] para time-series
        },
        format_options={
            "compression": "snappy",  # Snappy: bom equilíbrio velocidade/tamanho
        },
        transformation_ctx=f"write_{label}",
    )
    logger.info(f"[CHURN ETL] {label} salvo com sucesso: {full_path}")


write_to_s3(df_train, args["OUTPUT_S3_PATH"], "train")
write_to_s3(df_test,  args["OUTPUT_S3_PATH"], "test")

# ---------------------------------------------------------------------------
# 7. Métricas de qualidade — publicadas no CloudWatch
# ---------------------------------------------------------------------------
total_processed = df_transformed.count()
churn_rate = df_transformed.filter(F.col("churn") == 1).count() / total_processed * 100
invalid_rate = df_transformed.filter(F.col("data_quality_flag") == "INVALID_RANGE").count() / total_processed * 100

logger.info(f"[CHURN ETL] ── RESUMO DO JOB ──────────────────────────")
logger.info(f"[CHURN ETL] Total processado: {total_processed}")
logger.info(f"[CHURN ETL] Taxa de churn:    {churn_rate:.1f}%")
logger.info(f"[CHURN ETL] Dados inválidos:  {invalid_rate:.1f}%")
logger.info(f"[CHURN ETL] Split treino:     {df_train.count()}")
logger.info(f"[CHURN ETL] Split teste:      {df_test.count()}")

job.commit()
logger.info("[CHURN ETL] Job finalizado com sucesso!")
