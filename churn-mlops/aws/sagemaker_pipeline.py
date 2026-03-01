"""
=============================================================================
AWS SAGEMAKER PIPELINES — Orquestração Completa do MLOps
=============================================================================

SageMaker Pipelines é o equivalente AWS ao GitHub Actions, mas construído
especificamente para workloads de Machine Learning. Cada "step" é um nó
no grafo orientado acíclico (DAG) do pipeline.

ARQUITETURA DO PIPELINE:
  ┌──────────────────┐
  │  GlueETLStep     │  → ETL serverless com Spark (glue_etl_job.py)
  └────────┬─────────┘
           │ dados processados (S3 Parquet)
           ▼
  ┌──────────────────┐
  │  TrainingStep    │  → Training Job (train_sagemaker.py) em ml.m5.xlarge
  └────────┬─────────┘
           │ model.tar.gz (S3)
           ▼
  ┌──────────────────┐
  │  EvaluationStep  │  → Calcula métricas no conjunto de teste
  └────────┬─────────┘
           │ evaluation.json (ROC-AUC)
           ▼
  ┌────────────────────────────────────┐
  │  RegisterStep (condicionado)       │  → Só registra se ROC-AUC ≥ 0.75
  │  (ConditionStep: roc_auc >= 0.75)  │
  └────────────────────────────────────┘

COMO EXECUTAR:
  1. Configure as variáveis no início do script (BUCKET_NAME, ROLE_ARN, etc.)
  2. pip install sagemaker boto3
  3. python aws/sagemaker_pipeline.py

PRÉ-REQUISITOS:
  - Conta AWS com SageMaker e Glue habilitados
  - IAM Role com política AmazonSageMakerFullAccess
  - Bucket S3 criado

REFERÊNCIAS:
  - MLOps: Continuous Delivery for Machine Learning on AWS
  - Build a Secure Enterprise ML Platform on AWS
  - AWS Samples: automated-machine-learning-pipeline-options-on-aws
  - Machine Learning Lens — AWS Well-Architected Framework
  - Comparative Analysis of AWS Model Deployment Services
=============================================================================
"""

import json
import boto3

import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    TrainingStep,
    ProcessingStep,
    TransformStep,
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import (
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.properties import PropertyFile
from sagemaker.sklearn import SKLearn, SKLearnProcessor
from sagemaker.glue import GlueDataQualityResult
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker import get_execution_role, Session

# =============================================================================
# CONFIGURAÇÃO — ajuste para o seu ambiente AWS
# =============================================================================

# Nome do bucket S3 onde os dados e artefatos serão armazenados
BUCKET_NAME = "seu-bucket-mlops-churn"

# Prefixo S3 para organização dos artefatos
S3_PREFIX = "churn-prediction"

# Nome do Glue ETL Job criado no console (ou via CLI)
GLUE_JOB_NAME = "churn-etl-job"

# Nome do Model Package Group no SageMaker Model Registry
MODEL_PACKAGE_GROUP = "ChurnPredictionModels"

# Região AWS
AWS_REGION = "us-east-1"

# Instância para treinamento
TRAINING_INSTANCE = "ml.m5.xlarge"

# Framework scikit-learn compatível com o container do SageMaker
SKLEARN_FRAMEWORK_VERSION = "1.2-1"

# =============================================================================
# INICIALIZAÇÃO DA SESSÃO
# =============================================================================

boto_session = boto3.Session(region_name=AWS_REGION)
sagemaker_session = Session(boto_session=boto_session)

# Em produção, usa a role da instância ou do Studio
# Em desenvolvimento local, usa o profile AWS configurado
try:
    role = get_execution_role()
except Exception:
    # Desenvolvimento local — configure a role manualmente
    role = f"arn:aws:iam::SEU_ACCOUNT_ID:role/SageMakerExecutionRole"
    print(f"[WARN] Usando role local: {role}")

s3_base_uri = f"s3://{BUCKET_NAME}/{S3_PREFIX}"

# =============================================================================
# PARÂMETROS DO PIPELINE
# Permitem re-executar o pipeline com diferentes configurações sem alterar código
# =============================================================================

param_input_s3 = ParameterString(
    name="InputS3Path",
    default_value=f"{s3_base_uri}/raw/customers/",
)

param_output_s3 = ParameterString(
    name="ProcessedS3Path",
    default_value=f"{s3_base_uri}/processed/",
)

param_n_estimators = ParameterInteger(
    name="NEstimators",
    default_value=200,
)

param_learning_rate = ParameterFloat(
    name="LearningRate",
    default_value=0.05,
)

param_max_depth = ParameterInteger(
    name="MaxDepth",
    default_value=4,
)

# Threshold mínimo de ROC-AUC para registrar o modelo
param_roc_auc_threshold = ParameterFloat(
    name="RocAucThreshold",
    default_value=0.75,
)


# =============================================================================
# STEP 1: GLUE ETL — Processamento de Dados
#
# Executa o glue_etl_job.py via GlueStep ou ProcessingStep com container Glue.
# Aqui usamos ProcessingStep com SKLearnProcessor como alternativa portátil
# (não requer configuração prévia do Glue no ambiente de teste).
#
# Em produção com grandes volumes (TB+), substitua por GlueStep nativo.
# =============================================================================

def create_glue_etl_step(input_s3: str, output_s3: str) -> ProcessingStep:
    """
    Cria o step de ETL para o pipeline.

    Em ambientes com Glue configurado (recomendado para TB de dados):
      from sagemaker.workflow.steps import GlueStep
      glue_step = GlueStep(
          name="GlueETLStep",
          job_name=GLUE_JOB_NAME,
          arguments={"--INPUT_S3_PATH": input_s3, "--OUTPUT_S3_PATH": output_s3},
      )

    Aqui usamos SKLearnProcessor como alternativa para demonstração:
    """
    processor = SKLearnProcessor(
        framework_version=SKLEARN_FRAMEWORK_VERSION,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name="churn-etl",
        role=role,
        sagemaker_session=sagemaker_session,
    )

    etl_step = ProcessingStep(
        name="ChurnETLStep",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=input_s3,
                destination="/opt/ml/processing/input",
                input_name="raw-data",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"{output_s3}train/",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=f"{output_s3}test/",
            ),
        ],
        code="aws/glue_etl_job.py",
        job_arguments=["--train-ratio", "0.8"],
    )

    return etl_step


# =============================================================================
# STEP 2: TRAINING JOB — Treinamento do Modelo
# =============================================================================

def create_training_step(etl_step: ProcessingStep) -> TrainingStep:
    """
    Cria o SageMaker Training Job que executa aws/train_sagemaker.py.

    O SKLearn estimator utiliza o container gerenciado da AWS com
    Python 3.9 + scikit-learn 1.2 pré-instalados.

    O Training Job:
      1. Provisiona ml.m5.xlarge automaticamente
      2. Copia dados de S3 → /opt/ml/input/data/{channel}/
      3. Executa train_sagemaker.py
      4. Copia /opt/ml/model/ → S3 como model.tar.gz
      5. Desprovisiona a instância
    """
    estimator = SKLearn(
        entry_point="aws/train_sagemaker.py",
        framework_version=SKLEARN_FRAMEWORK_VERSION,
        instance_type=TRAINING_INSTANCE,
        instance_count=1,
        role=role,
        sagemaker_session=sagemaker_session,
        base_job_name="churn-training",
        # Hiperparâmetros como parâmetros do pipeline (tunáveis)
        hyperparameters={
            "n-estimators":   param_n_estimators,
            "learning-rate":  param_learning_rate,
            "max-depth":      param_max_depth,
            "subsample":      0.8,
            "random-state":   42,
        },
        # Métricas capturadas do stdout do container via Regex
        # Alimentam o CloudWatch e o SageMaker Experiments
        metric_definitions=[
            {"Name": "validation:roc_auc",       "Regex": r"ROC-AUC \(Teste\): ([0-9\.]+)"},
            {"Name": "validation:avg_precision",  "Regex": r"Avg-Precision: ([0-9\.]+)"},
            {"Name": "train:cv_roc_auc_mean",     "Regex": r"ROC-AUC \(CV\) — Média: ([0-9\.]+)"},
        ],
        # Configurações de segurança (AWS Well-Architected ML Lens)
        enable_sagemaker_metrics=True,
        output_path=f"{s3_base_uri}/model-artifacts/",
        code_location=f"{s3_base_uri}/code/",
    )

    training_step = TrainingStep(
        name="ChurnTrainingStep",
        estimator=estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=etl_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="application/x-parquet",
            ),
            "test": sagemaker.inputs.TrainingInput(
                s3_data=etl_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                content_type="application/x-parquet",
            ),
        },
        # Rastreamento de experimentos — cada execução fica registrada
        # no SageMaker Experiments para comparação de modelos
        cache_config=sagemaker.workflow.steps.CacheConfig(
            enable_caching=True,   # Reutiliza resultados se inputs não mudaram
            expire_after="30d",
        ),
    )

    return training_step


# =============================================================================
# STEP 3: EVALUATION STEP — Avaliação formal para o Model Registry
# =============================================================================

def create_evaluation_step(training_step: TrainingStep, etl_step: ProcessingStep):
    """
    Avalia o modelo treinado e gera o evaluation.json.

    O SageMaker Model Registry usa este JSON para registrar as métricas
    junto ao Model Package, permitindo auditoria e comparação de versões.
    """
    evaluate_processor = SKLearnProcessor(
        framework_version=SKLEARN_FRAMEWORK_VERSION,
        instance_type="ml.t3.medium",   # Instância leve — só avalia
        instance_count=1,
        base_job_name="churn-evaluation",
        role=role,
        sagemaker_session=sagemaker_session,
    )

    # O evaluation.json é capturado como PropertyFile para uso condicional
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    evaluate_step = ProcessingStep(
        name="ChurnEvaluationStep",
        processor=evaluate_processor,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
                input_name="model",
            ),
            ProcessingInput(
                source=etl_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
                input_name="test-data",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"{s3_base_uri}/evaluation/",
            )
        ],
        code="aws/evaluate.py",   # Script de avaliação (inline abaixo)
        property_files=[evaluation_report],
    )

    return evaluate_step, evaluation_report


# =============================================================================
# STEP 4: REGISTER MODEL — Registro no Model Registry (condicional)
#
# Só registra se ROC-AUC >= threshold.
# Em produção, adicione comparação com o modelo champion atual.
# =============================================================================

def create_register_step(
    training_step: TrainingStep,
    evaluate_step: ProcessingStep,
    evaluation_report: PropertyFile,
    roc_auc_threshold,
):
    """
    Registra o modelo no SageMaker Model Registry se as métricas
    ultrapassarem o threshold definido como parâmetro do pipeline.

    O Model Registry:
      - Versiona os modelos aprovados com metadados completos
      - Controla o status de aprovação (PendingManualApproval → Approved)
      - Alimenta o SageMaker Endpoint com a versão aprovada
      - Integra com o pipeline de CD para deploy automático
    """
    # Métricas registradas junto ao Model Package para auditoria
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"{evaluate_step.arguments['ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri']}/evaluation.json",
            content_type="application/json",
        ),
    )

    # Baselines para detecção de drift em produção (DataQuality, ModelQuality)
    drift_check_baselines = DriftCheckBaselines(
        model_statistics=MetricsSource(
            s3_uri=f"{evaluate_step.arguments['ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri']}/evaluation.json",
            content_type="application/json",
        ),
    )

    register_step = RegisterModel(
        name="ChurnRegisterModel",
        estimator=training_step.estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t3.medium", "ml.m5.large"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=MODEL_PACKAGE_GROUP,
        # Aprovação manual antes do deploy em produção (governance)
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
        drift_check_baselines=drift_check_baselines,
        description=f"Churn Prediction — GBM com Pipeline sklearn",
    )

    # Condição: só registra se ROC-AUC >= threshold
    condition_roc_auc = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluate_step.name,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.roc_auc.value",
        ),
        right=roc_auc_threshold,
    )

    condition_step = ConditionStep(
        name="CheckRocAucThreshold",
        conditions=[condition_roc_auc],
        if_steps=[register_step],    # Aprovado: registra no Model Registry
        else_steps=[],               # Reprovado: pipeline termina sem registrar
    )

    return condition_step


# =============================================================================
# CONSTRUÇÃO DO PIPELINE COMPLETO
# =============================================================================

def build_pipeline() -> Pipeline:
    """
    Monta e retorna o objeto Pipeline com todos os steps definidos.

    O Pipeline é um DAG (Directed Acyclic Graph):
      ETL ──▶ Training ──▶ Evaluation ──▶ ConditionStep ──▶ (Register | Stop)

    Após build_pipeline().upsert(), o pipeline fica disponível no console
    do SageMaker e pode ser executado manualmente ou via gatilho (EventBridge,
    GitHub Actions, etc.).
    """
    print("=" * 60)
    print("Construindo SageMaker Pipeline: churn-prediction-pipeline")
    print("=" * 60)

    # Step 1: ETL
    etl_step = create_glue_etl_step(
        input_s3=param_input_s3,
        output_s3=param_output_s3,
    )
    print("✅ Step 1 criado: ChurnETLStep")

    # Step 2: Training
    training_step = create_training_step(etl_step)
    print("✅ Step 2 criado: ChurnTrainingStep")

    # Step 3: Evaluation
    evaluate_step, evaluation_report = create_evaluation_step(training_step, etl_step)
    print("✅ Step 3 criado: ChurnEvaluationStep")

    # Step 4: Conditional register
    condition_step = create_register_step(
        training_step=training_step,
        evaluate_step=evaluate_step,
        evaluation_report=evaluation_report,
        roc_auc_threshold=param_roc_auc_threshold,
    )
    print("✅ Step 4 criado: CheckRocAucThreshold + ChurnRegisterModel")

    # Monta o pipeline com todos os parâmetros e steps
    pipeline = Pipeline(
        name="churn-prediction-pipeline",
        parameters=[
            param_input_s3,
            param_output_s3,
            param_n_estimators,
            param_learning_rate,
            param_max_depth,
            param_roc_auc_threshold,
        ],
        steps=[
            etl_step,
            training_step,
            evaluate_step,
            condition_step,
        ],
        sagemaker_session=sagemaker_session,
    )

    return pipeline


# =============================================================================
# FUNÇÕES DE OPERAÇÃO DO PIPELINE
# =============================================================================

def deploy_endpoint(model_package_arn: str, endpoint_name: str = "churn-realtime-endpoint"):
    """
    Deploya o modelo aprovado no Model Registry como endpoint de tempo real.

    O SageMaker gerencia o AutoScaling, o load balancing e o health check
    automaticamente. Você paga apenas pela instância ativa.

    Em produção, esta etapa seria automatizada por um EventBridge rule:
      Model Registry aprovado ──▶ EventBridge ──▶ Lambda ──▶ deploy_endpoint()

    Args:
        model_package_arn: ARN do Model Package aprovado no Registry.
        endpoint_name:     Nome do endpoint SageMaker.
    """
    from sagemaker import ModelPackage

    model = ModelPackage(
        role=role,
        model_package_arn=model_package_arn,
        sagemaker_session=sagemaker_session,
    )

    model.deploy(
        initial_instance_count=1,
        instance_type="ml.t3.medium",
        endpoint_name=endpoint_name,
        # Data capture para monitoramento de drift em produção
        data_capture_config=sagemaker.model_monitor.DataCaptureConfig(
            enable_capture=True,
            sampling_percentage=20,       # Captura 20% das requisições
            destination_s3_uri=f"{s3_base_uri}/data-capture/",
        ),
    )

    print(f"✅ Endpoint deployado: {endpoint_name}")
    print(f"   Invoque com: aws sagemaker-runtime invoke-endpoint --endpoint-name {endpoint_name} ...")
    return endpoint_name


def invoke_endpoint_example(endpoint_name: str):
    """
    Exemplo de invocação do endpoint em tempo real.

    Em produção, substitua pelo cliente HTTP da sua aplicação
    (ou utilize a API FastAPI do app/app.py como camada intermediária).
    """
    runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

    payload = json.dumps({
        "instances": [[
            3,      # tenure_months
            450.0,  # monthly_charges
            1350.0, # total_charges
            1,      # num_products
            12,     # support_calls
            25,     # payment_delay_days
            22,     # age
            1.5,    # satisfaction_score
            450.0,  # avg_revenue_per_product
            4.0,    # support_call_rate
            1,      # is_new_customer
        ]]
    })

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload,
    )

    result = json.loads(response["Body"].read().decode())
    print(f"Resposta do endpoint: {json.dumps(result, indent=2)}")
    return result


# =============================================================================
# MAIN — Cria/atualiza e executa o pipeline
# =============================================================================

if __name__ == "__main__":
    # 1. Constrói o pipeline
    pipeline = build_pipeline()

    # 2. Cria ou atualiza o pipeline no SageMaker
    #    upsert() é idempotente: cria se não existe, atualiza se existe
    print("\nEnviando pipeline para o SageMaker...")
    pipeline.upsert(role_arn=role)
    print("✅ Pipeline registrado no SageMaker!")
    print(f"   Visualize em: https://console.aws.amazon.com/sagemaker/home?region={AWS_REGION}#/studio")

    # 3. Executa o pipeline com os parâmetros padrão
    #    Para testar com parâmetros diferentes, passe um dict:
    execution = pipeline.start(
        parameters={
            "InputS3Path":     f"s3://{BUCKET_NAME}/{S3_PREFIX}/raw/customers/",
            "ProcessedS3Path": f"s3://{BUCKET_NAME}/{S3_PREFIX}/processed/",
            "NEstimators":     200,
            "LearningRate":    0.05,
            "RocAucThreshold": 0.75,
        }
    )

    print(f"\n🚀 Pipeline executando!")
    print(f"   ARN: {execution.arn}")
    print(f"   Acompanhe no console ou com:")
    print(f"   execution.wait()  # bloqueia até finalizar")
    print(f"   execution.list_steps()  # mostra o status de cada step")

    # Para aguardar a conclusão (em produção, use EventBridge + notificações):
    # execution.wait()
    # execution.list_steps()
