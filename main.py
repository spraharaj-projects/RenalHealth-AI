from cnn_classifier import logger
from cnn_classifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from cnn_classifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from cnn_classifier.pipeline.stage_03_training import TrainingPipeline


STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e


logger.info("****************************************************************")

STAGE_NAME = "Prepare Base Model"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = PrepareBaseModelPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e


logger.info("****************************************************************")

STAGE_NAME = "Training"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = TrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e
