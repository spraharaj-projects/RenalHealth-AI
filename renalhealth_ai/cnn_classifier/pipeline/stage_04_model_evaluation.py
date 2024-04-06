from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.model_evaluation import Evaluation
from cnn_classifier import logger

STAGE_NAME = "Evalution Stage"

class EvaluationPipeline:
  def __init__(self):
    pass

  def main(self):
    config = ConfigurationManager()
    evaluation_config = config.get_evaluation_config()
    evaluation = Evaluation(evaluation_config)
    evaluation.evaluation()
    evaluation.log_into_mlflow()


if __name__ == "__main__":
  try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
  except Exception as e:
    logger.exception(e)
    raise e
          