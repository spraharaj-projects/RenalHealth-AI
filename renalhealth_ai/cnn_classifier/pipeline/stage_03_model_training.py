from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.model_training import Training
from cnn_classifier import logger

STAGE_NAME = "Training Stage"

class TrainingPipeline:
  def __init__(self):
    pass

  def main(self):
    config = ConfigurationManager()
    training_config = config.get_training_config()
    training = Training(config=training_config)
    training.get_base_model()
    training.train_valid_generator()
    training.compile()
    training.process()


if __name__ == "__main__":
  try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = TrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
  except Exception as e:
    logger.exception(e)
    raise e
          