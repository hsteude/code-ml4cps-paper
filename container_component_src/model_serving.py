import kserve
import click
from serving.kserve import CustomTransformer, CustomPredictor

DEFAULT_MODEL_NAME = "model"


@click.group()
def cli():
    pass


@cli.command('transformer', context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option('--model_name', default=DEFAULT_MODEL_NAME,
              help='The name that the model is served under.', type=str)
@click.option('--predictor_host', help='The URL for the model predict function', required=True, type=str)
def serve_transformer(model_name, predictor_host):
    transformer = CustomTransformer(
        name=model_name,
        predictor_host=predictor_host,
    )
    server = kserve.ModelServer()
    server.start(models=[transformer])


@cli.command('predictor', context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option('--model_name', default=DEFAULT_MODEL_NAME,
              help='The name that the model is served under.', type=str)
def serve_predictor(model_name):
    predictor = CustomPredictor(
        name=model_name,
    )
    server = kserve.ModelServer()
    server.start(models=[predictor])


if __name__ == "__main__":
    cli()
