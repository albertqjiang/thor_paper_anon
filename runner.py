import argparse
import gin
import metric_logging

#This imports are to configure gin configurable classes
import jobs
import solvers

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file', action='append', default=[],
        help='Gin config files.'
    )
    parser.add_argument(
        '--config', action='append', default=[],
        help='Gin config overrides.'
    )
    parser.add_argument(
        '--mrunner', action='store_true',
        help='Add mrunner spec to gin-config overrides and Neptune to loggers.'
        '\nNOTE: It assumes that the last config override (--config argument) '
        'is a path to a pickled experiment config created by the mrunner CLI or'
        'a mrunner specification file.'
    )
    return parser.parse_args()

@gin.configurable()
def run(job_class):
    # metric_logging.log_text('host_name', platform.node())
    # metric_logging.log_text('n_gpus', str(torch.cuda.device_count()))

    job = job_class()
    return job.execute()

if __name__ == '__main__':
    args = _parse_args()

    gin_bindings = args.config
    if args.mrunner:
        from mrunner_utils import mrunner_client
        spec_path = gin_bindings.pop()
        specification, overrides = mrunner_client.get_configuration(spec_path)
        gin_bindings.extend(overrides)

        if 'use_neptune' in specification['parameters']:
            if specification['parameters']['use_neptune']:
                try:
                    neptune_logger = mrunner_client.configure_neptune(specification)
                    metric_logging.register_logger(neptune_logger)

                except mrunner_client.NeptuneAPITokenException:
                    print('HINT: To run with Neptune logging please set your '
                          'NEPTUNE_API_TOKEN environment variable')

    gin.parse_config_files_and_bindings(args.config_file, gin_bindings)
    run()