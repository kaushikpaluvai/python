import argparse

parser = argparse.ArgumentParser(description='Train a cross sell recommendation model with spark.')
parser.add_argument('-config-json', help='If you pass in a config json, it will be parsed and used to run the pipeline.')
parser.add_argument('-config-path', help='If you pass in a config path, that file will be used instead of the default config')
parser.add_argument('-seed', help='Integer seed that will be used in spark code.')
parser.add_argument('-override-run-id', help='manually specify a run id if desired, useful for testing mongo writes')
parser.add_argument('--synthetic-data', action='store_true', default=False, help='Run through the pipeline on synthetic, generated data (doesnt require data lake access, should only take a couple minutes)')
parser.add_argument('--write-to-cosmos', action='store_true', default=False,
                    help='Write the resulting trained model data out to cosmos.')
parser.add_argument('--sampled_data', action='store_true', default=False, help="Use sampled data for testing")
parser.add_argument('--overwrite-cosmos-collections', action='store_true', default=False)
parser.add_argument('-repo-head', help='Pass repo head, used for MLflow Logging. Required in pipeline')
parser.add_argument('-job-id', help='Pass job-id, used for MLflow Logging and tagging the JOB-ID RUN ID in AML Workspace. Required in pipeline')
parser.add_argument('-func-appname', help='Pass the function app name to be updated with the run id, Required in pipeline')
parser.add_argument('-resource-group', help='Pass the resource group name to which the function app belongs, Required in pipeline')
parser.add_argument('-log-level', default="INFO", help='Pass the log level to control the amount of logs to be written, Required in pipeline')
parser.add_argument('-subscription-id', help='Pass the subscription id to which the Databricks belongs to, Required in pipeline')
args = parser.parse_args()

if __name__ == "__main__":
    #parsed = parser.parse_args(["-config-path", "C:/test", "--synthetic-data"])
    parsed = parser.parse_args(
        ['/Applications/PyCharm.app/Contents/plugins/python/helpers/pydev/pydevconsole.py', '--mode=client',
         '--port=63176'])
    print(parsed)
    print(parsed.config_path)
    print(parsed.synthetic_data)
