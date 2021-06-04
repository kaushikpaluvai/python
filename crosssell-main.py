import json
import logging
import os
import random
from datetime import datetime

from DiplaCoreSDK.Logging.azure_sdk_helpers import get_azure_ml_workspace, complete_current_run
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from DiplaCoreSDK.DataLoading.data_loading import DataLoading, sample_master
from DiplaCoreSDK.Configurations.spark import initialize_spark
from DiplaCoreSDK.Configurations.environment_setup import environment_setup
from DiplaCoreSDK.Logging.configure_logging import configure_logging
from DiplaCoreSDK.Transformers.product_availability_transformer import ProductAvailabilityTransformer

from helpers.command_line_helpers import args
from helpers.config import load_config
from helpers.mongo import Mongo

from model.als_model import ALSModel
from model.customer_backfill import CustomerBackfill
from model.model_tuning import GridSearch
from model.rule_mining import RuleMining
from model.rule_relevance import RuleRelevance

# Import command line handler and the resulting args
from model.tree_builder import build_tree

'''
SET UP ENVIRONMENT VARIABLES AND CONNECTION OBJECTS
'''

environment_setup(path="local.settings.json")

if args.repo_head:
    os.environ["REPO_HEAD_HEXSHA"] = args.repo_head

# Load config either from an argument string, or from the default config on disk
if args.config_json:
    logging.info(f"Using provided config json string: {args.config_json}")
    config = json.loads(args.config_json)
else:
    config = load_config(config_path=args.config_path)
print(f"Loaded config: {config}")
# Try to set up mlflow logging, or fall back on local logging
run_id = configure_logging(mlflow=True, azure_logging=False, config=config, args=args, log_level=args.log_level)
logging.info(f"Using run id: {run_id}")

spark = initialize_spark()
spark.sparkContext.setCheckpointDir('/mnt/adl-fiona-cdm-rbs/HyperPersonalization/XSellModelCheckpoints/')

# Seed all pipeline components to the current datetime for reproducibility
if args.seed:
    seed = args.seed
else:
    seed = int(datetime.now().timestamp())
random.seed(seed)
logging.info(f"Random seed set as {seed}", extra={"metrics": {"seed": seed}})

''' 
FORMAT MASTER DATA FRAME TO BE USED FOR ALL MODEL TRAINING
'''
backfill_category = config["backfill_category"]
product_identifier = config["product_identifier"]
mining_category = config["mining_category"]

logging.info('BUILDING DATA')
dataLoading = DataLoading(**config, channels=["CnC", "Store"], banner_ids=[1000, 1001, 1002],
                          downsample_master_data=True,
                          testing_synthetic_data=args.synthetic_data,
                          testing_sample_data=args.sampled_data,
                          config_object = config)
dataLoading.run()

master: DataFrame = dataLoading.master
master = master.cache()

logging.info(f'MASTER HAS : {dataLoading.metrics["n_unique_order_ids_total"]} UNIQUE ORDER IDS')
cat_ranking: DataFrame = dataLoading.category_backfill
cat_ranking.write.mode("overwrite")\
    .parquet('/mnt/adl-fiona-cdm-rbs/HyperPersonalization/XSellModelCheckpoints/cat_ranking.parquet')

#all_customers: DataFrame = dataLoading.customers
#all_customers.write.mode('overwrite') \
#    .parquet('/mnt/adl-fiona-cdm-rbs/HyperPersonalization/XSellModelCheckpoints/all_customers.parquet')

logging.info(f"Metrics extracted from dataload job: { dataLoading.metrics }", extra={"metrics": dataLoading.metrics})

'''
MINE ASSOCIATION RULES
'''

additional_metrics = {'model_run_config': config}

grid: dict = {'min_prod_prevalence': config["min_prod_prevalence"],
              'min_prod_confidence': config["min_prod_confidence"],
              'min_prod_lift': config["min_prod_lift"],
              'min_cat_prevalence': config["min_cat_prevalence"],
              'min_cat_confidence': config["min_cat_confidence"],
              'min_cat_lift': config["min_cat_lift"],
              'n_rhs_per_category': config['n_rhs_per_category'],
              'category_lift_discount_coef': config['category_lift_discount_coef']}

# TODO remove n_prods/n_trans, incorporate pickling as a logging handler
init_params: dict = {'master': master,
                     'product_identifier': product_identifier,
                     'category_identifier': mining_category,
                     'product_mapping': dataLoading.category_mining,
                     'n_prod': dataLoading.metrics["n_unique_products_total"],
                     'n_trans': dataLoading.metrics["n_unique_order_ids_total"],
                     'downsample_rules_max': 100 if args.synthetic_data else None,
                     'recommendation_window_mining': config["recommendation_window_mining"]}

logging.info(f'starting rule mining at: {datetime.now()}')
with RuleMining(**init_params) as miner:
    miner_gs = GridSearch(model=miner,
                          optimization_metric='lhs_item_coverage_all_rules',
                          maximize=True,
                          grid=grid,
                          timeout_seconds=100000,
                          pickle_results=False,
                          additional_metrics=additional_metrics)
    miner_gs.fit()

mining_metrics = miner_gs.best_model.metrics
logging.info(f"Metrics extracted from rule mining job : { mining_metrics }", extra={"metrics": mining_metrics})
# logging.info("Plots extracted from rule mining job", extra={"plots": miner_gs.best_model.get_plots()})

'''
CALCULATE RULE RELEVANCE
'''
combined_rules: DataFrame = miner_gs.best_model.combined_rules
combined_rules = combined_rules.cache()

# Reformatting, checkpointing, and caching and running .count() to materialize drastically speeds up
#   relevance calculations
relevance_training = sample_master(master, config['cnc_sample_size_relevance']) \
    .select('cust', 'formatted_peapod_id', 'prdt_lvl_6_dsc', 'businessdate') \
    .cache()
formatted_rules = combined_rules.select('rule_id', 'consequent',
                                        'antecedent_prdt_lvl_6_dsc',
                                        'antecedent_formatted_peapod_id') \
    .cache()

logging.info(f'CALCULATED : {combined_rules.count()} COMBINED ASSOCIATION RULES')
logging.info(f'TRAINING RELEVANCE ON  : {formatted_rules.count()} RULES')
logging.info(f'TRAINING RELEVANCE ON  : {relevance_training.count()} TRANSACTIONS')

grid: dict = {'min_lhs_threshold': config["min_prod_lhs_threshold"],
              'min_rhs_threshold': config["min_prod_rhs_threshold"],
              'relevance_type': config["relevance_type"]}

init_params: dict = {'master': relevance_training,
                     'association_rules': formatted_rules,
                     'product_identifier': product_identifier,
                     'category_identifier': mining_category,
                     'recommendation_window_relevance': config["recommendation_window_relevance"]}

relevance = RuleRelevance(**init_params)
relevance.fit(**grid)

relevance_metrics = relevance.metrics
logging.info(f"Metrics extracted from rule relevance job: {relevance_metrics}", extra={"metrics": relevance_metrics})
# logging.info("Plots extracted from rule relevance job", extra={"plots": rel_gs.best_model.get_plots()})

'''
COMPUTE AVAILABILITY
'''
# Compute availability of the rules
rule_availability = ProductAvailabilityTransformer(spark=spark, product_column="podId")
rule_availability.fit()
combined_rules_with_availability = rule_availability.transform(combined_rules,
                                                               column_name="consequent")

# Compute availability of the backfill
cat_ranking = rule_availability.transform(cat_ranking, column_name="formatted_peapod_id", unpack_join_column=False)

logging.info(f"Metrics extracted from rule availability job: {rule_availability.metrics}",
             extra={"metrics": rule_availability.metrics})

'''
TRAIN ALS MODEL 
'''

# Convert relevance scores to % scaled index to be more effective in adjusting relevance in prediction function
relevance: DataFrame = relevance.relevance \
    .withColumn('relevance', F.col('relevance') / relevance.metrics['relevance_mean'])
relevance = relevance.checkpoint().cache()

relevance.write.mode('overwrite') \
    .parquet('/mnt/adl-fiona-cdm-rbs/HyperPersonalization/XSellModelCheckpoints/relevance.parquet')

combined_rules_with_availability.write.mode('overwrite') \
    .parquet('/mnt/adl-fiona-cdm-rbs/HyperPersonalization/XSellModelCheckpoints/combined_rules.parquet')

# Only printing count to ensure the cached dataframe is materialized before training ALS
logging.info(f'Found {relevance.count()} relevance scores')

als = ALSModel(relevance=relevance,
               product_identifier=product_identifier,
               category_identifier=mining_category,
               spark=spark,
               seed=seed,
               min_customer_relevance_count=config['min_customer_relevance_count'])
als.format_data()

ranks = [20]
maxIter = [10, 15]
regParams = [.01, .05]
alphaValues = [1]

als.tune_model(maxIter=maxIter,
               regParams=regParams,
               ranks=ranks,
               alpha_values=alphaValues)

als.fit_model()

als.create_latent_vectors(combined_rules_with_availability)
als.calculate_metrics()


als.user_factors.write.mode('overwrite') \
    .parquet('/mnt/adl-fiona-cdm-rbs/HyperPersonalization/XSellModelCheckpoints/als_user_factors.parquet')
als.item_factors.write.mode('overwrite') \
    .parquet('/mnt/adl-fiona-cdm-rbs/HyperPersonalization/XSellModelCheckpoints/als_item_factors.parquet')


logging.info(f"Metrics extracted from ALS job : {als.metrics}", extra={"metrics": als.metrics})
logging.info(f"Final params calculated, with {als.latent_cust.count()} customers and {als.latent_rule.count()} rules")

'''
Personalize customers that didn't make it to ALS
'''
#all_customers_with_als = all_customers.withColumn("cust_nbr", F.col("formatted_customer_id")).join(als.latent_cust, "cust_nbr", "left")
#als_backfill = CustomerBackfill(als_object=als)
#all_customers_with_filled_als = als_backfill.transform(all_customers_with_als)


'''
Build tree for rules collection
'''
import pickle
rule_tree = build_tree(als.latent_rule)
# TODO: Also write to blob storage/ADLS for azure function access.
try:
    base_path = "/dbfs/FileStore/tables/crosssell_tree"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    with open(os.path.join(base_path, f"{run_id}.pkl"), "wb") as f:
      pickle.dump(rule_tree, f)
except OSError as e:
    logging.error(f"A handled exception occured while writing out tree pickle: {e}")

'''
Write collections into CosmosDB
'''
als.latent_cust.write.mode("overwrite") \
    .parquet(f'/mnt/adl-fiona-cdm-rbs/HyperPersonalization/XSellModelCheckpoints/{run_id}/als_latent_cust.parquet')

als.latent_rule.write.mode("overwrite") \
    .parquet(f'/mnt/adl-fiona-cdm-rbs/HyperPersonalization/XSellModelCheckpoints/{run_id}/als_latent_rule.parquet')

# in a folder in dbfs named as args.repo_head create a file named as args.job_id.txt and add the run_id value into it. 
# This would be picked up by the pipeline and once read the folder would be deleted. 
# Do this only if write to cosmos was enabled as the Function that would be updated should point to the new collection
if args.write_to_cosmos:
    logging.info(f"Updating run id {run_id} to file dbfs:/folder/{args.job_id}.txt")        
    # args.config_path is of format /dbfs/folder/file.json, which has to be converted to dbfs:/folder/job_id.txt to update via dbutils
    runid_file = os.path.dirname(args.config_path) # this will return the /dbfs/folder
    runid_file = runid_file[:5] + ':' + runid_file[5:] + '/' + args.job_id + '.txt' # this should return /dbfs:/folder/<jobid>.txt
    dbutils.fs.put(runid_file[1:], str(run_id),  True)

if args.write_to_cosmos:
    mongo = Mongo(run_id=run_id, config=config, use_spark_connector=True)
    mongo.connect(overwrite_collections=args.overwrite_cosmos_collections)

    logging.info("Writing final trained model parameters to cosmos")
    logging.info(f'Data Size : customer : {als.latent_cust.count()} Rules: {als.latent_rule.count()} Items: {cat_ranking.count()}')
    mongo.push_objects_to_collection(objects=als.latent_cust.coalesce(8), mongo_collection=mongo.customers_collection)
    mongo.push_objects_to_collection(objects=als.latent_rule.orderBy(F.col('rule_id').desc()).repartition(20), mongo_collection=mongo.rules_collection)

    mongo.push_objects_to_collection(objects=cat_ranking.coalesce(8), mongo_collection=mongo.items_collection)

    mongo.reduce_ru_offers()

logging.info(f"Writing to cosmos is enabled {args.write_to_cosmos}", extra={"metrics": {"Cosmos_Writing_enabled" : args.write_to_cosmos, "Cosmos_Code_Completed" : "True"}})

complete_current_run(experiment_name="crosssellpoc", run_id= run_id)

