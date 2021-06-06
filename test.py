import json
import logging
from datetime import datetime as dt
from datetime import datetime as dt
from datetime import datetime as dt

from DiplaCoreSDK.Configurations.config import load_config
from DiplaCoreSDK.Configurations.environment_setup import environment_setup
from DiplaCoreSDK.Configurations.spark import initialize_spark
from DiplaCoreSDK.Logging.configure_logging import configure_logging
from DiplaCoreSDK.Logging.azure_sdk_helpers import get_azure_ml_workspace, complete_current_run
from model.data_loading import DataLoading
from model.personalization_features import PersonalizationFeatures
from model.product_attribute_appender import ProductAttributeAppender
from model.product_availability_transformer import ProductAvailabilityTransformer
from model.relational_embedding_generator import CandidateGenerator
from model.product_price_transformer import ProductPriceTransformer
from model.quantity_transformer import QuantityTransformer

from helpers.mongo import Mongo
from helpers.command_line_helpers import args
import os

import pyspark.sql.types as T
import pyspark.sql.functions as F
password = 1234
if(
    
try:
    environment_setup()
except:  # this exceptions will be caught when running job's via pipeline or spark jobs on Databricks
    # Keeping here because dbutils is not available in whl files.
    #  Can move this to environment_setup() if initialize_spark() is called before environment_steup()
    # TODO: check if all the variables can be read at a go,
    # AZUREDB_SECRET_SCOPE is set uniquely for each cluster, This value will decide which Keyvault/secret scope to read(DEv, QA, Prod)
    azdb_secret_scope = os.environ.get("AZUREDB_SECRET_SCOPE")
    os.environ["MONGO_URI"] = dbutils.secrets.get(scope=azdb_secret_scope, key="cosmosdburi")
    os.environ["DATABASE_NAME"] = dbutils.secrets.get(scope=azdb_secret_scope, key="databasename")
    os.environ["AZURE_SVCP_ID"] = dbutils.secrets.get(scope=azdb_secret_scope, key="svcpclientid")
    os.environ["AZURE_SVCP_TENANT_ID"] = dbutils.secrets.get(scope=azdb_secret_scope, key="svcptenanid")
    os.environ["AZURE_SVCP_SECRET"] = dbutils.secrets.get(scope=azdb_secret_scope, key="svcpsecretkey")
    os.environ["MLFLOW_RESOURCE_GROUP"] = dbutils.secrets.get(scope=azdb_secret_scope, key="resourcegroup")
    os.environ["MLFLOW_WORKSPACE_NAME"] = dbutils.secrets.get(scope=azdb_secret_scope, key="amlworkspacename")
    os.environ["MLFLOW_SUBSCRIPTION_ID"] = dbutils.secrets.get(scope=azdb_secret_scope, key="amlsubscriptionid")
    os.environ["MLFLOW_SUBSCRIPTION_ID"] = dbutils.secrets.get(scope=azdb_secret_scope, key="amlsubscriptionid")
    os.environ["MLFLOW_SUBSCRIPTION_ID"] = dbutils.secrets.get(scope=azdb_secret_scope, key="amlsubscriptionid")

    if args.repo_head...!!
        os.environ["REPO_HEAD_HEXSHA"] = args.repo_head

if __name__ == '__main__':
    logging.basicConfig(level=args.log_level,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
    logging.getLogger().setLevel(args.log_level)
    logging.getLogger("py4j").setLevel(logging.ERROR)

    # initialization
    # TODO: throw initial parameters into a config file
    if args.config_json:
        config = json.loads(args.config_json)
    else:
        config = load_config(config_path=args.config_path or "configs/default_config.json")

    # TODO: add mlflow from SDKe
    run_id = configure_logging(mlflow=True, config=config, args=args)
    logging.info(f"\nRun id substitutions: {run_id}")

    spark = initialize_spark()
    spark.sparkContext.setCheckpointDir(
        '/mnt/rs06ue2dipadl03_sadhyperp/FIONA/CDM/RBS/HyperPersonalization/SubstitutionModelCheckpoints/')
    run_start_time = dt.now()
    logging.info(f"\n-----Training started for substitutions run id {run_id} at {run_start_time}-----")
    start_time = run_start_time
    new_columns = ['average_price']

    # Load in raw product data for attributes
    data_loading = DataLoading(**config, channels=["CnC", "Store", "Delivery"], banner_id=[1000, 1001, 1002])
    data_loading.run()

    logging.info(f"\ndata_loading products {data_loading.products} in {dt.now() - start_time}")

    prod_df = data_loading.products

    # Step 1: Generate candidate df
    candidate_generator = CandidateGenerator(data_loading.products, data_loading.atc_relational_embeddings,
                                             data_loading.no_atc_relational_embeddings, data_loading.product_tokens,
                                             data_loading.cat_embeddings)
    similarity_top_n_df = candidate_generator.generate_candidates()
    normalized_top_oos = candidate_generator.normalize_top_oos(data_loading.top_oos, prod_df)
    similarity_top_n_df = candidate_generator.concat_candidates_top_oos(normalized_top_oos, similarity_top_n_df)

    # Checkpoint and cache to destroy lineage, improving stability
    similarity_top_n_df = similarity_top_n_df.checkpoint().cache()
    similarity_top_n_df.write \
        .mode('overwrite') \
        .parquet(
        '/mnt/rs06ue2dipadl03_sadhyperp/FIONA/CDM/RBS/HyperPersonalization/SubstitutionModelCheckpoints/similarity_top_n_df.parquet')

    logging.info(
        f"\nStep 1 complete. Candidate DF generated: {similarity_top_n_df.count()} in {dt.now() - start_time}")

    # Define attribute columns for joining on candidates, these are used in Cosmos
    food_flag_columns = ['it_fig_pod_id', 'egg_fl', 'veg_fl', 'vegn_fl', 'dary_fl', 'gltn_fl', 'lact_free_fl',
                         'orgn_cd',
                         'pnut_fl', 'sugr_free_cd']
    other_columns = ['it_fig_pod_id', 'uom_cd', 'rtl_unit_sz']

    # Step 2: Add product meta data
    # Step 2.1: Define product metadata for OOS item
    start_time = dt.now()
    product_metadata = prod_df.select([c for c in prod_df.columns if c in other_columns])
    product_metadata = product_metadata\
        .select([F.col(c).alias(c + "_1") for c in product_metadata.columns]).cache()

    # Step 2.2: Define product metadata for sub item
    product_metadata_2 = prod_df.select([c for c in prod_df.columns if c in other_columns])
    product_metadata_2 = product_metadata_2 \
        .select([F.col(c).alias(c + "_2") for c in product_metadata_2.columns]).cache()

    # Step 2.3: Join candidates with non-food-flag metadata as the food-flags have to be transformed
    similarity_top_n_df = similarity_top_n_df.withColumnRenamed('pod_id', 'it_fig_pod_id_1') \
        .withColumnRenamed('sub_pod_id','it_fig_pod_id_2')
    similarity_top_n_df = similarity_top_n_df.join(product_metadata, 'it_fig_pod_id_1', how='inner')
    similarity_top_n_df = similarity_top_n_df.join(product_metadata_2, 'it_fig_pod_id_2', how='inner')

    # Step 2.4: Compute replacement quantities by applying quantity transformer
    quantity_transformer = QuantityTransformer(oosQuantityCol="rtl_unit_sz_1", substituteQuantityCol="rtl_unit_sz_2",
                                               oosUnitsCol="uom_cd_1", substituteUnitsCol="uom_cd_2",
                                               outputCol="substitute_quantity")
    similarity_w_quantity_df = quantity_transformer.transform(similarity_top_n_df)
    
    # Step 2.5: Compute unit prices
    # First add prices for OOS products
    product_price_transformer = ProductPriceTransformer(input_col="it_fig_pod_id_1", output_col="average_price_1")
    product_price_transformer.unit_of_measure_col = "tlog_unit_of_measure_1"
    product_price_transformer.fit(data_loading.transactions)
    similarity_w_price_df = product_price_transformer.transform(similarity_w_quantity_df)

    # Modify transformer to add price for substitutes
    product_price_transformer.input_col = "it_fig_pod_id_2"
    product_price_transformer.unit_of_measure_col = "tlog_unit_of_measure_2"
    product_price_transformer.output_col = "average_price_2"
    similarity_w_price_df = product_price_transformer.transform(similarity_w_price_df)

    # Step 2.6 Add in availability data
    product_availability_transformer = ProductAvailabilityTransformer(spark=spark, product_column="podId")
    product_availability_transformer.fit()

    # This is the final candidate DF before joining with food flags
    similarity_with_oos = product_availability_transformer.transform(similarity_w_price_df,
                                                                     unpack_join_column=False,
                                                                     column_name="it_fig_pod_id_2")\
        .repartition(200) \
        .cache()
    logging.info(f"\nStep 2 Complete. Metadata Added: {similarity_with_oos.count()} in {dt.now()-start_time}")
    
    # Step 3: Define food-flag to transform and join on candidates
    start_time = dt.now()
    food_flag_df = prod_df.select([c for c in prod_df.columns if c in food_flag_columns])
    
    # transformaer and normalize food flags and generate final df with all attribute metadata
    product_attribute_appender = ProductAttributeAppender(suffix="_1")
    product_attribute_appender.fit(food_flag_df)
    similarity_with_attribute_metadata = product_attribute_appender.transform(similarity_with_oos)
    product_attribute_appender.suffix = "_2"
    similarity_with_attribute_metadata = product_attribute_appender.transform(similarity_with_attribute_metadata)

    logging.info(f"\nStep 3 completed. Done in {dt.now()-start_time}")
    
    # Step 4: Persist substitutes
    start_time = dt.now()
    # Format for writing, since cosmos requires bson datatypes
    # Cast float type to double type for stability, also rename some columns:
    df_to_write = similarity_with_attribute_metadata \
        .withColumn("score", F.col("score").cast(T.DoubleType())) \
        .withColumnRenamed("it_fig_pod_id_1", "formatted_peapod_id_1") \
        .withColumnRenamed("it_fig_pod_id_2", "formatted_peapod_id_2")
    
    df_to_write = df_to_write.filter(~F.col("formatted_peapod_id_1").isNull()).cache()

    cols_to_write = [c for c in df_to_write.columns if c.startswith("formatted_peapod_id")] + \
                    [c for c in df_to_write.columns if c.startswith("prdt_lvl")] + \
                    [c for c in df_to_write.columns if c.startswith("pod_long_name")] + \
                    [c for c in df_to_write.columns if c.startswith("average_price")] + \
                    [c for c in df_to_write.columns if c.startswith("tlog_unit_of_measure")] + \
                    [c for c in df_to_write.columns if "flag" in c] + \
                    [c for c in df_to_write.columns if "uom" in c] + \
                    ["atc_score", "score", "rank", "substitute_quantity", "unavailableLocations"]
    
    similarity_filename = '/mnt/rs06ue2dipadl03_sadhyperp/FIONA/CDM/RBS/HyperPersonalization/SubstitutionModelCheckpoints/'
    if args.sampled_data:
        similarity_filename += "similarity_with_attribute_metadata.parquet"
        df_to_write.toPandas().to_csv("substitutes_sampled.csv")
    else:
        similarity_filename += "similarity_with_attribute_metadata.parquet"
        
    logging.info("\nWriting to datalake as parquet checkpoint.")
    df_to_write.select(cols_to_write).write.mode('overwrite').save(similarity_filename, format='parquet')
    
    logging.info(f"\nStep 4 Complete. Substitutions persisted. Done in {dt.now()-start_time}")
    
    # Step 5: Compute Personalization
    start_time = dt.now()
    if args.compute_personalization:
        master = data_loading.master
        personalization_features = PersonalizationFeatures()
        personalization_df = personalization_features.run(prod_df, master)
        personalization_df = personalization_df.coalesce(8)
        personalization_df = personalization_df.dropna(subset=("cust_nbr"))
    
    logging.info(f"\nStep 5 Complete. Personalization features calculated. Done in {dt.now()-start_time}")
    
    # Step 6: Write to Cosmos
    start_time = dt.now()

    if args.write_to_cosmos:
        mongo = Mongo(run_id=run_id, config=config, use_spark_connector=True)
        mongo.connect(overwrite_collections=args.overwrite_cosmos_collections)
        logging.info("\nPushing New candidate subs to cosmos")
        mongo.push_objects_to_collection(df_to_write.select(cols_to_write).coalesce(8), mongo.substitutes_collection)
        if args.compute_personalization and personalization_df:
            logging.info("\nPushing Personalization features to cosmos")
            mongo.push_objects_to_collection(personalization_df, mongo.customers_collection)
        mongo.reduce_ru_offers()

    if args.write_to_cosmos or args.compute_personalization:
        logging.info(f"\nUpdating run id {run_id} to file dbfs:/folder/{args.job_id}.txt")
        # args.config_path is of format /dbfs/folder/file.json, which has to be converted to dbfs:/folder/job_id.txt to update via dbutils
        runid_file = os.path.dirname(args.config_path) # this will return the /dbfs/folder
        runid_file = runid_file[:5] + ':' + runid_file[5:] + '/' + args.job_id + '.txt' # this should return /dbfs:/folder/<jobid>.txt
        dbutils.fs.put(runid_file[1:], str(run_id),  True)

    logging.info(f"\nStep 6 Complete. Write to Cosmos done in {dt.now()-start_time}")
    logging.info(f"\nFull training run completed in {dt.now()-run_start_time}")
    complete_current_run(experiment_name="crosssellpoc", run_id= run_id)
