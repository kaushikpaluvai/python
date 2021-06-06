import logging

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import col, dense_rank, countDistinct, mean, first, coalesce
from pyspark.sql.functions import collect_set, size, lit

from helpers.logging_helpers.logging_helpers import df_to_metrics_dict
from testing.sampled_test_data import generate_sampled_data
from testing.synthetic_test_data import generate_synthetic_data


def initialize_spark(set_config=True) -> SparkSession:
    spark = SparkSession.builder.getOrCreate()
    # On initial runs, set config plus clear cache.
    if set_config:
        # TODO: Confirm this actually works PEOP-61
        spark.catalog.clearCache()
        spark.conf.set("spark.sql.debug.maxToStringFields", 1000)
        spark.conf.set("spark.debug.maxToStringFields", 1000)
        spark.conf.set(" spark.driver.maxResultSize", "6g")
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
        spark.conf.set("spark.sql.broadcastTimeout", 500)
    return spark


def build_data(spark: SparkSession,
               channels: [str],
               banner_id: int,
               max_transaction_size: int,
               min_transaction_size: int,
               backfill_category: str,
               mining_category: str,
               product_identifier: str,
               training_dates,
               use_synth_test_data=False,
               use_sample_test_data=False) -> dict:
    # Logic allowing customer master/product dataframes to be passed in, this is to mock this data for testing.
    if use_synth_test_data:
        master, product = generate_synthetic_data(spark=spark)
    else:
        if use_sample_test_data:
            master = generate_sampled_data(spark=spark)
        else:
            master = spark.read.parquet("/mnt/adl-fiona-cdm-rbs/HyperPersonalization/Item_Tlog/")
        product = spark.read.parquet("/mnt/adl-fiona-cdm-rbs/HyperPersonalization/HP_Product/")

    # Load product data, filter for banner. Removed discontinued products and unwanted categories.
    #   Unwanted categories include entries for rewards points, deposits, copient adjustments, and other
    #   items that appear in a transaction, but are not tied to a product.
    # CODE_REVIEW: discontnu_ind vs delete_date (Abhishek brought this up), Jay remembers using discontnu_ind
    product = product.filter((col("upc_dsc") != "UNKNOWN") &
                             (col("banner_id") == banner_id) &
                             (col("fig_cd") == 4) &
                             (col("discntnu_ind") == 0) &
                             (~col("prdt_lvl_3_dsc").isin(
                                 ['POS SUBD CIGARETTES', 'POS SUBD GASOLINE', 'POS SUBD PRESCRIPTIONS'])) &
                             (~col("prdt_lvl_4_dsc").isin(
                                 ['GROCERY OTHER MISC', 'GM OTR MISC', 'NON ADD MISC. SERV 855'])) &
                             (~col("prdt_lvl_5_dsc").isin(['MEAT OTHR MISC', 'BR WN MISC'])))
    # We couldn't come up with a reason why this would have multiples.
    # TODO: Calvin to confirm pod_ids with duplicates, to see if there still are any duplicates
    product = product.groupby("it_fig_pod_id").agg(
        *[first(col(c)).alias(c) for c in product.columns if c != "it_fig_pod_id"])
    logging.info(f"Count of product data before joining with tlog: {product.count()}")
    # Format master dataframe to be used by all future calculations
    # prdt_lvl_4_dsc is used to backfill products with not recommendations with top products in a category
    # prdt_lvl_6_dsc is intended to be used to find associations between low level categories instead of UPC/PODID
    #   This can be used to improve coverage.
    #   For example, instead of 1 pint st rawberries and 2 pint strawberries having specific prod_rules associating them with
    #   1 pint blueberries, we can simple calculate the relationship between strawberries and blueberries as a category.
    # TODO: look into shuffling from dropDuplicates() DONE
    # CODE_REVIEW: because of sequence number, same transaction shows up multiple times, dropDuplicates()
    # can probably be removed
    # CODE_REVIEW: the only pkg_pod_id records are those with peapod attributes, some products will only have it_fig_pod_id
    # TODO; review the join on clause for pkg_pod_id vs it_fig_pod_id
    master = master.filter(col('Channel').isin(channels)) \
        .filter(col("banner_id") == banner_id) \
        .filter(col("businessdate").isin(training_dates)) \
        .alias('trans') \
        .join(product.alias('prod'),
              on=col(f'trans.formatted_peapod_id') == col(f'prod.it_fig_pod_id'),
              how="inner") \
        .select("Channel",
                "formatted_customer_id",
                "formatted_order_id",
                "formatted_peapod_id",
                backfill_category,
                mining_category,
                col(f'trans.formatted_upc')) \
        .withColumnRenamed("formatted_customer_id", "cust")
    logging.info(f"Count of tlog after joining in product data: {master.count()}")

    # If store transactions are included in the prod_model, this step makes sure
    # that only online prods are available for modeling.
    if 'Store' in channels:
        logging.debug("Store found in channels, filtering out store-only products/customers")
        online_prods = master.filter(col('Channel') == 'CnC') \
            .select(col(product_identifier).alias('online_prods')) \
            .dropDuplicates()
        master = master.join(online_prods, col(product_identifier) == col('online_prods'), "inner")

        # Also mark customers that have no online transactions. These customers' data will be used to mine rules,
        # however these customer will be filtered out before relevance calculation + ALS to avoid wasting
        # computational resources on them, since they will never be used for prediction (since they don't shop online)
        online_customers = master.withColumn("in_store", (col("Channel") == "Store").astype("int")) \
            .groupBy("cust").agg(mean("in_store").alias("in_store_percentage")) \
            .filter(col("in_store_percentage") < 1.0) \
            .withColumn("online_customer", lit(True)) \
            .select(col("cust").alias("online_cust_nbr"), col("online_customer"))
        master = master.join(online_customers, master.cust == online_customers.online_cust_nbr, "left")
    else:
        master = master.withColumn("online_customer", lit(True))
    logging.info(f"Count of master after online products: {master.count()}")
    # Backfill prod_rules for products without prod_rules. This needs to be calculated after removing
    #   store upcs that are not online.
    cat_ranking = master.groupBy(backfill_category, product_identifier) \
        .agg(countDistinct("formatted_order_id").alias("count")) \
        .alias('cats') \
        .withColumn('rank',
                    dense_rank()
                    .over(Window.partitionBy(col(f'cats.{product_identifier}'))
                          .orderBy(col(f'cats.{backfill_category}')))) \
        .filter(col('rank') == 1) \
        .withColumn('rank',
                    dense_rank()
                    .over(Window.partitionBy(col(f'cats.{backfill_category}'))
                          .orderBy(col('cats.count').desc()))) \
        .drop(col('count'))
    cat_ranking.cache()

    # Calculate category ranking metrics
    prods_per_category = cat_ranking.groupBy(backfill_category).agg(
        countDistinct(product_identifier).alias("product_count"))
    cat_metrics = df_to_metrics_dict(prods_per_category,
                                     ["product_count"],
                                     optional_prefix="per_category_",
                                     count_name="n_categories_for_backfill")

    # Build category mapping table to tie products to category rules
    # Rank is used to map category antecedent products to top n consequent category's products
    # TODO: make sure this rank is still needed, and that products don't have a unique category already
    prod_mapping = master.groupBy(mining_category, product_identifier) \
        .agg(countDistinct("formatted_order_id").alias("count")) \
        .alias('cats') \
        .withColumn('rank',
                    dense_rank()
                    .over(Window.partitionBy(col(f'cats.{mining_category}'))
                          .orderBy(col(f'cats.{mining_category}')))) \
        .filter(col('rank') == 1) \
        .withColumn('rank',
                    dense_rank()
                    .over(Window.partitionBy(col(f'cats.{mining_category}'))
                          .orderBy(col('cats.count').desc()))) \
        .drop(col('count')) \
        .select(mining_category, product_identifier, 'rank') \
        .dropDuplicates()
    prod_mapping.cache()

    # Find transaction IDs that meet the transaction size criteria
    master = master.groupBy('Channel', 'cust', 'formatted_order_id', "online_customer") \
        .agg(collect_set(mining_category).alias(mining_category),
             collect_set(product_identifier).alias(product_identifier)) \
        .filter((size(product_identifier) > min_transaction_size) &
                (size(product_identifier) < max_transaction_size)) \
        .withColumn('online_customer', coalesce('online_customer', lit(False)))
    logging.info(f"Count of master after transaction size filter: {master.count()}")

    master.cache()

    return {'master': master,
            'cat_ranking': cat_ranking,
            'prod_mapping': prod_mapping,
            'cat_metrics': cat_metrics}


def sample_master(master, n):
    total_count = master.count()
    online = master.filter(col("online_customer") == lit(True))
    online_count = online.count()
    if total_count <= n:
        print('Sample size larger than count of records. Keeping all data.')
        return master
    if online_count > n:
        sample_perc = n / online_count
        sample = online.sample(withReplacement=False,
                               fraction=sample_perc,
                               seed=12345)
        return sample
    if online_count < n:
        store = master.filter(col("online_customer") == lit(False))
        store_count = store.count()
        sample_size = n - online_count
        sample_perc = sample_size / store_count

        sample = store.sample(withReplacement=False,
                              fraction=sample_perc,
                              seed=12345)
        sample = sample.union(online)
        return sample
