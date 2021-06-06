import json
import os
import ssl
import time

import bson
import pymongo
import pyspark

from helpers.config import load_config
from DiplaCoreSDK.Configurations.environment_setup import environment_setup
from DiplaCoreSDK.Configurations.spark import initialize_spark


class Mongo:
    def __init__(self, run_id="no-run-id", config={}, use_spark_connector = False):
        self.database = None
        self.rules_collection = None
        self.items_collection = None
        self.customers_collection = None
        self.run_id = run_id
        self.cosmos_client = None
        self.database_client = None
        self.container_client = None
        self.use_spark_connector = use_spark_connector
        self.config = config

        self.rules_collection = None
        self.items_collection = None
        self.customers_collection = None

    def connect(self, overwrite_collections=False):
        con = pymongo.MongoClient(os.environ["MONGO_URI"],
                                  retryWrites=False,
                                  ssl_cert_reqs=ssl.CERT_NONE)
        self.database = con[os.environ["DATABASE_NAME"]]
        self.rules_collection = self.database[f"rules_collection_{self.run_id}"]
        self.items_collection = self.database[f"items_collection_{self.run_id}"]
        self.customers_collection = self.database[f"customers_collection_{self.run_id}"]

        if overwrite_collections:
            self.delete_collections()

        # Create collections and indexing strategy
        # Note that we are using these extension commands per microsoft support thread with Calvin and Musa on 6/4/2020
        # Appearantly the python package for cosmos is not the support way of creating collections with specified shard
        # keys and indexing strategies.
        self.database.command({'customAction': 'CreateCollection',
                               'collection': f"rules_collection_{self.run_id}",
                               'autoScaleSettings': {'maxThroughput': 100000},
                               'shardKey': '_id'})

        self.database.command({'customAction': 'CreateCollection',
                               'collection': f"items_collection_{self.run_id}",
                               'autoScaleSettings': {'maxThroughput': 40000},
                               'shardKey': '_id'})
        self.items_collection.create_index([('formatted_peapod_id', pymongo.DESCENDING)])
        self.items_collection.create_index([(self.config['backfill_category'], pymongo.DESCENDING)])

        self.database.command({'customAction': 'CreateCollection',
                               'collection': f"customers_collection_{self.run_id}",
                               'autoScaleSettings': {'maxThroughput': 100000},
                               'shardKey': '_id'})
        self.customers_collection.create_index([('cust_nbr', pymongo.DESCENDING)])

    def list_collections(self):
        """Helpful operation for managing cosmos, not used in code"""
        if not self.database:
            environment_setup()
            con = pymongo.MongoClient(os.environ["MONGO_URI"], retryWrites=False, ssl_cert_reqs=ssl.CERT_NONE)
            self.database = con[os.environ["DATABASE_NAME"]]
        return self.database.collection_names()

    def drop_collections(self, collections):
        """Helpful operation for managing cosmos, not used in code"""
        if not self.database:
            environment_setup()
            con = pymongo.MongoClient(os.environ["MONGO_URI"], retryWrites=False, ssl_cert_reqs=ssl.CERT_NONE)
            self.database = con[os.environ["DATABASE_NAME"]]
        for collection in collections:
            self.database[collection].drop()

    def push_objects_to_collection(self, objects, mongo_collection):
        if self.use_spark_connector:
            assert isinstance(objects, pyspark.sql.DataFrame), "Input data should be a spark dataframe if use_spark_connector"
            objects.write.format("mongo").mode("append") \
                .option("spark.mongodb.output.uri", os.environ["MONGO_URI"]) \
                .option("database", os.environ["DATABASE_NAME"]) \
                .option("collection", mongo_collection.name).save()

        else:
            assert isinstance(objects, list), "Input data should be a list of dicts if not use_spark_connector"
            objects = self.get_json(objects)
            for i in objects:
                i.update({'_id': bson.objectid.ObjectId()})
            try:
                mongo_collection.delete_many({})
            except Exception as e:
                print(e)

            mongo_collection.insert_many(objects)

    def delete_collections(self):
        """
        Used during testing to quickly drop test collections.
        :return:
        """
        self.rules_collection.drop()
        self.items_collection.drop()
        self.customers_collection.drop()

    def get_json(self, obj):
        return json.loads(json.dumps(obj, default=lambda o: str(o)))

    def reduce_ru_offers(self):
        self.database.command({'customAction': 'UpdateCollection',
                               'collection': f"customers_collection_{self.run_id}",
                               'autoScaleSettings': {'maxThroughput': 10000}})
        self.database.command({'customAction': 'UpdateCollection',
                               'collection': f"items_collection_{self.run_id}",
                               'autoScaleSettings': {'maxThroughput': 10000}})
        self.database.command({'customAction': 'UpdateCollection',
                               'collection': f"rules_collection_{self.run_id}",
                               'autoScaleSettings': {'maxThroughput': 10000}})


if __name__ == "__main__":
    print("Testing mongo functionality")
    environment_setup()
    config = load_config()

    try:
        mongo = Mongo(run_id="testing", config=config, use_spark_connector=False)
        mongo.connect()
        print(mongo.rules_collection.index_information())
        object_to_insert = {"test": "test", "_id": bson.ObjectId()}
        print("Inserting: ", object_to_insert)
        mongo.push_objects_to_collection([object_to_insert], mongo.rules_collection)
        mongo.delete_collections()

        mongo = Mongo(run_id="testing", config=config, use_spark_connector=True)
        mongo.connect()
        print(mongo.rules_collection.index_information())
        object_to_insert = {"test": "test"}
        spark = initialize_spark(set_config=False)
        df = spark.createDataFrame([object_to_insert])
        print("Inserting: ", object_to_insert)
        mongo.push_objects_to_collection(df, mongo.rules_collection)
        mongo.delete_collections()
    except Exception as e:
        mongo.delete_collections()
        raise e
