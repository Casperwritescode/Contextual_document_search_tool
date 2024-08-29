from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, input_file_name, monotonically_increasing_id, current_timestamp, regexp_extract
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType

# Create a SparkSession
spark = SparkSession.builder.getOrCreate()

# Define the directory path
directory_path = "/path/documents/txt_docs"

# Read the documents into an RDD with wholeTextFiles
rdd = spark.sparkContext.wholeTextFiles(directory_path)

# Convert the RDD to a DataFrame
doc_text = rdd.toDF(["identifier", "value"])

# Read the .txt files from the directory using wholeTextFiles
doc_text = doc_text.withColumn("file_name", regexp_extract("identifier", r'[^/]+$', 0))

# Add a column with the file name as identifier/metadata
doc_text = doc_text.withColumnRenamed("identifier", "id")
doc_text = doc_text.withColumn("id", monotonically_increasing_id())
doc_text = doc_text.withColumn("timestamp", current_timestamp())

display(doc_text)