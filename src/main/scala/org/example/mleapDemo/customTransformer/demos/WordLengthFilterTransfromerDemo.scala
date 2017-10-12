package org.example.mleapDemo.customTransformer.demos

import org.apache.spark.ml.feature.{CountVectorizer, StopWordsRemover, Tokenizer}
import org.apache.spark.ml.mleap.feature.WordLengthFilter
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.example.mleapDemo.util.MleapUtil

object WordLengthFilterTransfromerDemo {
  def createPipeline(dataFrame: DataFrame): PipelineModel = {

    val tokenizer = new Tokenizer().setInputCol("content").setOutputCol("words")

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("words_filtered")

    val cv = new CountVectorizer().setInputCol(remover.getOutputCol).setOutputCol("features").setVocabSize(50000)

    val filterWords = new WordLengthFilter().setInputCol(remover.getOutputCol).setOutputCol("filteredWords").setWordLength(3)


    new Pipeline().setStages(Array(tokenizer, remover, cv, filterWords)).fit(dataFrame)
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("lr-demo").master("local[4]").getOrCreate()


    // Prepare training documents from a list of (id, text, label) tuples.
    val training = spark.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0)
    )).toDF("id", "content", "label")

    val modelName = "./data/mleap-examples/word-filter-json.zip"

    val pipelineModel = createPipeline(training)

    MleapUtil.writePipeline(pipelineModel, training, modelName)

    val model = MleapUtil.deserializeModel(modelName)

    val test = spark.createDataFrame(Seq(
      (4L, "spark asd"),
      (5L, "hadoop 1 2 3 4 5 lamer"),
      (6L, "apache k l m n o p q valami meg")
    )).toDF("id", "content")

    model.transform(test).show(4, false)
    println(model.transform(test).count())
    // Make predictions on test documents.
    println(

      model.transform(test)
        .select("id", "content", "filteredWords")
        .collect().mkString)
  }

}
