package org.example.mleapDemo

import java.io.File

import ml.combust.bundle.BundleFile
import ml.combust.bundle.serializer.SerializationFormat
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression}
import org.apache.spark.ml.feature.{CountVectorizer, HashingTF, Tokenizer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import resource.managed
import ml.combust.mleap.spark.SparkSupport._
import org.apache.spark.ml
object LogregExample {


  def createLRPipeline(dataFrame: DataFrame): PipelineModel = {
    // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val cvModel = new CountVectorizer()
      .setVocabSize(10)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    val dt = new DecisionTreeClassifier().setFeaturesCol(cvModel.getOutputCol)

    val lr = new LogisticRegression()
      .setFeaturesCol(cvModel.getOutputCol)
      //.setMaxIter(10)
      //.setRegParam(0.001)
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, cvModel, lr))

    pipeline.fit(dataFrame)
  }

  def writePipeline(pipelineModel: PipelineModel, dataFrame: DataFrame, fileName: String): Unit = {
    val file = new File(fileName)
    file.getParentFile.mkdirs
    val sbc = SparkBundleContext().withDataset(pipelineModel.transform(dataFrame))
    for (bundle <- managed(BundleFile(file))) {
      pipelineModel.writeBundle.format(SerializationFormat.Json).save(bundle)(sbc)
      //parkPipelineLr.writeBundle.save(bf)
    }

  }

  def deserializeModel(fileName: String): ml.Transformer = {
    //val zipBundle = (for (bundle <- managed(BundleFile(new URI("./data/mleap-examples/simple-json.zip")))) yield {
    //  bundle.loadSparkBundle().get
    //}).opt.get

    val zipBundle = (for (bundle <- managed(BundleFile(new File(fileName)))) yield {
      bundle.loadSparkBundle().get
    })


    zipBundle.opt.get.root
  }


  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("lr-demo").master("local[4]").getOrCreate()


    // Prepare training documents from a list of (id, text, label) tuples.
    val training = spark.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0)
    )).toDF("id", "text", "label")

    val modelName = "./data/mleap-examples/lr-json.zip"

    val pipelineModel = createLRPipeline(training)

    writePipeline(pipelineModel, training, modelName)

    val model = deserializeModel(modelName)

    val test = spark.createDataFrame(Seq(
      (4L, "spark i j k"),
      (5L, "l m n"),
      (6L, "spark hadoop spark"),
      (7L, "apache hadoop")
    )).toDF("id", "text")

    model.transform(test).show(4, false)
    println(model.transform(test).count())
    // Make predictions on test documents.
    println(

      model.transform(test)
        .select("id", "text", "prediction")
        .collect().mkString)


  }

}
